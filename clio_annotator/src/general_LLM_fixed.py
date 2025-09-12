import io
import base64
import os
import json
import pickle
import hashlib
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

from openai import OpenAI
from pydantic import BaseModel

class FrontierScore(BaseModel):
    frontier_id: int
    score: float
    reasoning: str

# Simple disk cache implementation
class SimpleCache:
    def __init__(self, cache_dir: str = ".cache", maxsize: int = 100000):
        self.cache_dir = cache_dir
        self.maxsize = maxsize
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except:
            pass

# Helper functions

def load_json_schema(schema_name: str, path = os.path.join("base_prompts")):
    if not schema_name.endswith(".json"):
        schema_name += ".json"
    with open(os.path.join(path, schema_name), "r") as f:
        content = json.load(f)
    return content

def image_to_base64(image):
    image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def add_image_to_prompt(prompt: str, image: Image, image_description: str = None):
    if image_description is None:
        image_description = "\nAlso consider the given image."

    if isinstance(prompt, str):
        prompt = [{"type": "text", "text": prompt}]

    if image is not None:
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_to_base64(image)}"
            }
        }
        
        text_content = {"type": "text", "text": image_description}
        prompt = prompt + [text_content, image_content]

    return prompt

def load_prompt(prompt: str, path = os.path.join("base_prompts", "AEG")) -> str:
    if not prompt.endswith(".txt"):
        prompt += ".txt"

    with open(os.path.join(path, prompt), "r") as f:
        content = f.read()
    return content

# Base class for LLM agents
# TODO: decide on consistent return structure do we return our results or directly update the scene graph?
class LLMAgent:
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.8,
                 prompt_path: str = os.path.join("base_prompts"),
                 use_cache: bool = True,
                 ):

        self.client = OpenAI(
            organization=os.environ["OPENAI_ORG"],
            #project=os.environ["OPENAI_PROJECT"],
        )
        
        self.cache = SimpleCache() if use_cache else None
        self.model = model
        self.temperature = temperature
        self.prompt_path = prompt_path

        try:
            self.system_prompt = load_prompt("system_prompt", prompt_path)
        except FileNotFoundError:
            self.system_prompt = "You are a helpful assistant."
            # raise warning that the system prompt was not found
            print(f"Warning: System prompt not found in {prompt_path}. Using default system prompt. {self.system_prompt}")

    def _call_openai_with_schema(self, messages: List[Dict], response_schema: Dict) -> Dict:
        """Call OpenAI API with function calling for structured output"""
        
        # Create cache key if caching is enabled
        cache_key = None
        if self.cache:
            cache_data = {
                'messages': messages,
                'model': self.model,
                'temperature': self.temperature,
                'schema': response_schema
            }
            cache_key = self.cache._get_cache_key(**cache_data)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Define the function for structured output
        function_def = {
            "name": "provide_structured_response",
            "description": "Provide a structured response according to the specified schema",
            "parameters": response_schema
        }
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=[function_def],
            function_call={"name": "provide_structured_response"},
            temperature=self.temperature
        )
        
        # Parse the function call response
        function_call = response.choices[0].message.function_call
        result = json.loads(function_call.arguments)
        
        # Cache result if caching is enabled
        if self.cache and cache_key:
            self.cache.set(cache_key, result)
        
        return result

    def _call_openai_with_iterable_schema(self, messages: List[Dict], response_model) -> List[Dict]:
        """Call OpenAI API expecting multiple items of the same schema"""
        
        # For FrontierScore, we expect multiple scores
        if response_model == FrontierScore:
            schema = {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "frontier_id": {"type": "integer"},
                                "score": {"type": "number"},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["frontier_id", "score", "reasoning"]
                        }
                    }
                },
                "required": ["scores"]
            }
            
            result = self._call_openai_with_schema(messages, schema)
            return result.get("scores", [])
        
        return []

    def annotate_scene_graph(self, scene_graph, **kwargs):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def identify_misplaced_object(self, scene_graph, expanded_nodes, **kwargs):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def generate_placements(self, scene_graph, rearrange_object, **kwargs):
        raise NotImplementedError("This method should be implemented in a subclass.")

    # TODO: adapt from AEG
    def score_positions(self, scene_graph, navigation, frontiers, object, image=None):
        base_prompt = (f"Score the following frontiers for how likely it is that a matching receptacle for the object "
                       f"{object} can be found there when exploring. The scores should be between 0 and 1, where 0 means that there is no "
                       f"matching receptacle and 1 means that there is a matching receptacle. Think about the objects that you may "
                       f"find in the unexplored area around each frontier considering the other objects in the scene and what is "
                       f"nearby the frontier in question. Return a score for EACH frontier and ONLY the mentioned frontiers in the same order as they are given.\n")

        if image is not None:
            base_prompt += "Also consider the given image of the object.\n"

        content = base_prompt
        
        # Prepare messages with image support
        user_content = []
        if image is not None:
            try:
                user_content = [
                    {"type": "text", "text": content + f"\n The {object} is shown in the given image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_to_base64(image)}"
                        }
                    }
                ]
            except:
                user_content = [{"type": "text", "text": content}]
        else:
            user_content = [{"type": "text", "text": content}]

        free_area, frontiers = zip(*frontiers)
        frontiers = np.array(frontiers)
        frontiers = navigation.mapper.get_environment_positions(frontiers)
        frontiers = list(zip(frontiers, free_area))
        
        # Add scene graph to content
        if isinstance(user_content[0], dict) and user_content[0]["type"] == "text":
            user_content[0]["text"] += "\nConsider the following scene graph:\n" + scene_graph.textify("nl", frontiers=frontiers)
        else:
            user_content.append({
                "type": "text", 
                "text": "\nConsider the following scene graph:\n" + scene_graph.textify("nl", frontiers=frontiers)
            })

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            position_scores_data = self._call_openai_with_iterable_schema(messages, FrontierScore)
            
            # Convert to FrontierScore objects
            position_scores = [FrontierScore(**score_data) for score_data in position_scores_data]
            
            # sort by frontier id
            position_scores = sorted(position_scores, key=lambda x: x.frontier_id)
        except Exception as e:
            print(f"Error getting position scores: {e}")
            return np.zeros(len(frontiers))

        ret = np.zeros(len(frontiers))
        for position_score in position_scores:
            # print(f"Position {position_score.frontier_id} - Score: {position_score.score} - Reasoning: {position_score.reasoning}")
            if position_score.frontier_id < 0 or position_score.frontier_id >= len(frontiers):
                # print(f"Warning: Position score for frontier {position_score.frontier_id} is out of bounds. Returning zero scores.")
                return np.zeros(len(frontiers))
            ret[position_score.frontier_id] = position_score.score

        return ret
