import io
import base64
import os
import json

import numpy as np
from PIL import Image

from openai import OpenAI
import instructor
from instructor.cache import AutoCache, DiskCache
from pydantic import BaseModel

class FrontierScore(BaseModel):
    frontier_id: int
    score: float
    reasoning: str

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
        prompt = [prompt]

    if image is not None:
        try:
            image = instructor.Image.from_base64("data:image/png;base64," + image_to_base64(image))
        except AttributeError:
            return prompt

        prompt = prompt + [image_description, image]

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
                 ):

        self.base_client = OpenAI(
            organization=os.environ["OPENAI_ORG"],
            #project=os.environ["OPENAI_PROJECT"],
        )
        self.client = instructor.from_openai(
            self.base_client,
            cache=DiskCache(maxsize=100000) # AutoCache(maxsize=100000) # TODO: controll via arguments.
        )

        self.model = model
        self.temperature = temperature

        self.prompt_path = prompt_path

        try:
            self.system_prompt = load_prompt("system_prompt", prompt_path)
        except FileNotFoundError:
            self.system_prompt = "You are a helpful assistant."
            # raise warning that the system prompt was not found
            print(f"Warning: System prompt not found in {prompt_path}. Using default system prompt. {self.system_prompt}")

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
        if image is not None:
            try:
                image = instructor.Image.from_base64("data:image/png;base64," + image_to_base64(image))
                content = [
                    content + f"\n The {object} is shown in the given image.", image
                ]
            except AttributeError:
                pass

        free_area, frontiers = zip(*frontiers)
        frontiers = np.array(frontiers)
        frontiers = navigation.mapper.get_environment_positions(frontiers)
        frontiers = list(zip(frontiers, free_area))
        content += "\nConsider the following scene graph:\n" + scene_graph.textify("nl", frontiers=frontiers)

        position_scores = self.client.chat.completions.create_iterable(
            model=self.model,
            response_model=FrontierScore,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ],
            temperature=self.temperature
        )
        # sort by frontier id
        try:
            position_scores = sorted(position_scores, key=lambda x: x.frontier_id)
        except:
            # print("Error sorting position scores. Returning zero scores.")
            return np.zeros(len(frontiers))

        ret = np.zeros(len(frontiers))
        for position_score in position_scores:
            # print(f"Position {position_score.frontier_id} - Score: {position_score.score} - Reasoning: {position_score.reasoning}")
            if position_score.frontier_id < 0 or position_score.frontier_id >= len(frontiers):
                # print(f"Warning: Position score for frontier {position_score.frontier_id} is out of bounds. Returning zero scores.")
                return np.zeros(len(frontiers))
            ret[position_score.frontier_id] = position_score.score

        return ret