import os.path
from typing import Union, List, Dict, Tuple
import json

import pydantic_core
import tqdm
import numpy as np

from pydantic import BaseModel, Field, AfterValidator, field_validator, model_validator
from typing_extensions import Annotated
import openai
import pydantic_core

try:
    from .general_LLM import LLMAgent, load_prompt, add_image_to_prompt
except ImportError:
    from general_LLM import LLMAgent, load_prompt, add_image_to_prompt

def valid_score(s):
    if not 0 <= s <= 100:
        raise ValueError("Score must be between 0 and 100")
    return s

Score = Annotated[Union[float, str], AfterValidator(valid_score), Field(default=0)]

class ScoreAnalysis(BaseModel):
    score: Score
    analysis: str

class PlacementSuggestion(BaseModel):
    best_place: str
    certainty: int
    analysis: str

class ReceptacleAffordanceAnalysis(BaseModel):
    geometry_and_position: str
    relationships: str
    unique_usage: str
    fine_grained_category: str

class CarriableAffordanceAnalysis(BaseModel):
    geometry_and_functionality: str
    fine_grained_category: str

class AreaAnalysis(BaseModel):
    name: str
    description: str

class SemanticEdges(BaseModel):
    receptacle: str
    related_objects: List[str] = Field(default_factory=list)
    semantic_relationships: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_related_objects_length(self):
        if len(self.related_objects) != len(self.semantic_relationships):
            raise ValueError("Length of related_objects and semantic_relationships must match")
        return self

# Adapters for dict-based graph
def get_nodes_by_attribute(graph: Dict, attribute: str, condition):
    matching = []
    for node_id, node in graph['nodes'].items():
        if condition(node.get(attribute, None)):
            matching.append(node_id)
    return matching

def get_related_nodes(graph: Dict, node_id: str, inverse=False, bidirectional=False, valid_relations=None, include_augmented_edges=False):
    related = []
    if not inverse:
        edges = graph.get('edges', {})
        augmented_edges = graph.get('augmented_edges', {}) if include_augmented_edges else {}
        for edge_key in edges:
            if isinstance(edge_key, tuple) and edge_key[0] == node_id:
                if valid_relations is None or edges[edge_key] in valid_relations:
                    related.append((edge_key[1], edges[edge_key]))
        for edge_key in augmented_edges:
            if isinstance(edge_key, tuple) and edge_key[0] == node_id:
                if valid_relations is None or augmented_edges[edge_key] in valid_relations:
                    related.append((edge_key[1], augmented_edges[edge_key]))
    else:
        edges = graph.get('edges', {})
        augmented_edges = graph.get('augmented_edges', {}) if include_augmented_edges else {}
        for edge_key in edges:
            if isinstance(edge_key, tuple) and edge_key[1] == node_id:
                if valid_relations is None or edges[edge_key] in valid_relations:
                    related.append((edge_key[0], edges[edge_key]))
        for edge_key in augmented_edges:
            if isinstance(edge_key, tuple) and edge_key[1] == node_id:
                if valid_relations is None or augmented_edges[edge_key] in valid_relations:
                    related.append((edge_key[0], augmented_edges[edge_key]))
    
    if bidirectional:
        related += get_related_nodes(graph, node_id, inverse=not inverse, valid_relations=valid_relations, include_augmented_edges=include_augmented_edges)
    
    return related

def get_related_nodes_multi_hop(graph: Dict, node_id: str, max_depth=-1, inverse=False, bidirectional=False, valid_relations=None, include_augmented_edges=False):
    def recurse(node_id, depth, inv, seen):
        rel_nodes = {}
        for rel_id, relation in get_related_nodes(graph, node_id, inverse=inv, valid_relations=valid_relations, include_augmented_edges=include_augmented_edges):
            if rel_id not in seen:
                rel_nodes[rel_id] = relation
                seen.add(rel_id)
                if max_depth > 0:
                    rel_nodes.update(recurse(rel_id, depth - 1, inv, seen))
                else:
                    rel_nodes.update(recurse(rel_id, -1, inv, seen))
        return rel_nodes
    seen = {node_id}
    return recurse(node_id, max_depth, inverse, seen)

def get_nearby_nodes(graph: Dict, reference, distance_threshold=2, node_subset=None):
    if isinstance(reference, str):
        ref_pos = graph['nodes'][reference].get('transform', np.zeros(3))[:3, 3] if 'transform' in graph['nodes'][reference] else np.zeros(3)
    else:
        ref_pos = np.array(reference)
    nearby = []
    for node_id, node in graph['nodes'].items():
        if node_subset and node_id not in node_subset:
            continue
        node_pos = node.get('transform', np.zeros(3))[:3, 3] if 'transform' in node else np.zeros(3)
        dist = np.linalg.norm(node_pos - ref_pos)
        if dist < distance_threshold:
            nearby.append((node_id, dist))
    return sorted(nearby, key=lambda x: x[1])

def get_nodes_by_type(graph: Dict, node_type: str):
    return get_nodes_by_attribute(graph, "type", lambda x: x == node_type)

def get_parent_room(graph: Dict, object_id: str):
    related = get_related_nodes(graph, object_id, inverse=True)
    for rel_id, _ in related:
        if graph['nodes'][rel_id].get('type') == "room":
            return rel_id
    return None

def get_detailed_item_description(graph: Dict, item_id: str, add_floor_remark=True):
    node = graph['nodes'][item_id]
    desc = f"ID: {item_id}\nCategory: {node.get('name', node.get('category', 'unknown'))}\n"
    attributes = []
    if node.get('pickupable', False):
        attributes += ["Pickupable"]
    if node.get('receptacle', False):
        attributes += ["Receptacle"]
    desc += f"Type: {attributes}\n"
    if add_floor_remark and node.get('type') in ["room", "area"]:
        desc += " (possibly Floor)\n"
    parent_room = get_parent_room(graph, item_id)
    desc += f"Room located: {parent_room}\n"
    fields = ["geometry_and_position", "relationships", "unique_usage", "fine_grained_category", "geometry_and_functionality"]
    for field in fields:
        if field in node:
            desc += f"\n{field.replace('_', ' ')}: {node[field]}"
    surrounding_objects = get_nearby_nodes(graph, item_id)
    surrounding_str = ", ".join([graph['nodes'][n_id].get('name', graph['nodes'][n_id].get('category', 'unknown')) for n_id, _ in surrounding_objects]) if surrounding_objects else "None"
    desc += f"\nSurrounding objects: {surrounding_str}"
    related_nodes = get_related_nodes(graph, item_id, inverse=True, bidirectional=True, include_augmented_edges=True)
    related_items = []
    for n_id, relation in related_nodes:
        fine_cat = graph['nodes'][n_id].get('fine_grained_category', graph['nodes'][n_id].get('category', 'unknown'))
        related_items.append(f"\t - {fine_cat}: {relation}")
    related_str = "\n".join(related_items) if related_items else "None"
    desc += f"\nRelated objects: {related_str}"
    return desc

class ClioAEGLLMAgent(LLMAgent):
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.8):
        prompt_path = os.path.join(os.path.dirname(__file__), "base_prompts", "AEG")
        super().__init__(model=model, temperature=temperature, prompt_path=prompt_path)

        # (pickupable, receptacle) -> (score, analysis)
        self.score_history = {}
        self.graph = {'nodes': {}, 'edges': {}, 'augmented_edges': {}}

    def _create_schema_for_model(self, model_class):
        """Convert Pydantic model to JSON schema for function calling"""
        if model_class == ReceptacleAffordanceAnalysis:
            return {
                "type": "object",
                "properties": {
                    "geometry_and_position": {"type": "string"},
                    "relationships": {"type": "string"},
                    "unique_usage": {"type": "string"},
                    "fine_grained_category": {"type": "string"}
                },
                "required": ["geometry_and_position", "relationships", "unique_usage", "fine_grained_category"]
            }
        elif model_class == CarriableAffordanceAnalysis:
            return {
                "type": "object",
                "properties": {
                    "geometry_and_functionality": {"type": "string"},
                    "fine_grained_category": {"type": "string"}
                },
                "required": ["geometry_and_functionality", "fine_grained_category"]
            }
        elif model_class == AreaAnalysis:
            return {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "description"]
            }
        elif model_class == SemanticEdges:
            return {
                "type": "object",
                "properties": {
                    "receptacle": {"type": "string"},
                    "related_objects": {"type": "array", "items": {"type": "string"}},
                    "semantic_relationships": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["receptacle", "related_objects", "semantic_relationships"]
            }
        elif model_class == ScoreAnalysis:
            return {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 100},
                    "analysis": {"type": "string"}
                },
                "required": ["score", "analysis"]
            }
        elif model_class == PlacementSuggestion:
            return {
                "type": "object",
                "properties": {
                    "best_place": {"type": "string"},
                    "certainty": {"type": "integer"},
                    "analysis": {"type": "string"}
                },
                "required": ["best_place", "certainty", "analysis"]
            }
        else:
            raise ValueError(f"Unknown model class: {model_class}")

    def annotate_object(self, object):
        attributes = []

        if self.graph['nodes'][object].get("pickupable", False):
            attributes += ["Pickupable"]
        if self.graph['nodes'][object].get("receptacle", False):
            attributes += ["Receptacle"]

        for attribute in attributes:
            local_context_prompt = load_prompt(f"local_context_{attribute.lower()}", self.prompt_path)

            surrounding_objects = get_nearby_nodes(self.graph, object)
            surrounding_objects = [self.graph['nodes'][node_id].get('category', self.graph['nodes'][node_id].get('name', 'unknown')) for node_id, distance in surrounding_objects]
            surrounding_objects = ", ".join(surrounding_objects) if surrounding_objects else "None"

            related_nodes = get_related_nodes(self.graph, object, inverse=True, bidirectional=True, include_augmented_edges=True)
            related_nodes_formatted = []
            for node_id, relation in related_nodes:
                fine_cat = self.graph['nodes'][node_id].get('fine_grained_category', self.graph['nodes'][node_id].get('category', 'unknown'))
                related_nodes_formatted.append(f"\t - {fine_cat}: {relation}")
            related_nodes = "\n".join(related_nodes_formatted) if related_nodes_formatted else "None"

            item_description = (
                f"Category: {self.graph['nodes'][object].get('category', self.graph['nodes'][object].get('name', 'unknown'))}\n"
                f"Type: {attributes}\n"
                f"Room located: {get_parent_room(self.graph, object)}\n"
                f"Surrounding objects: {surrounding_objects}\n"
                f"Related objects: {related_nodes}\n"
            )

            for additional_context in ["geometry_and_position", "relationships", "unique_usage", "fine_grained_category", "geometry_and_functionality"]:
                if additional_context in self.graph['nodes'][object]:
                    item_description += f"\n{additional_context.replace('_', ' ')}: {self.graph['nodes'][object][additional_context]}"

            item_description = add_image_to_prompt(item_description, self.graph['nodes'][object].get("image", None))

            # TODO: add key-frame

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": local_context_prompt},
                {"role": "user", "content": item_description},
            ]

            try:
                response_model = ReceptacleAffordanceAnalysis if attribute == "Receptacle" else CarriableAffordanceAnalysis
                schema = self._create_schema_for_model(response_model)
                response_data = self._call_openai_with_schema(messages, schema)
                
                for key, value in response_data.items():
                    self.graph['nodes'][object][key] = value
            except (pydantic_core._pydantic_core.ValidationError, openai.APIError):
                continue

    def annotate_scene_graph(self, query_objects = None, **kwargs):
        unannotated_object = get_nodes_by_attribute(self.graph, None, lambda node_dict: "fine_grained_category" not in node_dict)

        if query_objects is not None:
            unannotated_object = [objectId for objectId in unannotated_object if objectId in query_objects]

        # local context analysis
        for object in tqdm.tqdm(unannotated_object, desc="Annotating objects") if unannotated_object else unannotated_object:
            self.annotate_object(object)

        area_analysis_prompt = load_prompt("area_analysis", self.prompt_path)

        areas = get_nodes_by_type(self.graph, "area")

        # global context analysis
        # - label areas
        for area in tqdm.tqdm(areas, desc="Annotating areas") if areas else areas:
            area_objects = get_related_nodes_multi_hop(self.graph, area, inverse=True, bidirectional=False, valid_relations=["inside", "supported-by"])
            area_objects = [self.graph['nodes'][obj_id].get('category', self.graph['nodes'][obj_id].get('name', 'unknown')) for obj_id in area_objects]
            area_objects = ("the area contains " + ", ".join(list(area_objects))) if area_objects else "None"
            area_image = None # TODO: get an image of the area

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": area_analysis_prompt},
                {"role": "user", "content": area_objects},
                #{"role": "user", "content": area_image},
            ]
            try:
                schema = self._create_schema_for_model(AreaAnalysis)
                response_data = self._call_openai_with_schema(messages, schema)
                
                self.graph['nodes'][area]["fine_grained_category"] = response_data["name"]
                self.graph['nodes'][area]["analysis"] = response_data["description"]
            except (pydantic_core._pydantic_core.ValidationError, openai.APIError):
                continue

        # - create room summaries for each room to be used in semantic edge creation
        room_descriptions = {}
        for room in get_nodes_by_type(self.graph, "room"):
            children = get_related_nodes(self.graph, room, inverse=True, bidirectional=False, valid_relations=["inside"])
            child_areas = [child_id for child_id, relationship in children
                           if self.graph['nodes'][child_id].get("type") == "area"]
            child_objects = [child_id for child_id, relationship in children
                             if self.graph['nodes'][child_id].get("type") != "area"]
            description = f"The room is a {self.graph['nodes'][room].get('category', self.graph['nodes'][room].get('name', 'unknown'))} and contains: \n\n"
            for child_area in child_areas:
                area_objects = get_related_nodes_multi_hop(self.graph, child_area, inverse=True, bidirectional=False, valid_relations=["inside", "supported-by"])
                area_obj_list = []
                for obj_id in area_objects:
                    fine_cat = self.graph['nodes'][obj_id].get('fine_grained_category', self.graph['nodes'][obj_id].get('category', 'unknown'))
                    area_obj_list.append(f"{fine_cat} ({obj_id})")
                description += (f"{child_area} - {self.graph['nodes'][child_area].get('fine_grained_category', 'unknown')}: \n"
                                f"{self.graph['nodes'][child_area].get('analysis', '')}\n"
                                f"contains: {', '.join(area_obj_list) if area_obj_list else 'None'}\n\n")

            if child_objects:
                child_obj_list = []
                for obj_id in child_objects:
                    fine_cat = self.graph['nodes'][obj_id].get('fine_grained_category', self.graph['nodes'][obj_id].get('category', 'unknown'))
                    child_obj_list.append(f"{fine_cat} ({obj_id})")
                description += "other objects: " + ", ".join(child_obj_list) + "\n"
            room_descriptions[room] = description

        semantic_edges_prompt = load_prompt("semantic_edges", self.prompt_path)

        receptacles = get_nodes_by_attribute(self.graph, "receptacle", lambda x: x is True)

        if query_objects is not None:
            receptacles = [receptacle for receptacle in receptacles if receptacle in query_objects]

        # - go through each receptacle and add functional edges
        for receptacle in tqdm.tqdm(receptacles, desc="Adding semantic edges") if receptacles else receptacles:
            receptacle_description = get_detailed_item_description(self.graph, receptacle)

            parent_room = get_parent_room(self.graph, receptacle)

            parent_room_description = room_descriptions[parent_room] if parent_room in room_descriptions else "None"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": semantic_edges_prompt},
                {"role": "user", "content": receptacle_description},
                {"role": "user", "content": parent_room_description},
            ]

            try:
                schema = self._create_schema_for_model(SemanticEdges)
                response_data = self._call_openai_with_schema(messages, schema)

                if response_data.get("semantic_relationships") is not None:
                    self.graph['augmented_edges'].update(
                        {(receptacle, related_object): semantic_relation
                         for semantic_relation, related_object in zip(response_data["semantic_relationships"], response_data["related_objects"])})
            except (pydantic_core._pydantic_core.ValidationError, openai.APIError):
                pass

        # affordance updates
        for receptacle in tqdm.tqdm(receptacles, desc="Updating receptacles") if receptacles else receptacles:
            self.annotate_object(receptacle)

    def get_placement_score(self, scoring_prompt: str, target_description: str, reference_item_description: str, target_image = None, reference_image = None) -> Tuple[int, str]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": scoring_prompt},
            {"role": "user", "content": add_image_to_prompt("carriable object:\n" + reference_item_description,
                                                            reference_image,
                                                            "The item is pictured in the following image")},
            {"role": "user", "content": add_image_to_prompt("receptacle:\n" + target_description,
                                                            target_image,
                                                            "The item is pictured in the following image")}
        ]

        schema = self._create_schema_for_model(ScoreAnalysis)
        response_data = self._call_openai_with_schema(messages, schema)

        return response_data["score"], response_data["analysis"]

    def identify_misplaced_object(self, expanded_nodes, **kwargs):

        placement_scoring_prompt = load_prompt("placement_scoring", self.prompt_path)
        item_descriptions = {item: (get_detailed_item_description(self.graph, item), self.graph['nodes'][item].get("image", None)) for item in get_nodes_by_attribute(self.graph, "pickupable", lambda x: x is True) if expanded_nodes is None or item in expanded_nodes}
        scores = {
            item: np.mean([self.get_placement_score(placement_scoring_prompt,
                                                    target_description=get_detailed_item_description(self.graph, parent_id),
                                                    reference_item_description=item_description[0],
                                                    target_image=self.graph['nodes'][parent_id].get("image", None),
                                                    reference_image=item_description[1])[0]
                           for parent_id, _ in get_related_nodes(self.graph, item, inverse=False, valid_relations=["inside", "supported-by"])])
            for item, item_description in tqdm.tqdm(item_descriptions.items(), desc="Identifying misplaced objects")
        }
        misplaced_items = [item for item, score in scores.items() if score < 50]  # Assuming a threshold of 50 for misplaced items
        print(f"Identified {len(misplaced_items)} misplaced items: {misplaced_items}")

        for item, score in scores.items():
            self.graph['nodes'][item]["placement_quality"] = score / 100

        return misplaced_items

    def generate_placements(self, rearrange_object, k = 5,
            target_subset = None, **kwargs):
        # score all receptacles in the scene graph
        receptacle_scoring_prompt = load_prompt("receptacle_scoring", self.prompt_path)

        item_description = get_detailed_item_description(self.graph, rearrange_object)
        item_image = self.graph['nodes'][rearrange_object].get("image", None)

        receptacle_descriptions = {item: (get_detailed_item_description(self.graph, item), self.graph['nodes'][item].get("image", None)) for item in get_nodes_by_attribute(self.graph, "receptacle", lambda x: x is True) if item != rearrange_object}

        print("scoring receptacles...")
        scores = {}

        new_info = False

        for receptacle, (receptacle_description, receptacle_image) in tqdm.tqdm(receptacle_descriptions.items(), desc="Scoring possible receptacles"):
            # TODO: reset/update when new information is gained
            if (rearrange_object, receptacle) in self.score_history: # check if we already analyzed this specific pair
                score, analysis = self.score_history[(rearrange_object, receptacle)]
            else:
                score, analysis = self.get_placement_score(receptacle_scoring_prompt,
                                                           receptacle_description,
                                                           item_description,
                                                           receptacle_image,
                                                           item_image)
                self.score_history[(rearrange_object, receptacle)] = (score, analysis)
                new_info = True
            scores[receptacle] = (score, analysis)

        if not new_info:
            return

        # get top k receptacles
        top_k_receptacles = sorted(scores.keys(), key=lambda x: scores[x][0], reverse=True)[:k]

        print(f"selecting placement for {rearrange_object}. Options: {top_k_receptacles}")

        placement_generator_prompt = load_prompt("placement_generator", self.prompt_path)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": placement_generator_prompt},
            {"role": "user", "content": add_image_to_prompt("carriable object:\n" + item_description, item_image)},
        ] + \
        [
            {"role": "user", "content": add_image_to_prompt("receptacle:\n" + receptacle_descriptions[receptacle][0], receptacle_descriptions[receptacle][1])}
             for receptacle in top_k_receptacles
        ]

        try:
            schema = self._create_schema_for_model(PlacementSuggestion)
            response_data = self._call_openai_with_schema(messages, schema)

            placements_dict = {response_data["best_place"]: {"placement_quality": response_data["certainty"], "analysis": response_data["analysis"], "generic": False}}
            if rearrange_object is not None:
                self.graph['nodes'][rearrange_object]["placement_suggestions"] = placements_dict
        except (pydantic_core._pydantic_core.ValidationError, openai.APIError):
            return
