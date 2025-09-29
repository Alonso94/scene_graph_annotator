#!/usr/bin/env python
import rospy
from hydra_msgs.msg import DsgUpdate
from clio_annotator.msg import AnnotatedDsgUpdate, Annotation
import json
import numpy as np
import base64
try:
    import open3d as o3d
except ImportError:
    rospy.logwarn("open3d not available, point cloud processing disabled")
    o3d = None

# Import from same package - handle both direct execution and package import
try:
    from .clio_aeg_llm import ClioAEGLLMAgent
except ImportError:
    from clio_aeg_llm import ClioAEGLLMAgent

class ClioAEGAnnotator:
    def __init__(self):
        rospy.init_node('clio_aeg_annotator')
        self.graph = {'nodes': {}, 'edges': {}, 'augmented_edges': {}}  # Dict-based graph
        
        # Get ROS parameters for LLM configuration
        llm_model = rospy.get_param('~llm_model', 'gpt-4o-mini')
        temperature = rospy.get_param('~temperature', 0.8)
        
        self.agent = ClioAEGLLMAgent(model=llm_model, temperature=temperature)
        self.pub = rospy.Publisher('~annotated_dsg_update', AnnotatedDsgUpdate, queue_size=10)
        self.sub = rospy.Subscriber('~dsg_update', DsgUpdate, self.dsg_callback)
        self.update_count = 0
        self.annotate_every = rospy.get_param('~annotate_every', 10)
        rospy.loginfo("Clio AEG Annotator started.")

    def dsg_callback(self, msg: DsgUpdate):
        # Parse layer_contents (assume JSON bytes of {'layers': [{'id': layer_id, 'nodes': [node_dict...]}...]})
        try:
            layer_json = json.loads(''.join(map(chr, msg.layer_contents)))
        except Exception as e:
            rospy.logerr(f"Failed to parse layer_contents: {e}")
            return

        # Apply updates
        for layer in layer_json.get('layers', []):
            for node_dict in layer.get('nodes', []):
                node_id = str(node_dict['id'])  # Use str for IDs
                self.graph['nodes'][node_id] = node_dict
                self.agent.graph['nodes'][node_id] = node_dict  # Update agent's graph too
                # Infer pickupable/receptacle if not present
                if 'type' in node_dict:
                    self.graph['nodes'][node_id]['pickupable'] = node_dict['type'] == 'object' and 'bounds' in node_dict and np.linalg.norm(node_dict['bounds'][1] - node_dict['bounds'][0]) < 0.5
                    self.graph['nodes'][node_id]['receptacle'] = node_dict['type'] in ['structure'] or node_dict.get('children', [])
                    self.agent.graph['nodes'][node_id]['pickupable'] = self.graph['nodes'][node_id]['pickupable']
                    self.agent.graph['nodes'][node_id]['receptacle'] = self.graph['nodes'][node_id]['receptacle']
                # Add extra dict if not present
                self.graph['nodes'][node_id]['extra'] = self.graph['nodes'][node_id].get('extra', {})
                self.agent.graph['nodes'][node_id]['extra'] = self.graph['nodes'][node_id]['extra']
                # Transform to np if serialized
                if 'transform' in node_dict and isinstance(node_dict['transform'], list):
                    self.graph['nodes'][node_id]['transform'] = np.array(node_dict['transform']).reshape(4,4)
                    self.agent.graph['nodes'][node_id]['transform'] = self.graph['nodes'][node_id]['transform']
                # Point cloud if base64
                if 'point_cloud' in node_dict and node_dict['point_cloud'] and o3d is not None:
                    pc_bytes = base64.b64decode(node_dict['point_cloud'])
                    pc = o3d.io.read_point_cloud_from_buffer(pc_bytes)  # Assume pcd format; adjust
                    self.graph['nodes'][node_id]['point_cloud'] = pc
                    self.agent.graph['nodes'][node_id]['point_cloud'] = pc

        # Delete nodes
        for del_id in msg.deleted_nodes:
            self.graph['nodes'].pop(str(del_id), None)
            self.agent.graph['nodes'].pop(str(del_id), None)

        # Delete edges (assume edges from children; simplify)
        for del_edge in msg.deleted_edges:
            # If edges stored, remove; skip for now
            pass

        self.update_count += 1
        if msg.full_update or self.update_count % self.annotate_every == 0:
            self.agent.annotate_scene_graph()
            rospy.loginfo("Annotated scene graph.")

        # Publish annotated
        annotated_msg = AnnotatedDsgUpdate()
        annotated_msg.header = msg.header
        annotated_msg.layer_contents = msg.layer_contents  # Copy original
        annotated_msg.deleted_nodes = msg.deleted_nodes
        annotated_msg.deleted_edges = msg.deleted_edges
        annotated_msg.full_update = msg.full_update
        annotated_msg.sequence_number = msg.sequence_number
        # Add annotations
        for node_id, node in self.graph['nodes'].items():
            if 'extra' in node and node['extra']:
                ann = Annotation()
                ann.target_id = int(node_id) if node_id.isdigit() else 0  # Assume numeric
                ann.target_type = 'node'
                extra = node['extra']
                ann.fine_grained_category = extra.get('fine_grained_category', '')
                ann.geometry_and_position = extra.get('geometry_and_position', '')
                ann.relationships = extra.get('relationships', '')
                ann.unique_usage = extra.get('unique_usage', '')
                ann.geometry_and_functionality = extra.get('geometry_and_functionality', '')
                ann.analysis = extra.get('analysis', '')
                ann.placement_quality = extra.get('placement_quality', 0.0)
                ann.semantic_relation = extra.get('semantic_relation', '')
                annotated_msg.annotations.append(ann)
        self.pub.publish(annotated_msg)

if __name__ == '__main__':
    try:
        ClioAEGAnnotator()
    except rospy.ROSInterruptException:
        pass