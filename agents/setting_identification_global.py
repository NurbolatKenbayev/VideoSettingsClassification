import copy
import logging
from typing import Dict, List

import networkx as nx


class GlobalSettingAssignmentAgent:
    """
    Assigns global setting IDs by merging per-character settings based on co-occurrence in frames.
    """

    def __init__(self, scenes: List[Dict]):
        self.logger = logging.getLogger("GlobalSettingAssignmentAgent")
        self.scenes = copy.deepcopy(scenes)

    def assign_global_settings(self) -> List[Dict]:
        self.logger.info("Starting global setting assignment.")

        # Step 1: Build the graph
        graph = nx.Graph()

        # Collect per-character setting IDs from all frames
        for scene in self.scenes:
            frames_metadata = scene.get("frames_metadata", [])
            for frame_metadata in frames_metadata:
                frame_setting_ids = set()
                faces = frame_metadata.get("faces", [])
                for face in faces:
                    person_id = face["person_id"]
                    setting_id = face.get("setting_ID", None)
                    assert (
                        setting_id is not None
                    ), f"`setting_id` is not found for person ID: {person_id}"
                    frame_setting_ids.add(setting_id)
                    graph.add_node(setting_id)

                # Add edges between all pairs of per-character setting IDs in the frame
                for setting_id1 in frame_setting_ids:
                    for setting_id2 in frame_setting_ids:
                        if setting_id1 != setting_id2:
                            graph.add_edge(setting_id1, setting_id2)

        self.logger.info(
            f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
        )

        # Step 2: Find connected components
        connected_components = list(nx.connected_components(graph))
        self.logger.info(f"Found {len(connected_components)} connected components.")

        # Step 3: Assign global setting IDs
        setting_id_to_global = {}
        for idx, component in enumerate(connected_components):
            global_setting_id = f"global_setting_{idx}"
            for setting_id in component:
                setting_id_to_global[setting_id] = global_setting_id

        # Step 4: Update frames with global setting IDs
        for scene in self.scenes:
            frames_metadata = scene.get("frames_metadata", [])
            for frame_metadata in frames_metadata:
                frame_global_setting_ids = set()
                faces = frame_metadata.get("faces", [])
                for face in faces:
                    # Get per-character setting IDs
                    setting_id = face.get("setting_ID", None)
                    assert (
                        setting_id is not None
                    ), f"`setting_id` is not found for person ID: {person_id}"

                    # Map to global setting IDs
                    face_global_setting_id = setting_id_to_global[setting_id]
                    face["global_setting_ID"] = face_global_setting_id
                    frame_global_setting_ids.add(face_global_setting_id)
                assert (
                    len(frame_global_setting_ids) <= 1
                ), f"There are more than one candidate for the `global_setting_id` for single frame: {frame_metadata}"
                # For the frame, assign the global setting ID
                if len(faces) == 0:
                    frame_metadata["global_setting_ID"] = "Unknown"
                else:
                    frame_metadata["global_setting_ID"] = face_global_setting_id

        self.logger.info("Global setting assignment completed.")

        return self.scenes
