import copy
import logging
import os
import pickle
from typing import Dict, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class CharactersClassificationAgent:
    """
    Detects faces in frames and assigns unique IDs to each person using InsightFace.
    """

    def __init__(
        self,
        scenes: List[Dict],
        threshold: float = 1.0,
        known_faces_db: str = "./characters/known_faces_insightface.pkl",
        model_name: str = "buffalo_l",
        providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ctx_id: int = 0,
    ):
        self.logger = logging.getLogger("CharactersClassificationAgent")
        self.scenes = copy.deepcopy(scenes)
        self.threshold = threshold  # Threshold for the characters' similarity based on the L2-norm. Adjust based on desired sensitivity
        self.known_faces_db = known_faces_db
        self.known_face_embeddings = []
        self.known_face_ids = []
        self.next_person_id = 0  # Counter for assigning unique IDs

        # Initialize the InsightFace app
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id)

        # Load known faces database if it exists
        if os.path.exists(self.known_faces_db):
            self._load_known_faces()
        else:
            self.logger.info("No existing known faces database found. Starting fresh.")

    def _load_known_faces(self):
        with open(self.known_faces_db, "rb") as f:
            data = pickle.load(f)
            self.known_face_embeddings = data["embeddings"]
            self.known_face_ids = data["ids"]
            self.next_person_id = data["next_id"]
            self.logger.info(
                f"Loaded known faces database with {len(self.known_face_embeddings)} faces."
            )

    def _save_known_faces(self):
        data = {
            "embeddings": self.known_face_embeddings,
            "ids": self.known_face_ids,
            "next_id": self.next_person_id,
        }
        os.makedirs("/".join(self.known_faces_db.split("/")[:-1]), exist_ok=True)
        with open(self.known_faces_db, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(
            f"Saved known faces database with {len(self.known_face_embeddings)} faces."
        )

    def classify_characters(self) -> List[Dict]:
        """
        Detects faces in frames and assigns unique IDs.
        """
        self.logger.info("Starting character classification using InsightFace.")

        for scene in self.scenes:
            scene_number = scene["cut_scene_number"]
            frames_metadata = scene.get("frames_metadata", [])
            if not frames_metadata:
                self.logger.warning(
                    f"No 'frames_metadata' found for Scene {scene_number}. Skipping character classification."
                )
                continue

            for frame_metadata in frames_metadata:
                frame_path = frame_metadata["frame_path"]
                frame = cv2.imread(frame_path)

                if frame is None:
                    self.logger.error(f"Failed to read frame image: {frame_path}")
                    continue

                # Perform face analysis
                faces = self.app.get(frame)

                face_data = []

                for face in faces:
                    # Get face embedding
                    face_embedding = face.embedding  # numpy array of shape (512,)

                    # Compare face embedding with known faces
                    person_id = self._identify_person(face_embedding)

                    # Store face data
                    bbox = face.bbox.astype(int)  # Bounding box coordinates
                    face_data.append(
                        {
                            "person_id": person_id,
                            "face_location": {
                                "top": bbox[1],
                                "right": bbox[2],
                                "bottom": bbox[3],
                                "left": bbox[0],
                            },
                        }
                    )

                # Update caption data with face data
                frame_metadata["faces"] = (
                    face_data  # changes `frames_metadata` and these changes will be reflected in the `self.scenes`
                )

            self.logger.info(
                f"Completed character classification for Scene {scene_number}"
            )

        # Save the known faces database
        self._save_known_faces()

        self.logger.info("Character classification completed.")
        return self.scenes

    def _identify_person(self, face_embedding):
        """
        Identifies a person by comparing the face embedding with known faces.
        Assigns a new ID if no match is found.
        """
        # Normalize the face embedding
        face_embedding = self._normalize_embedding(face_embedding)

        if not self.known_face_embeddings:
            # No known faces yet
            person_id = self.next_person_id
            self.next_person_id += 1
            self.known_face_embeddings.append(face_embedding)
            self.known_face_ids.append(person_id)
            self.logger.info(f"New person detected with ID {person_id}")
            return person_id
        else:
            # Compare with known faces
            embeddings_array = np.vstack(
                self.known_face_embeddings
            )  # Known embeddings are already normalized

            # Compute cosine similarities
            similarities = embeddings_array @ face_embedding  # Dot product
            max_similarity = np.max(similarities)

            if max_similarity >= self.threshold:
                # Match found. Remark: if there are multiple candidates the first one will be selected.
                index = np.argmax(similarities)
                person_id = self.known_face_ids[index]
                self.logger.info(
                    f"Match found for person ID {person_id} with similarity {max_similarity}"
                )
                return person_id
            else:
                # No match, assign new ID
                person_id = self.next_person_id
                self.next_person_id += 1
                self.known_face_embeddings.append(face_embedding)
                self.known_face_ids.append(person_id)
                self.logger.info(f"New person detected with ID {person_id}")
                return person_id

    def _normalize_embedding(self, embedding):
        """
        Normalizes a face embedding to unit length.

        Args:
            embedding (np.ndarray): The face embedding vector.

        Returns:
            np.ndarray: The normalized embedding vector.
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            self.logger.warning("Encountered face embedding with zero norm.")
            return embedding  # Avoid division by zero
        return embedding / norm
