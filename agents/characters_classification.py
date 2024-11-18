import logging
import os
import copy
import pickle
from typing import Dict, List
import cv2

import face_recognition



class CharactersClassificationAgent:
    """
    Detects faces in frames and assigns unique IDs to each person.
    """

    def __init__(
        self,
        scenes: List[Dict],
        known_faces_db: str = "./characters/known_faces.pkl",
        tolerance: float = 0.6,  # Threshold for face comparison (lower values are more strict)
    ):
        self.logger = logging.getLogger("CharactersClassificationAgent")
        self.scenes = copy.deepcopy(scenes)
        self.known_faces_db = known_faces_db
        self.tolerance = tolerance  # Threshold for face comparison
        self.known_face_encodings = []
        self.known_face_ids = []
        self.next_person_id = 0  # Counter for assigning unique IDs

        # Load known faces database if it exists
        if os.path.exists(self.known_faces_db):
            self._load_known_faces()
        else:
            self.logger.info("No existing known faces database found. Starting fresh.")

    def _load_known_faces(self):
        with open(self.known_faces_db, "rb") as f:
            data = pickle.load(f)
            self.known_face_encodings = data["encodings"]
            self.known_face_ids = data["ids"]
            self.next_person_id = data["next_id"]
            self.logger.info(
                f"Loaded known faces database with {len(self.known_face_encodings)} faces."
            )

    def _save_known_faces(self):
        data = {
            "encodings": self.known_face_encodings,
            "ids": self.known_face_ids,
            "next_id": self.next_person_id,
        }
        os.makedirs("/".join(self.known_faces_db.split("/")[:-1]), exist_ok=True)
        with open(self.known_faces_db, "wb") as f:
            pickle.dump(data, f)
        self.logger.info(
            f"Saved known faces database with {len(self.known_face_encodings)} faces."
        )

    def classify_characters(self) -> List[Dict]:
        """
        Detects faces in frames and assigns unique IDs.
        """
        self.logger.info("Starting character classification.")

        for scene in self.scenes:
            scene_number = scene["cut_scene_number"]
            frames_metadata = scene.get("frames_metadata", [])
            if not frames_metadata:
                self.logger.warning(
                    f"No `frames_metadata` found for Scene {scene_number}. Skipping character classification."
                )
                continue

            for frame_metadata in frames_metadata:
                frame_path = frame_metadata["frame_path"]
                frame = cv2.imread(frame_path)

                if frame is None:
                    self.logger.error(f"Failed to read frame image: {frame_path}")
                    continue

                # Convert the image from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces and get encodings
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations
                )

                face_data = []

                for face_location, face_encoding in zip(face_locations, face_encodings):
                    # Compare face encodings with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,  # Remark: only first match is enough, early stopping criteria might speed up the process
                        face_encoding,
                        tolerance=self.tolerance,
                    )
                    person_id = None

                    if True in matches:
                        first_match_index = matches.index(True)
                        person_id = self.known_face_ids[first_match_index]
                        self.logger.debug(
                            f"Match found for person ID {person_id} in frame {frame_path}"
                        )
                    else:
                        # Assign a new ID to the new face
                        person_id = self.next_person_id
                        self.next_person_id += 1
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_ids.append(person_id)
                        self.logger.debug(
                            f"New person detected with ID {person_id} in frame {frame_path}"
                        )

                    # Store face data
                    top, right, bottom, left = face_location
                    face_data.append(
                        {
                            "person_id": person_id,
                            "face_location": {
                                "top": top,
                                "right": right,
                                "bottom": bottom,
                                "left": left,
                            },
                        }
                    )

                # Update caption data with face data
                frame_metadata["faces"] = face_data # changes `frames_metadata` and these changes will be reflected in the `self.scenes`

            self.logger.info(
                f"Completed character classification for Scene {scene_number}"
            )

        # Save the known faces database
        self._save_known_faces()

        self.logger.info("Character classification completed.")

        return self.scenes
