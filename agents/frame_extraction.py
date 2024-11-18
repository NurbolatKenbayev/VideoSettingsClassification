import copy
import logging
import os
from typing import Dict, List

import cv2


class FrameExtractor:
    """
    Extracts frames corresponding to each scene from the video.
    """

    def __init__(
        self,
        video_path: str,
        scenes: List[Dict],
        output_dir: str = "frames",
        frames_per_scene: int = 1,
    ):
        self.logger = logging.getLogger("FrameExtractor")
        self.video_path = video_path
        self.scenes = copy.deepcopy(scenes)
        self.output_dir = output_dir
        self.frames_per_scene = (
            frames_per_scene  # Number of frames to extract per scene
        )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_frames(self) -> List[Dict]:
        """
        Extracts frames for each scene and stores them alongside the metadata.

        Returns:
            List[Dict]: Updated scene metadata including frame file paths.
        """
        self.logger.info("Starting frame extraction.")
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            self.logger.error(f"Failed to open video file: {self.video_path}")
            raise IOError(f"Failed to open video file: {self.video_path}")

        for scene in self.scenes:
            scene_number = scene["cut_scene_number"]
            start_frame = int(scene["start_frame"])
            end_frame = int(scene["end_frame"])
            total_frames = end_frame - start_frame + 1

            # self.logger.debug(
            #     f"Extracting frames for Scene {scene_number}: Frames {start_frame} to {end_frame}"
            # )

            # Determine frame indices to extract
            if self.frames_per_scene >= total_frames:
                # Extract all frames in the scene
                frame_indices = list(range(start_frame, end_frame + 1))
            else:
                # Evenly distribute frames across the scene
                frame_indices = self._get_frame_indices(
                    start_frame, end_frame, self.frames_per_scene
                )

            frame_paths = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    self.logger.warning(
                        f"Failed to read frame {frame_idx} for Scene {scene_number}"
                    )
                    continue

                frame_filename = f"scene_{scene_number}_frame_{frame_idx}.jpg"
                frame_path = os.path.join(self.output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)

            # Update scene metadata with frame paths
            scene["frame_paths"] = frame_paths  # changes `self.scenes`

            self.logger.info(
                f"Extracted {len(frame_paths)} frames for Scene {scene_number}"
            )

        cap.release()
        self.logger.info("Frame extraction completed.")

        return self.scenes

    def _get_frame_indices(
        self, start_frame: int, end_frame: int, num_frames: int
    ) -> List[int]:
        """
        Evenly distributes frame indices across the scene.

        Args:
            start_frame (int): Starting frame of the scene.
            end_frame (int): Ending frame of the scene.
            num_frames (int): Number of frames to extract.

        Returns:
            List[int]: List of frame indices to extract.
        """
        total_frames = end_frame - start_frame + 1
        interval = total_frames / num_frames
        frame_indices = [int(start_frame + i * interval) for i in range(num_frames)]
        return frame_indices
