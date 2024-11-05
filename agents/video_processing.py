import logging
import os
from typing import Dict, List

from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import ContentDetector, ThresholdDetector


class VideoProcessor:
    """
    Handles video loading and scene detection using PySceneDetect.
    """

    def __init__(
        self, video_path: str, detector_type: str = "content", threshold: float = None
    ):
        # Obtain a logger for this class/module
        self.logger = logging.getLogger("VideoProcessor")

        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.scene_list = []
        self.detector_type = detector_type.lower()
        self.threshold = threshold

    def detect_scenes(self) -> List[Dict]:
        """
        Detects scenes in the video using the specified detection method.

        Returns:
            List[Dict]: A list of dictionaries containing scene metadata.
        """
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()

        # Select the detector based on the specified type
        if self.detector_type == "content":
            # Effective at detecting cuts where the visual content changes abruptly.
            # Suitable for a wide variety of videos with diverse content.
            # Less sensitive to small changes in lighting or noise, as it focuses on significant content differences.
            threshold_value = (
                self.threshold if self.threshold is not None else 27.0
            )  # Default value
            detector = ContentDetector(threshold=threshold_value)
            self.logger.info(f"Using ContentDetector with threshold={threshold_value}")
        elif self.detector_type == "threshold":
            # Detects scene changes based on changes in the average luminance (brightness) of frames.
            # Monitors the brightness level of frames to identify transitions.
            threshold_value = (
                self.threshold if self.threshold is not None else 12.0
            )  # Default value
            detector = ThresholdDetector(threshold=threshold_value)
            self.logger.info(
                f"Using ThresholdDetector with threshold={threshold_value}"
            )
        else:
            self.logger.error(f"Unsupported detector type: {self.detector_type}")
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

        scene_manager.add_detector(detector)
        video_manager.set_downscale_factor()

        try:
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()

            # If no scenes are detected, log a warning
            if not scene_list:
                self.logger.warning("No scenes were detected in the video.")

            self.scene_list = [
                {
                    "start_timecode": start.get_timecode(),
                    "end_timecode": end.get_timecode(),
                    "start_seconds": start.get_seconds(),
                    "end_seconds": end.get_seconds(),
                    "start_frame": start.get_frames(),
                    "end_frame": end.get_frames(),
                    "cut_scene_number": idx + 1,
                }
                for idx, (start, end) in enumerate(scene_list)
            ]

            self.logger.info(
                f"Detected {len(self.scene_list)} scenes using {self.detector_type} detector."
            )

            return self.scene_list

        except Exception as e:
            self.logger.error(f"Error during scene detection: {e}")
            raise

        finally:
            video_manager.release()
            self.logger.info("VideoManager resources have been released.")
