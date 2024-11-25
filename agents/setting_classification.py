import copy
import logging
from typing import Dict, List, Tuple

import openai


def classify_settings_one_frame(
    frame_metadata: dict,
    openai_api_key: str,
    logger: logging.Logger,
) -> str:
    caption = frame_metadata.get("caption", "")
    if caption:
        # ### For debugging and testing just uncomment this part
        # setting = "debugging"
        # logger.info(
        #     "Predicted setting '{:}' for frame '{:}' with caption: \n'{:}'".format(
        #         setting, frame_metadata["frame_path"], caption
        #     )
        # )
        # frame_metadata["global_setting_name"] = setting
        # return setting
        # ###

        openai.api_key = openai_api_key  # Set API key in the process

        prompt = (
            f"Based on the following description, classify the setting of the scene. Setting can be defined as a location, where some actions are taking place. If in one video actions are taking place in several places, then there would be several settings.\n\n"
            f"Description: {caption}\n\n"
            f"Answer format: Only provide the setting name."
        )
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that classifies video frames into settings.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.0,
            )

            setting = response.choices[0].message.content.strip().lower()

            logger.info(
                "Predicted setting '{:}' for frame '{:}' with caption: \n'{:}'".format(
                    setting, frame_metadata["frame_path"], caption
                )
            )

        except openai.RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            # Handle rate limit error (e.g., retry after some time)
            setting = "unknown"

        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            setting = "unknown"

    else:
        setting = "unknown"

    frame_metadata["global_setting_name"] = setting
    return setting


def classify_settings_one_cut_scene(
    scene: dict,
    openai_api_key: str,
    global_setting_ID_to_name_dict: Dict,
    logger: logging.Logger,
) -> dict:
    scene_number = scene["cut_scene_number"]
    frames_metadata = scene.get("frames_metadata", [])

    if not frames_metadata:
        logger.warning(
            f"No 'frames_metadata' found for Scene {scene_number}. Skipping classification."
        )
        return scene

    logger.info(
        f"Classifying settings for Scene {scene_number} with {len(frames_metadata)} frames."
    )

    for frame_metadata in frames_metadata:
        if frame_metadata["global_setting_ID"] in global_setting_ID_to_name_dict:
            frame_metadata["global_setting_name"] = global_setting_ID_to_name_dict[
                frame_metadata["global_setting_ID"]
            ]  # changes `frame_metadata`, so `frames_metadata`, so `scene`
        else:
            setting = classify_settings_one_frame(
                frame_metadata, openai_api_key, logger
            )  # changes `frame_metadata`, so `frames_metadata`, so `scene`
            if frame_metadata["global_setting_ID"] != "Unknown":
                global_setting_ID_to_name_dict[frame_metadata["global_setting_ID"]] = (
                    setting
                )

    logger.info(f"Completed setting classification for Scene {scene_number}.")

    return global_setting_ID_to_name_dict


class SettingClassifierAgent:
    """
    Assigns settings for frames extracted from scenes based on their caption.
    """

    def __init__(
        self,
        num_processes: int,
        scenes: List[Dict],
        openai_api_key: str,
    ):
        self.logger = logging.getLogger("SettingClassifierAgent")
        self.num_processes = num_processes
        self.scenes = copy.deepcopy(scenes)
        self.openai_api_key = openai_api_key

    def classify_settings(self) -> Tuple[List[Dict], Dict]:
        """
        Generates settings for each frame in the scenes.

        Returns:
            List[Dict]: Updated scene metadata including settings.
        """
        self.logger.info("Starting setting classification for scenes.")

        global_setting_ID_to_name_dict = {}
        for scene in self.scenes:
            global_setting_ID_to_name_dict = classify_settings_one_cut_scene(
                scene, self.openai_api_key, global_setting_ID_to_name_dict, self.logger
            )  # changes `scene`, so `self.scenes`

        self.logger.info("Setting classification for all scenes completed.")

        return self.scenes, global_setting_ID_to_name_dict
