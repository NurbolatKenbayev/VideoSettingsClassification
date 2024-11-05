import logging
import multiprocessing
import os
from typing import Dict, List

import openai

from agents.utils import setup_logger


def classify_frame_setting(
    frame_path: str,
    caption: str,
    possible_settings: List[str],
    openai_api_key: str,
    logger: logging.Logger,
) -> str:
    openai.api_key = openai_api_key  # Set API key in the process

    prompt = (
        f"Based on the following description, classify the setting of the scene into one of the predefined categories.\n\n"
        f"Description: {caption}\n\n"
        f"Possible settings: {', '.join(possible_settings)}\n\n"
        f"Answer format: Only provide the setting name from the list above."
    )
    # Notice specific details and objects in the description classify the setting of the scene into one of the predefined categories
    # thinking about which of these categories most likely will have such objects.
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that classifies video frames into predefined settings.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.0,
        )

        setting = response.choices[0].message.content.strip().lower()

        # Validate the setting
        if setting not in [s.lower() for s in possible_settings]:
            logger.warning(
                f"Invalid setting '{setting}' received from OpenAI. Setting to 'unknown'."
            )
            setting = "unknown"
        else:
            logger.info(
                f"Predicted setting '{setting}' for frame '{frame_path}' with caption: \n'{caption}'"
            )

        return setting

    except openai.RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        # Handle rate limit error (e.g., retry after some time)
        return "unknown"

    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        return "unknown"


def process_frame_setting(
    caption_data: dict,
    possible_settings: List[str],
    openai_api_key: str,
    logger: logging.Logger,
) -> dict:
    caption = caption_data.get("caption", "")
    frame_path = caption_data["frame_path"]
    if caption:
        setting = classify_frame_setting(
            frame_path, caption, possible_settings, openai_api_key, logger
        )
        caption_data["setting"] = setting
    else:
        caption_data["setting"] = "unknown"
    return caption_data


def classify_settings_one_cut_scene(
    scene: dict, possible_settings: List[str], openai_api_key: str
) -> dict:
    log_folder = "./logs/SettingClassifierAgent_logs"
    os.makedirs(log_folder, exist_ok=True)
    logger = setup_logger(log_folder)

    scene_number = scene["cut_scene_number"]
    captions = scene.get("captions", [])

    if not captions:
        logger.warning(
            f"No captions found for Scene {scene_number}. Skipping classification."
        )
        scene["captions"] = []
        return scene

    logger.info(
        f"Classifying settings for Scene {scene_number} with {len(captions)} frames."
    )

    setting_list = []
    for caption in captions:
        frame_setting = process_frame_setting(
            caption, possible_settings, openai_api_key, logger
        )
        setting_list.append(frame_setting)

    # Update scene metadata with captions
    scene["captions"] = setting_list

    logger.info(f"Completed setting classification for Scene {scene_number}.")

    return scene


class SettingClassifierAgent:
    """
    Assigns settings for frames extracted from scenes based on their caption.
    """

    def __init__(
        self,
        num_processes: int,
        scenes: List[Dict],
        possible_settings: List[str],
        openai_api_key: str,
    ):
        self.logger = logging.getLogger("SettingClassifierAgent")
        self.num_processes = num_processes
        self.scenes = scenes
        self.possible_settings = possible_settings
        self.openai_api_key = openai_api_key

    def classify_settings(self) -> List[Dict]:
        """
        Generates settings for each frame in the scenes.

        Returns:
            List[Dict]: Updated scene metadata including settings.
        """
        self.logger.info("Starting setting classification for scenes.")

        # Prepare chunks for multiprocessing
        chunks = [
            (scene, self.possible_settings, self.openai_api_key)
            for scene in self.scenes
        ]

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = pool.starmap(classify_settings_one_cut_scene, chunks)

        self.logger.info("Setting classification for all scenes completed.")

        return results
