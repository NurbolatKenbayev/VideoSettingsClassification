import base64
import logging
import multiprocessing
import os
from typing import Dict, List

import openai

from agents.utils import setup_logger


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_caption_one_image(
    image_path: str, openai_api_key: str, logger: logging.Logger
) -> str:
    # # For debugging and testing just uncomment this part
    # logger.info(f"Generated caption for frame {image_path}")
    # return {"frame_path": image_path, "caption": None}

    openai.api_key = openai_api_key  # Set API key in the process

    # Getting the base64 string
    base64_image = encode_image(image_path)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant for image captioning.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in the picture, noting down specific details: physical environment, surroundings, places.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=300,
        )

        logger.info(f"Generated caption for frame {image_path}")

        return {
            "frame_path": image_path,
            "caption": response.choices[0].message.content,
        }

    except openai.RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        # Handle rate limit error (e.g., retry after some time)
        return {"frame_path": image_path, "caption": None}

    except Exception as e:
        logger.error(f"Error during OpenAI API call for frame {image_path}: {e}")
        return {"frame_path": image_path, "caption": None}


def generate_caption_one_cut_scene(scene: dict, openai_api_key: str) -> dict:
    log_folder = "./logs/ImageCaptioningAgent_logs"
    os.makedirs(log_folder, exist_ok=True)
    logger = setup_logger(log_folder)

    scene_number = scene["cut_scene_number"]
    frame_paths = scene.get("frame_paths", [])

    if not frame_paths:
        logger.warning(
            f"No frames found for Scene {scene_number}. Skipping captioning."
        )
        scene["captions"] = []
        return scene

    logger.info(
        f"Generating captions for Scene {scene_number} with {len(frame_paths)} frames."
    )

    caption_list = []
    for frame_path in frame_paths:
        frame_caption = generate_caption_one_image(frame_path, openai_api_key, logger)
        caption_list.append(frame_caption)

    # Update scene metadata with captions
    scene["captions"] = caption_list

    logger.info(f"Completed caption generation for Scene {scene_number}.")

    return scene


class ImageCaptioningAgent:
    """
    Generates captions for frames extracted from scenes.
    """

    def __init__(self, num_processes: int, scenes: List[Dict], openai_api_key: str):
        self.logger = logging.getLogger("ImageCaptioningAgent")
        self.num_processes = num_processes
        self.scenes = scenes
        self.openai_api_key = openai_api_key

    def generate_captions(self) -> List[Dict]:
        """
        Generates captions for each frame in the scenes.

        Returns:
            List[Dict]: Updated scene metadata including captions.
        """
        self.logger.info("Starting image captioning for scenes.")

        # Prepare chunks for multiprocessing
        chunks = [(scene, self.openai_api_key) for scene in self.scenes]

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = pool.starmap(generate_caption_one_cut_scene, chunks)

        self.logger.info("Image captioning for all scenes completed.")

        return results
