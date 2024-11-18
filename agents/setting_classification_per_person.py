"""
Complexity of the request calls to openai API is n(n - 1)/2 = O(n^2),
since each new frame in a row iterates over previous frames and is compared with them.
But note that usually subsequent frames per person have similar setting
and early stopping criteria will work most of the time.
"""

import copy
import logging
from typing import Dict, List

import openai

from agents.utils import encode_image


def compare_setting_two_frames(
    frame1: str, frame2: str, openai_api_key: str, logger: logging.Logger
) -> bool:
    # # For debugging and testing just uncomment this part
    # return True

    openai.api_key = openai_api_key  # Set API key in the process

    # Getting the base64 string
    base64_image1 = encode_image(frame1)
    base64_image2 = encode_image(frame2)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that compares two images to determine if they show the same setting.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Based on the two images given below, determine if they have the same setting. Setting can be defined as a location, where some actions are taking place. If in one video actions are taking place in several places, then there would be several settings. Pay attention to people's pose, hairstyle, clothes, and background lighting. Answer just 'Yes' or 'No'.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image1}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image2}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=300,
        )

        flag = response.choices[0].message.content
        logger.info(f"Answer for {frame1} and {frame2}: {flag}")
        ans_list = ["true", "true.", "yes", "yes."]

        return flag.lower() in ans_list

    except openai.RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        # Handle rate limit error (e.g., retry after some time)
        return False

    except Exception as e:
        logger.error(
            f"Error during OpenAI API call for frames {frame1} and {frame2}: {e}"
        )
        return False


def classify_setting_one_person(
    person_id: int, frames_data: List[Dict], openai_api_key: str, logger: logging.Logger
) -> List[Dict]:
    next_setting_idx = 0
    for i, data in enumerate(frames_data):
        if i == 0:
            data["setting_ID"] = (
                f"person_{person_id}_setting_{next_setting_idx}"  # changes `frames_data`
            )
            next_setting_idx += 1
        else:
            j = i - 1
            frame1 = data["frame_path"]
            flag = False
            while j >= 0 and not flag:
                frame2 = frames_data[j]["frame_path"]

                flag = compare_setting_two_frames(
                    frame1, frame2, openai_api_key, logger
                )
                # flag = True  # For debugging

                j -= 1

            if flag:
                data["setting_ID"] = frames_data[j + 1][
                    "setting_ID"
                ]  # changes `frames_data`
            else:
                data["setting_ID"] = (
                    f"person_{person_id}_setting_{next_setting_idx}"  # changes `frames_data`
                )
                next_setting_idx += 1

    return frames_data


class SettingClassificationCharacterAgent:
    """
    Clusters frames per person based on visual similarity to assign setting IDs.
    """

    def __init__(self, scenes: List[Dict], openai_api_key: str):
        self.logger = logging.getLogger("SettingClassificationCharacterAgent")
        self.scenes = copy.deepcopy(scenes)
        self.openai_api_key = openai_api_key

    def cluster_frames(self) -> List[Dict]:
        """
        Clusters frames per person and assigns setting IDs.
        """
        self.logger.info("Starting setting classification per person.")

        # Mapping from person_id to list of frames
        person_frames = {}

        # Collect frames per person
        for scene in self.scenes:
            frames_metadata = scene.get("frames_metadata", [])
            for frame_metadata in frames_metadata:
                frame_path = frame_metadata["frame_path"]
                faces = frame_metadata.get("faces", [])
                for face in faces:
                    person_id = face["person_id"]
                    if person_id not in person_frames:
                        person_frames[person_id] = []
                    person_frames[person_id].append(
                        {
                            "cut_scene_number": scene["cut_scene_number"],
                            "frame_path": frame_path,
                            "caption": frame_metadata.get("caption", None),
                            "faces": frame_metadata["faces"],
                        }
                    )

        self.logger.info(f"There are {len(person_frames)} different persons.")

        # Process each person separately
        for person_id, frames_data in person_frames.items():
            self.logger.info(
                f"Processing person ID {person_id} with {len(frames_data)} frames."
            )

            if len(frames_data) == 0:
                self.logger.info(
                    f"There are no frames found with person ID {person_id}. Skipping setting classification for this person."
                )
                continue

            frames_data = classify_setting_one_person(
                person_id, frames_data, self.openai_api_key, self.logger
            )  # changes `frames_data` and accordingly `person_frames`

        self.logger.info("Setting classification per person completed.")

        return self.scenes, person_frames
