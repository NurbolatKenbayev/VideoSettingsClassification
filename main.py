import argparse
import json
import logging
import os

from dotenv import load_dotenv

from agents.frame_extraction import FrameExtractor
from agents.image_captioning import ImageCaptioningAgent
from agents.setting_classification import SettingClassifierAgent
from agents.video_processing import VideoProcessor

if __name__ == "__main__":
    # rm -rf ./frames ./logs
    # python main.py --video_path "./input_data/minecraft.mp4" --num_processes 28 --possible_settings_path "./input_data/possible_settings_minecraft_processed.json"
    parser = argparse.ArgumentParser(
        description="Video settings classification with agents' orchestration."
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to the input video data.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to be used in multiprocessing.",
    )
    parser.add_argument(
        "--possible_settings_path",
        type=str,
        default=None,
        help="Path to the dictionary with possible settings.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Configure logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, "main_log.txt"),
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - \n%(message)s \n",
        level=logging.INFO,
    )

    main_logger = logging.getLogger(__name__)
    main_logger.info("Main pipeline process started.")

    # Step 1: Scene Detection
    video_processor = VideoProcessor(
        args.video_path, detector_type="content", threshold=27.0
    )
    scenes = video_processor.detect_scenes()

    # Step 2: Frame Extraction
    frame_extractor = FrameExtractor(
        video_path=args.video_path,
        scenes=scenes,
        output_dir="frames",
        frames_per_scene=1,  # You can adjust this to extract more frames per scene
    )
    scenes_with_frames = frame_extractor.extract_frames()

    # Step 3: Image Captioning
    load_dotenv()  # take environment variables from .env.
    openai_api_key = os.getenv("OPENAI_API_KEY")

    image_captioning_agent = ImageCaptioningAgent(
        num_processes=args.num_processes,
        scenes=scenes_with_frames,
        openai_api_key=openai_api_key,
    )
    scenes_with_captions = image_captioning_agent.generate_captions()

    # Step 4: Setting Classification
    with open(args.possible_settings_path, "r", encoding="utf-8") as json_file:
        possible_settings_dict = json.load(json_file)
    possible_settings = list(possible_settings_dict.keys())
    possible_settings = [item.lower() for item in possible_settings]

    setting_classifier_agent = SettingClassifierAgent(
        num_processes=args.num_processes,
        scenes=scenes_with_captions,
        possible_settings=possible_settings,
        openai_api_key=openai_api_key,
    )
    scenes_with_settings = setting_classifier_agent.classify_settings()
    # Now scenes_with_settings contains settings per frame
    # You can proceed to aggregate settings or perform further analysis

    # Saving results
    dir_to_save = "./results"
    os.makedirs(dir_to_save, exist_ok=True)
    json_file_path = os.path.join(dir_to_save, "scenes_with_settings_predicted.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(scenes_with_settings, json_file, ensure_ascii=False, indent=4)
