import os

import cv2


def detect_faces_frame(frame, face_data, text_type, font_scale=1.0, thickness=2):
    # Draw bounding boxes and IDs on the frame
    for face in face_data:
        top = face["face_location"]["top"]
        right = face["face_location"]["right"]
        bottom = face["face_location"]["bottom"]
        left = face["face_location"]["left"]

        if text_type == "person_id":
            person_id = face["person_id"]
            text = f"person_{person_id}"
            position = (left, top - 10)
        elif text_type == "person_id_setting_ID":
            setting_ID = face["setting_ID"]
            text = f"{setting_ID}"
            position = (left - 10, top - 10)
        else:
            raise ValueError(
                f"text_type = '{text_type}' is not implemented for detect_faces_frame()."
            )

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness)
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            thickness,
        )

    return frame


def put_text_one_frame(frame, text, font_scale=1.0, thickness=2):
    # Get the dimensions of the frame
    image_height, image_width = frame.shape[:2]

    # Define font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the size of the text
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Calculate the position for the text
    x = (image_width - text_width) // 2  # Center horizontally
    y = image_height - baseline - 10  # Position 10 pixels above the bottom

    # Optional: Draw a background rectangle for better visibility
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + baseline + 5),
        (0, 0, 0),  # Black rectangle
        -1,  # Filled rectangle
    )

    # Draw the text on the frame
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    return frame


def put_text_frames(
    scenes, visualization_type, text_type, logger, font_scale=1.0, thickness=2
):
    if visualization_type == "faces":
        os.makedirs(f"./frames_{text_type}", exist_ok=True)
    elif visualization_type == "text_global_setting_ID":
        os.makedirs("./frames_global_setting_ID", exist_ok=True)
    elif visualization_type == "text_global_setting_name":
        os.makedirs("./frames_global_setting_name", exist_ok=True)
    else:
        raise ValueError(
            f"visualization_type = '{visualization_type}' is not implemented for put_text_frames()."
        )

    for scene in scenes:
        scene_number = scene["cut_scene_number"]
        frames_metadata = scene.get("frames_metadata", [])
        if not frames_metadata:
            logger.warning(
                f"No `frames_metadata` found for Scene {scene_number}. Skipping 'put_text_frames()'."
            )
            continue

        for frame_metadata in frames_metadata:
            frame = cv2.imread(frame_metadata["frame_path"])

            if visualization_type == "faces":
                faces = frame_metadata.get("faces", [])
                if len(faces) == 0:
                    continue

                frame = detect_faces_frame(
                    frame=frame,
                    face_data=faces,
                    text_type=text_type,
                    font_scale=font_scale,
                    thickness=thickness,
                )
                annotated_frame_path = frame_metadata["frame_path"].replace(
                    "frames/", f"frames_{text_type}/"
                )
                cv2.imwrite(annotated_frame_path, frame)  # Save the annotated frame

            elif visualization_type == "text_global_setting_ID":
                global_setting_ID = frame_metadata.get("global_setting_ID", None)
                assert (
                    global_setting_ID is not None
                ), "'global_setting_ID' is missing for frame: {:}".format(
                    frame_metadata["frame_path"]
                )

                frame = put_text_one_frame(
                    frame=frame,
                    text=global_setting_ID,
                    font_scale=font_scale,
                    thickness=thickness,
                )
                annotated_frame_path = frame_metadata["frame_path"].replace(
                    "frames/", "frames_global_setting_ID/"
                )
                cv2.imwrite(annotated_frame_path, frame)  # Save the annotated frame

            elif visualization_type == "text_global_setting_name":
                global_setting_ID = frame_metadata.get("global_setting_ID", None)
                assert (
                    global_setting_ID is not None
                ), "'global_setting_ID' is missing for frame: {:}".format(
                    frame_metadata["frame_path"]
                )
                global_setting_name = frame_metadata.get("global_setting_name", None)
                assert (
                    global_setting_name is not None
                ), "'global_setting_name' is missing for frame: {:}".format(
                    frame_metadata["frame_path"]
                )

                frame = put_text_one_frame(
                    frame=frame,
                    text=global_setting_ID + ": " + global_setting_name,
                    font_scale=font_scale,
                    thickness=thickness,
                )
                annotated_frame_path = frame_metadata["frame_path"].replace(
                    "frames/", "frames_global_setting_name/"
                )
                cv2.imwrite(annotated_frame_path, frame)  # Save the annotated frame

            else:
                raise ValueError(
                    f"visualization_type = '{visualization_type}' is not implemented for put_text_frames()."
                )

        logger.info(f"Completed 'put_text_frames()' for Scene {scene_number}.")
