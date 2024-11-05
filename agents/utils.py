import logging
import multiprocessing
import os
from datetime import datetime


def setup_logger(log_folder):
    # Get the current process ID using multiprocessing
    process_id = multiprocessing.current_process().pid
    # Create a timestamp string for the filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(log_folder, exist_ok=True)
    log_filename = os.path.join(
        log_folder,
        f"log_{timestamp}_PID{process_id}.txt",
    )

    # Create a logger for the process
    logger_name = f"process_{process_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Prevent log messages from being propagated to ancestor loggers
    logger.propagate = False

    # Check if the logger already has handlers to avoid adding multiple handlers
    if not logger.handlers:
        # Create a file handler for the process-specific log file
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the file handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - \n%(message)s \n"
        )
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    return logger
