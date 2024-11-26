import logging
import os

def setup_logger(log_file):
    """
    Set up the logger with a specific log file.
    
    Args:
        log_file (str): Path to the log file.

    Returns:
        logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):  # Avoid issues if log_dir is empty
        os.makedirs(log_dir)

    logger = logging.getLogger(log_file)  # Use log_file as the logger's name for uniqueness
    logger.setLevel(logging.DEBUG)  # Capture all levels of logs (DEBUG, INFO, etc.)

    # Check if handlers already exist (to prevent duplicate logs)
    if not logger.handlers:
        # Create file handler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler for logging to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
