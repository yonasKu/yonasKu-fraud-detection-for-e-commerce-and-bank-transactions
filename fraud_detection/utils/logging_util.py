import logging

def setup_logger(log_file):
    """
    Set up the logger with a specific log file.
    
    Args:
        log_file (str): Path to the log file.

    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

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
