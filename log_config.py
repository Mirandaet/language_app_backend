import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    # Ensure the logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, 'app.log'), 
        maxBytes=10485760, 
        backupCount=5
    )

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    return logger