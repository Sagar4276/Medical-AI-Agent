print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\utils\\logger.py | Starting file execution")

import logging
import sys
import os
from datetime import datetime

def get_logger(name, level=logging.INFO):
    """
    Set up a logger with the specified name and level
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if logs directory exists
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    log_file = os.path.join(
        logs_dir, 
        f"{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent log propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger