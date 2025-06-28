import logging
import sys
import time
import os

class Logger:
    """A class to make logging easier
    """
    def __init__(self, name, filename):
        """Initialise logger

        Args:
            name (str): name of experiment
            filename (str): File to output log to
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s|%(name)s|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        if not os.path.isfile(filename):
            open(filename, "w").close()
        
        file_handler = logging.FileHandler(filename)
        self.logger.addHandler(file_handler)

        # Instead of print(msg) adding this handler automatically prints the message with the correct format
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

    def log(self, msg):
        """log a given message

        Args:
            msg (str): message to log
        """
        self.logger.info(msg)
    
    def __del__(self):
        self.logger.info("Run Complete \n\n")


if __name__ == "__main__":
    log = Logger("TEST","/home/crae/CompSim/project_v2/orbital_motion.log")
    log.log("Hello World")