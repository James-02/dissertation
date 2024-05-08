import logging
import sys

class Logger:
    """A wrapper class for the Python logging module with colored output."""

    log_levels = {
        0: logging.NOTSET,
        1: logging.DEBUG,
        2: logging.INFO,
        3: logging.WARNING,
        4: logging.ERROR,
        5: logging.CRITICAL
    }

    def __init__(self, name=__name__, level=1, datefmt='%Y-%m-%d %H:%M:%S', log_file=None):
        """
        Initialize the Logger instance.

        Args:
            name (str): The logger name.
            level (int): The logging level {0: NOTSET, 1: DEBUG, 2: INFO, 3: WARNING, 4: ERROR, 5: CRITICAL}.
            datefmt (str): The format for the timestamp in log messages.
            log_file (str): Path to the log file. If None, logging only occurs to console.
        """
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_levels.get(level))
        self.formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt=datefmt)
        self._configure_handlers()

    def _configure_handlers(self):
        """Configure the logging handlers."""
        # Check if a stream handler already exists with the same characteristics
        stream_handler_exists = any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers)
        if not stream_handler_exists:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(stream_handler)

        # Check if a file handler already exists with the same log file path
        if self.log_file:
            file_handler_exists = any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers)
            if not file_handler_exists:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(self.formatter)
                self.logger.addHandler(file_handler)

    def set_level(self, level: int) -> None:
        """Set the logging level."""
        self.logger.setLevel(level)

    def log(self, level: int, message: str) -> None:
        """Log a message with the specified log level."""
        self.logger.log(level, message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(logging.DEBUG, message)

    def info(self, message: str) -> None:
        """Log an informational message."""
        self.log(logging.INFO, message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(logging.WARNING, message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(logging.ERROR, message)

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.log(logging.CRITICAL, message)
