import logging

class Logger:
    """A wrapper class for the Python logging module with colored output."""

    log_levels = {
        0 : logging.NOTSET,
        1 : logging.DEBUG,
        2 : logging.INFO,
        3 : logging.WARNING,
        4 : logging.ERROR,
        5 : logging.CRITICAL
    }
    
    def __init__(self, name=__name__, level=log_levels[1], datefmt='%Y-%m-%d %H:%M:%S', log_file=None):
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
        self._configure_handler(datefmt, self.log_file)

    def _configure_handler(self, datefmt, log_file):
        """Configure the logging handlers."""
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt=datefmt)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def set_level(self, level: int) -> None:
        """Set the logging level."""
        self.logger.setLevel(level)

    def _colorize(self, message: str, color: str) -> str:
        """Colorize the log message."""
        colors = {
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'reset': '\033[0m'
        }
        return f"{colors[color]}{message}{colors['reset']}"

    def _get_colored_levelname(self, levelname: str) -> str:
        """Get the colored log level name."""
        level_colors = {
            'DEBUG': 'blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'magenta'
        }
        return self._colorize(levelname, level_colors.get(levelname, 'reset'))

    def log(self, level: int, message: str) -> None:
        """Log a message with the specified log level."""
        if self.log_file:
            self.logger.log(level, f"{logging.getLevelName(level)} - [{self.logger.name}] - {message}")
        else:
            colored_levelname = self._get_colored_levelname(logging.getLevelName(level))
            self.logger.log(level, f"{colored_levelname} - [{self.logger.name}] - {message}")

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
