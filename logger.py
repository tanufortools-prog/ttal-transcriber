import os
import sys
import time
import logging
import traceback
from typing import Any, Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "ttal_app.log")

# Setup root logger with file + console handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers to prevent duplicates
if root_logger.hasHandlers():
    root_logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)
console_handler.setFormatter(console_formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

class FeatureLogger:
    """
    Feature-oriented logger providing structured lifecycle tracking, execution timing,
    payload inspection, and stack trace logging for easier debugging.
    """
    def __init__(self, feature_name: str):
        self.feature_name = feature_name.upper()
        self.logger = logging.getLogger(f"FEATURE:{self.feature_name}")
        self._start_time: Optional[float] = None

    def start(self, step: str, details: Optional[str] = None):
        """Mark feature operation start."""
        self._start_time = time.time()
        msg = f"[STEP:START] [{step}]"
        if details:
            msg += f" - {details}"
        self.logger.info(msg)

    def progress(self, step: str, details: str):
        """Log intermediate feature progress."""
        self.logger.info(f"[STEP:PROGRESS] [{step}] - {details}")

    def debug(self, step: str, details: Any):
        """Log granular payload / debug telemetry."""
        self.logger.debug(f"[STEP:DEBUG] [{step}] - {details}")

    def warning(self, step: str, details: str):
        """Log non-fatal warning or fallback trigger."""
        self.logger.warning(f"[STEP:WARNING] [{step}] - {details}")

    def complete(self, step: str, details: Optional[str] = None):
        """Mark feature operation completion with duration."""
        elapsed_ms = round((time.time() - self._start_time) * 1000, 2) if self._start_time else 0
        msg = f"[STEP:COMPLETE] [{step}] (took {elapsed_ms}ms)"
        if details:
            msg += f" - {details}"
        self.logger.info(msg)

    def error(self, step: str, error: Exception, context: Optional[str] = None):
        """Log feature failure with full traceback for debugging."""
        msg = f"[STEP:ERROR] [{step}] Failure: {str(error)}"
        if context:
            msg += f" | Context: {context}"
        self.logger.error(msg)
        self.logger.error(f"[TRACEBACK]\n{traceback.format_exc()}")

def get_logger(feature_name: str) -> FeatureLogger:
    """Returns a FeatureLogger instance for the given feature tag."""
    return FeatureLogger(feature_name)
