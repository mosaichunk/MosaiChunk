"""Record generated media locally without an external tracking service."""

import json
from pathlib import Path

from mosaichunk.runtime import record_environment_on_exit


class LocalMedia:
    def __init__(self, config):
        self.config = config
        record_environment_on_exit()

    def before_configuration(self, logger, **kwargs):
        self.logger = logger
        logger.inject_methods(self, {"log_video": self.log_video})

    def log_video(self, videos, step=None, **kwargs):
        path = self.logger.iter_dir("media", step) / "index.json"
        path.write_text(
            json.dumps({k: str(Path(v).resolve()) for k, v in videos.items()}, indent=2) + "\n"
        )
