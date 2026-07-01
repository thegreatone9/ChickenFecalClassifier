import os
import yaml
import json
import base64
from box.exceptions import BoxValueError
from box import ConfigBox
from pathlib import Path
from cnnClassifier import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return a ConfigBox object."""
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")


def create_directories(path_to_directories: list, verbose=True):
    """Create a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict):
    """Save a dictionary as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


def decodeImage(imgstring, fileName):
    """Decode a base64 image string and write to file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
