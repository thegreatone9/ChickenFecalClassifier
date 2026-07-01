import os
import zipfile
import urllib.request as request
from cnnClassifier import logger


class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with info:\n{headers}")
        else:
            logger.info(f"File already exists: {self.config.local_data_file}")

    def extract_zip_file(self):
        """Extract the zip file into the data directory."""
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
