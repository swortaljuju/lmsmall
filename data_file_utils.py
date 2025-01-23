import os
from enum import Enum
from pathlib import Path
import re 

class Split(Enum):
    TRAIN = "training"
    VALIDATION = "validation"
    TEST = "testing"

class DataFileUtils:
    _base_path = "/tmp/lmsmall"
    base_dataset_path = _base_path + "/datasets"
    base_token_path = _base_path + "/tokens"

    def __init__(
        self,
        data_name: str
    ):
        """
        Initializes the BaseDataPreparer.
        Args:
            data_name (str): The name of the data.
        """
        self._data_name = data_name
        
    def createTokenFileNameWithoutSplit(self, shard_index):
        return os.path.join(
            self.base_token_path, f"{self._data_name}_{shard_index:06d}"
        )
    
    def fetchDataFiles(self, split: Split = None):
        if split is None:
            return Path(self.base_token_path).glob(f"{self._data_name}_*.npy")
        else:
            return Path(self.base_token_path).glob(f"{self._data_name}_*_{split.value}.npy")

    def removeSplitSuffix(self):
        matching_files = self.fetchDataFiles()
        for file in matching_files:
            file.rename(Path(f"{file.parent}/{re.sub(
                "|".join(["_" + split.value for split in Split]), '', file.stem)}.npy"))
