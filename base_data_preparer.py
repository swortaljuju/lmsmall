import os
import multiprocessing as mp
import numpy as np
import tiktoken
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict
from functools import partial
import re 
from data_file_utils import DataFileUtils, Split


@dataclass
class DatasetConfig:
    dataset_name: str
    data_keys: List[str]
    dataset_split: str = "train"


# This class is used to configure the split of the data into training, validation and testing sets. The partition should be a number between 0 and 10.
@dataclass
class SplitConfig:
    training_data_partition: int
    validation_data_partition: int
    testing_data_partition: int


# This is the base class for all data preparers. It is responsible for preparing the data for the model.
class BaseDataPreparer:
    _shared_size = int(1e8)  # 100M tokens per shard

    def __init__(
        self,
        data_name: str,
        dataset_config_list: List[DatasetConfig],
        tokenizer="gpt2",
    ):
        """
        Initializes the BaseDataPreparer.
        Args:
            data_name (str): The name of the data.
            dataset_config_list (List[DatasetConfig]): A list of dataset configurations.
            tokenizer (str, optional): The tokenizer to use. Defaults to "gpt2".
        """
        self._data_file_utils = DataFileUtils(data_name)
        self._encoder = tiktoken.get_encoding(tokenizer)
        self._dataset_config_list = dataset_config_list

    def _tokenize(self, doc: Dict, data_keys: List[str]) -> np.ndarray:
        # tokenizes a single document and returns a numpy array of uint16 tokens
        tokens = [
            self._encoder._special_tokens["<|endoftext|>"]
        ]  # the special <|endoftext|> token delimits all documents
        tokens.extend(
            self._encoder.encode_ordinary(" ".join([doc[key] for key in data_keys]))
        )
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (
            tokens_np < 2**16
        ).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def __write_datafile(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def prepare(self):
        os.makedirs(DataFileUtils.base_token_path, exist_ok=True)
        # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
        nprocs = max(1, os.cpu_count() // 2)
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            # preallocate buffer to hold current shard
            all_tokens_np = np.empty((BaseDataPreparer._shared_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            for dataset_config in self._dataset_config_list:
                fw = load_dataset(
                    dataset_config.dataset_name,
                    split=dataset_config.dataset_split,
                    cache_dir=DataFileUtils.base_dataset_path,
                )
                for tokens in pool.imap(
                    partial(self._tokenize, data_keys=dataset_config.data_keys),
                    fw,
                    chunksize=16,
                ):
                    # is there enough space in the current shard for the new tokens?
                    if token_count + len(tokens) < BaseDataPreparer._shared_size:
                        # simply append tokens to current shard
                        all_tokens_np[token_count : token_count + len(tokens)] = tokens
                        token_count += len(tokens)
                        # update progress bar
                        if progress_bar is None:
                            progress_bar = tqdm(
                                total=BaseDataPreparer._shared_size,
                                unit="tokens",
                                desc=f"Shard {shard_index}",
                            )
                        progress_bar.update(len(tokens))
                    else:
                        # write the current shard and start a new one
                        filename = self._data_file_utils.createTokenFileNameWithoutSplit(shard_index)
                        # split the document into whatever fits in this shard; the remainder goes to next one
                        remainder = BaseDataPreparer._shared_size - token_count
                        progress_bar.update(remainder)
                        all_tokens_np[token_count : token_count + remainder] = tokens[
                            :remainder
                        ]
                        self.__write_datafile(filename, all_tokens_np)
                        shard_index += 1
                        progress_bar = None
                        # populate the next shard with the leftovers of the current doc
                        all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                        token_count = len(tokens) - remainder

            # write any remaining tokens as the last shard
            if token_count != 0:
                filename = self._data_file_utils.createTokenFileNameWithoutSplit(shard_index)
                self.__write_datafile(filename, all_tokens_np[:token_count])

    # Split the data into train, validation and test sets
    # This will only split the file shards, not all rows in the dataset
    def split(self, split_config: SplitConfig):
        assert (
            split_config.training_data_partition
            + split_config.validation_data_partition
            + split_config.testing_data_partition
            == 10
        ), "The sum of the training, validation and testing data percentages must be 10"
        self._data_file_utils.removeSplitSuffix()
        matching_files = self.fetchDataFiles()
        validation_and_training_partition = (
            split_config.validation_data_partition
            + split_config.training_data_partition
        )
        for i, file in enumerate(matching_files):
            rem = i % 10
            if rem < split_config.training_data_partition:
                split = Split.TRAIN
            elif rem < validation_and_training_partition:
                split = Split.VALIDATION
            else:
                split = Split.TEST
            new_file_name = Path(f"{file.parent}/{file.stem}_{split}.npy") 
            print(f"renaming {file.name} to {new_file_name}")   
            file.rename(new_file_name)
