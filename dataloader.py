# -----------------------------------------------------------------------------
import numpy as np
import torch
from data_file_utils import Split, DataFileUtils
from prepare_math_reasoning_data import MATH_REASONING_DATA_NAME


def _load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(
        self,
        micro_batch_size: int,
        sequence_length: int,
        process_rank,
        num_processes,
        master_process: bool,
        split: Split,
        data_name: str,
        num_tokens_to_predict: int = 1,
    ):
        self.__micro_batch_size = micro_batch_size
        self.__sequence_length = sequence_length
        self.__process_rank = process_rank
        self.__num_processes = num_processes
        self.__num_tokens_to_predict = num_tokens_to_predict

        assert split is not None, "split must be specified"
        assert data_name in [MATH_REASONING_DATA_NAME], f"unknown data_name {data_name}"
        self.__data_file_utils = DataFileUtils(data_name)

        # get the shard filenames
        self.__shards = sorted(
            [p.resolve() for p in self.__data_file_utils.fetchDataFiles(split)]
        )
        assert len(self.__shards) > 0, f"no shards found for split {split.value}"
        if master_process:
            print(f"found {len(self.__shards)} shards for split {split.value}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.__current_shard = 0
        self.__tokens = _load_tokens(self.__shards[self.__current_shard])
        self.__current_position = (
            self.__micro_batch_size * self.__sequence_length * self.__process_rank
        )

    def next_batch(self):
        B, T = self.__micro_batch_size, self.__sequence_length
        buf = self.__tokens[
            self.__current_position : self.__current_position
            + B * T
            + self.__num_tokens_to_predict
        ]
        x = (buf[: B * T]).view(B, T)  # inputs
        y = (
            buf[self.__num_tokens_to_predict : B * T + self.__num_tokens_to_predict]
        ).view(
            B, T
        )  # targets
        # advance the position in the tensor
        self.__current_position += B * T * self.__num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.__current_position + (
            B * T * self.__num_processes + self.__num_tokens_to_predict
        ) > len(self.__tokens):
            self.__current_shard = (self.__current_shard + 1) % len(self.__shards)
            self.__tokens = _load_tokens(self.__shards[self.__current_shard])
            self.__current_position = B * T * self.__process_rank
        return x, y
