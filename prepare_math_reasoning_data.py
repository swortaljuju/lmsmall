from base_data_preparer import BaseDataPreparer, DatasetConfig, SplitConfig


def content_fetcher1(doc):
    return doc["instruction"] + doc["output"]


def content_fetcher2(doc):
    return doc["text"]

MATH_REASONING_DATA_NAME = "reasoningtokens"

data_preparer = BaseDataPreparer(
    MATH_REASONING_DATA_NAME,
    [
        DatasetConfig(
            "ajibawa-2023/Maths-College",
            data_keys = ["instruction", "output"],
        ),
        DatasetConfig("open-web-math/open-web-math", data_keys=["text"]),
    ],
)

data_preparer.prepare()

data_preparer.split(SplitConfig(8, 1, 1))