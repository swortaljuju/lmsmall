from base_data_preparer import BaseDataPreparer, DatasetConfig, SplitConfig


def content_fetcher1(doc):
    return doc["instruction"] + doc["output"]


def content_fetcher2(doc):
    return doc["text"]


data_preparer = BaseDataPreparer(
    "reasoningtokens",
    [
        DatasetConfig(
            "ajibawa-2023/Maths-College",
            data_keys = ["instruction", "output"],
        ),
        DatasetConfig("open-web-math/open-web-math", data_keys=["text"]),
    ],
)

# data_preparer.prepare()

data_preparer.split(SplitConfig(8, 1, 1))