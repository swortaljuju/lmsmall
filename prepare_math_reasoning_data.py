from base_data_preparer import BaseDataPreparer, DatasetConfig


def content_fetcher1(doc):
    return doc["instruction"] + doc["output"]


def content_fetcher2(doc):
    return doc["text"]


BaseDataPreparer(
    "reasoningtokens",
    [
        DatasetConfig(
            "ajibawa-2023/Maths-College",
            data_keys = ["instruction", "output"],
        ),
        DatasetConfig("open-web-math/open-web-math", data_keys=["text"]),
    ],
).prepare()
