import ijson


def read_first_example(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for example in ijson.items(file, "item"):
            return example