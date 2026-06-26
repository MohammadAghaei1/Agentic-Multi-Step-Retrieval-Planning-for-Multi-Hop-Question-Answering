import ijson


def read_examples(file_path, limit=5):
    examples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for example in ijson.items(file, "item"):
            examples.append(example)

            if len(examples) >= limit:
                break

    return examples