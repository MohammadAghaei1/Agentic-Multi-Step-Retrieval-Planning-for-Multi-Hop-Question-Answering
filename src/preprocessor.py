def context_to_passages(sample):
    passages = []

    sample_id = sample["id"]
    titles = sample["context"]["title"]
    sentences_list = sample["context"]["sentences"]

    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences)

        passage = {
            "passage_id": f"{sample_id}::{title}",
            "sample_id": sample_id,
            "title": title,
            "text": text,
            "content": f"{title}. {text}"
        }

        passages.append(passage)

    return passages


def samples_to_passages(samples):
    all_passages = []

    for sample in samples:
        passages = context_to_passages(sample)
        all_passages.extend(passages)

    return all_passages