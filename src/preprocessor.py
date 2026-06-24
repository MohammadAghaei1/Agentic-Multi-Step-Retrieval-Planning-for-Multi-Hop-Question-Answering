def context_to_passages(sample):
    passages = []

    titles = sample["context"]["title"]
    sentences_by_title = sample["context"]["sentences"]

    for title, sentences in zip(titles, sentences_by_title):
        text = " ".join(sentences)

        passages.append({
            "passage_id": f"{sample['id']}::{title}",
            "sample_id": sample["id"],
            "title": title,
            "text": text,
            "content": f"{title}. {text}"
        })

    return passages