from sklearn.metrics.pairwise import cosine_similarity


def embed_passages(model, passages):
    texts = [passage["content"] for passage in passages]
    embeddings = model.encode(texts)
    return embeddings

from sklearn.metrics.pairwise import cosine_similarity


def search(query, model, passages, passage_embeddings, top_k=3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, passage_embeddings)[0]

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for index in ranked_indices:
        result = passages[index].copy()
        result["score"] = float(scores[index])
        results.append(result)

    return results