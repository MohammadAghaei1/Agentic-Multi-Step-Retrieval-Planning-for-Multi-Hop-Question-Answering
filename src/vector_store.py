import chromadb


def get_collection(collection_name="hotpotqa_passages"):
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection


def add_passages(collection, model, passages):
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for passage in passages:
        ids.append(passage["passage_id"])
        documents.append(passage["content"])
        metadatas.append({
            "sample_id": passage["sample_id"],
            "title": passage["title"],
            "source": "local_kb"
        })

    embeddings = model.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )


def search_passages(collection, model, query, top_k=3):
    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    search_results = []

    for i in range(len(results["ids"][0])):
        search_results.append({
            "passage_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "sample_id": results["metadatas"][0][i]["sample_id"],
            "source": results["metadatas"][0][i]["source"],
            "score": float(results["distances"][0][i])
        })

    return search_results


def reset_collection(collection_name="hotpotqa_passages"):
    client = chromadb.PersistentClient(path="chroma_db")

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    return client.get_or_create_collection(name=collection_name)