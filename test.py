from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.data_loader import read_examples
from src.preprocessor import samples_to_passages
from src.vector_store import reset_collection, add_passages, search_passages
from src.planner import generate_plan, evaluate_step_evidence


DATASET_LIMIT = 5
TOP_K = 2

train_path = Path("data/raw/hotpotqa_train.json")

# 1. Read dataset samples
samples = read_examples(train_path, limit=DATASET_LIMIT)

# 2. Convert selected samples into passages
passages = samples_to_passages(samples)

# 3. Load embedding model
model = SentenceTransformer("embedmodel")

# 4. Build Chroma vector database from selected samples
collection = reset_collection()
add_passages(collection, model, passages)

print("Number of samples:", len(samples))
print("Number of passages:", len(passages))
print("Collection count:", collection.count())

# 5. Pick a difficult sample
sample = samples[3]
question = sample["question"]

print("\nQuestion:")
print(question)
print("Gold answer:", sample["answer"])

# 6. Planner generates as many sub-queries as needed
queries = generate_plan(question)

print("\nGenerated retrieval plan:")
for i, query in enumerate(queries, start=1):
    print(f"Step {i}: {query}")

# 7. For each sub-query, retrieve and evaluate evidence
evidence_memory = []

for query in queries:
    print("\n" + "-" * 80)
    print("Original sub-query:")
    print(query)

    # First local search
    results = search_passages(
        collection=collection,
        model=model,
        query=query,
        top_k=TOP_K,
    )

    print("\nInitial local results:")
    for result in results:
        print("Title:", result["title"])
        print("Distance:", result["score"])
        print("Source:", result["source"])

    # LLM evaluates whether evidence is useful for this sub-query
    evaluation = evaluate_step_evidence(
        question=question,
        query=query,
        results=results,
    )

    print("\nInitial evidence evaluation:")
    print("Decision:", evaluation["decision"])
    print("Suggested query:", evaluation["query"])

    final_query = query
    final_results = results
    final_decision = evaluation["decision"]

    # If evidence is weak, retry locally once before Tavily
    if evaluation["decision"] != "GOOD" and evaluation["query"]:
        retry_query = evaluation["query"]

        print("\nRetrying local search with reformulated query:")
        print(retry_query)

        retry_results = search_passages(
            collection=collection,
            model=model,
            query=retry_query,
            top_k=TOP_K,
        )

        print("\nRetry local results:")
        for result in retry_results:
            print("Title:", result["title"])
            print("Distance:", result["score"])
            print("Source:", result["source"])

        retry_evaluation = evaluate_step_evidence(
            question=question,
            query=retry_query,
            results=retry_results,
        )

        print("\nRetry evidence evaluation:")
        print("Decision:", retry_evaluation["decision"])
        print("Suggested query:", retry_evaluation["query"])

        final_query = retry_query
        final_results = retry_results
        final_decision = retry_evaluation["decision"]

    evidence_memory.append({
        "step": len(evidence_memory) + 1,
        "original_query": query,
        "query": final_query,
        "decision": final_decision,
        "results": final_results,
    })

# 8. Show final step-level evidence memory
print("\n" + "=" * 80)
print("Final evidence memory:")

for item in evidence_memory:
    print("\nStep:", item["step"])
    print("Original query:", item["original_query"])
    print("Final query:", item["query"])
    print("Final decision:", item["decision"])

    for result in item["results"]:
        print("Title:", result["title"])
        print("Distance:", result["score"])
        print("Source:", result["source"])