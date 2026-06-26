from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.data_loader import read_examples
from src.preprocessor import context_to_passages
from src.retriever import embed_passages, search
from src.planner import generate_plan
from src.synthesizer import collect_top_evidence, build_answer_prompt
from src.llm_client import generate_answer
from src.vector_store import get_collection

def run_pipeline(sample, model):
    print("\n" + "=" * 80)
    print("Question:")
    print(sample["question"])

    passages = context_to_passages(sample)
    passage_embeddings = embed_passages(model, passages)

    queries = generate_plan(sample["question"])

    print("\nGenerated retrieval plan:")
    for i, query in enumerate(queries, start=1):
        print(f"Step {i}: {query}")

    evidence_memory = []

    for query in queries:
        results = search(
            query=query,
            model=model,
            passages=passages,
            passage_embeddings=passage_embeddings,
            top_k=2
        )

        evidence_memory.append({
            "step": len(evidence_memory) + 1,
            "query": query,
            "results": results
        })

    top_evidence = collect_top_evidence(evidence_memory)

    print("\nRetrieved evidence:")
    for evidence in top_evidence:
        print(f"\nStep {evidence['step']}")
        print("Query:", evidence["query"])
        print("Title:", evidence["title"])
        print("Score:", evidence["score"])
        print("Text:", evidence["text"])

    answer_prompt = build_answer_prompt(sample["question"], top_evidence)
    final_answer = generate_answer(answer_prompt)

    print("\nGenerated final answer:")
    print(final_answer)

    print("\nGold answer:")
    print(sample["answer"])


def main():
    train_path = Path("data/raw/hotpotqa_train.json")

    samples = read_examples(train_path, limit=5)

    model = SentenceTransformer("embedmodel")

    for sample in samples:
        run_pipeline(sample, model)


if __name__ == "__main__":
    main()