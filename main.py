from pathlib import Path
from src.data_loader import read_first_example
from src.preprocessor import context_to_passages
from sentence_transformers import SentenceTransformer
from src.retriever import embed_passages
from src.retriever import search
from src.synthesizer import collect_top_evidence
from src.synthesizer import build_answer_prompt

train_path = Path("data/raw/hotpotqa_train.json")

sample = read_first_example(train_path)

print("Question:", sample["question"])
print("Answer:", sample["answer"])

print("Context titles:", sample["context"]["title"])
print("Number of context documents:", len(sample["context"]["title"]))

passages = context_to_passages(sample)

print("Number of passages:", len(passages))
print("First passage title:", passages[0]["title"])
print("First passage text:", passages[0]["text"])
print("First passage id:", passages[0]["passage_id"])
print("First passage content:", passages[0]["content"])

model = SentenceTransformer("embedmodel")

embedding = model.encode(passages[0]["content"])

print("Embedding size:", embedding.shape)
print("First 5 embedding values:", embedding[:5])


passage_embeddings = embed_passages(model, passages)
print("Passage embeddings shape:", passage_embeddings.shape)


queries = [
    "When was Arthur's Magazine started?",
    "When was First for Women started?"
]

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

print("\nEvidence memory:")
for item in evidence_memory:
    print("Step:", item["step"])
    print("Query:", item["query"])
    print("Top evidence:", item["results"][0]["title"])



print("\nTop evidence texts:")
for item in evidence_memory:
    top_result = item["results"][0]
    print("\nStep:", item["step"])
    print("Query:", item["query"])
    print("Title:", top_result["title"])
    print("Text:", top_result["text"])


top_evidence = collect_top_evidence(evidence_memory)
print("\nCollected top evidence:")
for evidence in top_evidence:
    print("Step:", evidence["step"])
    print("Title:", evidence["title"])
    print("Score:", evidence["score"])



answer_prompt = build_answer_prompt(sample["question"], top_evidence)
print("\nAnswer prompt:")
print(answer_prompt)