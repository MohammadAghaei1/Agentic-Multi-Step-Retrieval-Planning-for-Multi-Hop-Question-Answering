def collect_top_evidence(evidence_memory):
    top_evidence = []

    for item in evidence_memory:
        top_result = item["results"][0]
        top_evidence.append({
            "step": item["step"],
            "query": item["query"],
            "title": top_result["title"],
            "text": top_result["text"],
            "score": top_result["score"]
        })

    return top_evidence


def build_answer_prompt(question, top_evidence):
    evidence_text = ""

    for evidence in top_evidence:
        evidence_text += f"[Evidence {evidence['step']} - {evidence['title']}]\n"
        evidence_text += evidence["text"] + "\n\n"

    prompt = f"""
Answer the question using only the evidence below.

Question:
{question}

Evidence:
{evidence_text}

Final answer:
"""

    return prompt