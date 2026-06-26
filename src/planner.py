from src.llm_client import generate_answer


def build_planning_prompt(question):
    prompt = f"""
You are a retrieval planning agent.

Decompose the user question into the focused local search queries needed to answer it.

Generate only genuinely different sub-queries.
Do not generate paraphrases, backup queries, or repeated variants of the same information need.
Each query must retrieve one distinct essential piece of evidence.

If the question needs one piece of evidence, return one query.
If it needs multiple reasoning steps, return one query per distinct step.

Return only the search queries, one per line.

Question:
{question}

Search queries:
"""
    return prompt


def parse_planning_output(output):
    queries = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove bullet formatting
        line = line.lstrip("-").strip()

        # Remove simple numbering like "1. query"
        if len(line) > 2 and line[0].isdigit() and line[1] == ".":
            line = line[2:].strip()

        if line:
            queries.append(line)

    return queries


def generate_plan(question):
    prompt = build_planning_prompt(question)
    output = generate_answer(prompt)
    queries = parse_planning_output(output)
    return queries


def build_step_evaluation_prompt(question, query, results):
    evidence_text = ""

    for result in results:
        evidence_text += f"- Title: {result['title']}\n"
        evidence_text += f"  Source: {result['source']}\n"
        evidence_text += f"  Score: {result['score']}\n"
        evidence_text += f"  Text: {result['text']}\n\n"

    prompt = f"""
You are an evidence evaluation agent in a multi-step retrieval pipeline.

Your task is to decide whether the retrieved evidence is useful for answering the current sub-query.

Original user question:
{question}

Current sub-query:
{query}

Retrieved evidence:
{evidence_text}

Return exactly one of the following formats:

GOOD

RETRY_LOCAL: <a better reformulated local search query>

WEB_SEARCH: <a focused web search query>

Decision rules:
- Use GOOD only if the evidence directly answers the current sub-query.
- Use GOOD only if the evidence is about the exact same entity or clearly proves the entity link needed by the question.
- If the evidence mentions a similar name or related entity but does not prove it is the same entity, do not use GOOD.
- Use RETRY_LOCAL if the evidence is related but incomplete, ambiguous, or probably about the wrong entity.
- Use WEB_SEARCH only if the local knowledge base evidence is clearly missing after a better local query would not likely help.
- Do not explain your decision.
- Return only one line.

Decision:
"""
    return prompt


def parse_step_evaluation_output(output):
    output = output.strip()

    if not output:
        return {
            "decision": "UNKNOWN",
            "query": None
        }

    first_line = output.splitlines()[0].strip()

    if first_line == "GOOD":
        return {
            "decision": "GOOD",
            "query": None
        }

    if first_line.startswith("RETRY_LOCAL:"):
        query = first_line.replace("RETRY_LOCAL:", "", 1).strip()
        return {
            "decision": "RETRY_LOCAL",
            "query": query
        }

    if first_line.startswith("WEB_SEARCH:"):
        query = first_line.replace("WEB_SEARCH:", "", 1).strip()
        return {
            "decision": "WEB_SEARCH",
            "query": query
        }

    return {
        "decision": "UNKNOWN",
        "query": None
    }


def evaluate_step_evidence(question, query, results):
    prompt = build_step_evaluation_prompt(question, query, results)
    output = generate_answer(prompt)
    evaluation = parse_step_evaluation_output(output)
    return evaluation