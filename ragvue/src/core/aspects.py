from __future__ import annotations
from typing import List, Optional
from ragvue import chat_once, get_openai_client

DEFAULT_MODEL = "gpt-4o-mini"


def get_aspects(
    question: str,
    aspects: Optional[List[str]] = None,
    client=None,
    model: Optional[str] = None,
    max_aspects: int = 5,
) -> List[str]:
    """
    Return minimal aspects for verifying an answer.
      - If `aspects` are given, truncate and return them.
      - Else extract aspects from the question only.
    """
    if aspects:
        return [str(a).strip() for a in aspects][:max_aspects]
    return extract_aspects_from_question_llm(
        question=question,
        max_aspects=max_aspects,
        client=client,
        model=model,
    )


def extract_aspects_from_question_llm(
    question: str,
    max_aspects: int = 5,
    client=None,
    model: Optional[str] = None,
) -> List[str]:
    """
    Extract 1–3 atomic aspects directly implied by the question.
    Aspects describe what needs to be answered, not the answers themselves.
    """
    model = model or DEFAULT_MODEL
    client = client or get_openai_client()

    system_prompt = (
        "You split a question into a small set of atomic aspects.\n"
        "Each aspect describes a distinct information need that must be satisfied to fully answer the question.\n"
        "Do NOT answer the question and do NOT use outside knowledge.\n"
        "Do NOT insert specific names, dates, or facts that are not already written in the question.\n"
        "If the question uses words like 'who', 'when', 'where', etc., keep these words or use a generic descriptor "
        "such as 'author of the book' or 'year of publication'; do not replace them with concrete answers.\n"
        "Aspects must stay in question space, not look like filled-in answers."
    )

    user_prompt = f"""
            Question: {question}
            
            List the minimal aspects that the answer must cover.
            
            Rules:
            - Return 1–3 aspects.
            - Each aspect should be a short phrase (about 3–8 words), not a full sentence.
            - Describe what information is being asked for, not the answer itself.
            - Do NOT introduce new names, dates, or facts that are not explicitly present in the question.
            - Do NOT copy the whole question; break it into smaller information needs if it has multiple parts.
            - No explanation, no extra text.
            
            Format:
            - One aspect per line, starting with a dash.
            
            Example:
            Q: Who wrote the novel '1984' and when was it first published?
            A:
            - who wrote the novel '1984'
            - when the novel '1984' was first published
            """

    out = chat_once(
        client,
        model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    ).strip()

    aspects: List[str] = []
    for line in out.splitlines():
        t = line.strip().lstrip("-*0123456789. ").strip()
        if t:
            aspects.append(t)
    # Truncate to avoid verbosity
    return aspects[:max_aspects]


# Legacy shim for backward compatibility (ignore contexts)
def extract_aspects_llm(
    question: str,
    contexts: List[str],  # ignored
    max_aspects: int = 5,
    client=None,
    model: Optional[str] = None,
) -> List[str]:
    return extract_aspects_from_question_llm(
        question=question,
        max_aspects=max_aspects,
        client=client,
        model=model,
    )


IS_METRIC = False
