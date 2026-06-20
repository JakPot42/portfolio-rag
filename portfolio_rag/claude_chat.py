import anthropic

from portfolio_rag.bm25_index import Chunk

_QUERY_SYSTEM = """\
You are a precise research assistant answering questions about JakPot42's software portfolio.

Rules:
1. Answer using ONLY the provided context chunks. Do not use any external knowledge.
2. For EVERY factual claim, cite the source in brackets: [Project Name — filename]
3. If multiple chunks confirm a claim, cite all relevant sources.
4. If the context does not contain enough information, say explicitly:
   "Not found in the indexed portfolio files."
5. Never speculate or infer beyond what the source text states.
6. Quote exact specifics when they appear in context: thresholds, library names,
   test counts, line numbers, API endpoints, formula values.
"""

_TRANSCRIPT_SYSTEM = """\
Extract all decisions and action items from the meeting transcript or planning
conversation below. Format them exactly as CLAUDE.md Decision Points sections:

## Decision Points

**Decision: [Short Title]**
What was decided: [one sentence]
Rationale: [why this choice was made]
Impact: [what this affects going forward]

**Action Item: [Short Title]**
What: [specific task]
Owner: [person if mentioned, otherwise "unspecified"]
Timeline: [deadline if mentioned, otherwise "unspecified"]

**Open Question: [Topic]** (for anything discussed but not resolved)
Context: [brief note on what was raised]

Be exhaustive — capture every explicit decision and every committed action item.
"""


def answer_query(
    query: str,
    chunks: list[Chunk],
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    client = anthropic.Anthropic()

    context_blocks = []
    for chunk in chunks:
        context_blocks.append(
            f"SOURCE: [{chunk.project} — {chunk.filename}]\n"
            f"Section: {chunk.section}\n\n"
            f"{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_QUERY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n\n{context}\n\nQuestion: {query}",
                }
            ],
        )
        return msg.content[0].text
    except Exception as exc:
        raise RuntimeError(f"Claude API error: {exc}") from exc


def extract_transcript_decisions(
    transcript: str,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    client = anthropic.Anthropic()

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            system=_TRANSCRIPT_SYSTEM,
            messages=[{"role": "user", "content": f"Transcript:\n\n{transcript}"}],
        )
        return msg.content[0].text
    except Exception as exc:
        raise RuntimeError(f"Claude API error: {exc}") from exc
