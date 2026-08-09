import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "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",
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=OPENAI_API_KEY,
)

CLASS_NOTE_SYSTEM_PROMPT = (
    "You are a helpful assistant that converts meeting transcripts into clean, structured class notes. "
    "Produce detailed, accurate notes that include definitions, explanations, examples, and key takeaways. "
    "Return only Markdown content without extra explanation."
)

CLASS_NOTE_USER_TEMPLATE = (
    "Convert the transcript below into a detailed Markdown class note document. "
    "Plan the note sections based on the content of the transcript, and include sections such as Title, Summary, Topics Covered, Definitions, Examples, Action Items, Questions, and Additional Notes only when relevant. "
    "Write clear, informative bullets and short paragraphs where needed.\n\n"
    "Transcript:\n{transcript}\n\n"
    "Important: output only valid Markdown with headings and bullets."
)


def build_class_notes_prompt(transcript_text: str) -> str:
    transcript_text = transcript_text.strip()
    if len(transcript_text) > 20000:
        transcript_text = transcript_text[:20000].rsplit("\n", 1)[0]
    return CLASS_NOTE_USER_TEMPLATE.format(transcript=transcript_text)


def create_class_notes(transcript_text: str, model: str = "nvidia/nemotron-3-ultra-550b-a55b") -> str:
    prompt = build_class_notes_prompt(transcript_text)
    logger.info("Requesting class notes generation from LLM; transcript length=%d", len(transcript_text))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLASS_NOTE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=1.0,
        max_tokens=2048,
    )

    if not response.choices:
        logger.warning("LLM response contains no choices")
        return ""

    choice = response.choices[0]
    if hasattr(choice, "message"):
        content = getattr(choice.message, "content", "")
    else:
        content = choice.get("message", {}).get("content", "")

    result = content.strip() if content else ""
    logger.info("LLM returned %d characters of markdown output", len(result))
    return result


def create_class_notes_from_file(transcript_path: Path, output_path: Path, model: str = "nvidia/nemotron-3-ultra-550b-a55b") -> Path:
    transcript_text = transcript_path.read_text(encoding="utf-8", errors="ignore")
    markdown = create_class_notes(transcript_text, model=model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path