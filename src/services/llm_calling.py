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
    "You are a meticulous scribe that converts class/meeting transcripts into comprehensive, detailed Markdown class notes. "
    "Your notes must preserve the full depth of the discussion: every concept explained, every example given, every "
    "definition, every question and answer, and every action item. Do not condense the material into a brief summary — "
    "write thorough notes detailed enough for a student to study from without having attended the class. "
    "Return only Markdown content without extra explanation."
)

CLASS_NOTE_USER_TEMPLATE = (
    "Convert the transcript below into detailed, comprehensive Markdown class notes. Do NOT write a short summary — write "
    "full notes that capture everything of substance that was said: concepts, definitions, explanations, examples, "
    "step-by-step walkthroughs, questions asked and answers given, and action items. Preserve technical detail and "
    "reasoning, not just conclusions.\n\n"
    "Plan the note sections based on the actual content of the transcript, for example Title, Topics Covered, Detailed "
    "Notes by Topic, Definitions, Examples, Questions & Answers, Action Items, and Additional Notes — include only the "
    "sections that are relevant, and skip any 'Summary' section in favor of full detailed coverage. "
    "Use clear headings, detailed bullets, and short paragraphs where a concept needs explanation rather than a one-line bullet.\n\n"
    "Transcript:\n{transcript}\n\n"
    "Important: output only valid Markdown with headings and bullets. Prioritize completeness and depth over brevity."
)

MAX_TRANSCRIPT_CHARS = 14000


def build_class_notes_prompt(transcript_text: str) -> str:
    transcript_text = transcript_text.strip()
    return CLASS_NOTE_USER_TEMPLATE.format(transcript=transcript_text)


def split_transcript_text(transcript_text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> list[str]:
    transcript_text = transcript_text.strip()
    if len(transcript_text) <= max_chars:
        return [transcript_text]

    paragraphs = [p.strip() for p in transcript_text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            # fallback split long paragraphs by lines when a single paragraph is too long
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            for line in lines:
                if len(current_chunk) + len(line) + 2 > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = line + "\n\n"
                else:
                    current_chunk += line + "\n\n"
        elif len(current_chunk) + len(paragraph) + 2 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_class_notes_for_long_transcript(transcript_text: str, model: str) -> str:
    chunks = split_transcript_text(transcript_text)
    if len(chunks) == 1:
        return create_class_notes(chunks[0], model=model)

    logger.info("Transcript split into %d chunks for LLM processing", len(chunks))
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        logger.info("Generating notes for chunk %d/%d", index, len(chunks))
        chunk_notes = create_class_notes(chunk, model=model)
        parts.append(f"## Segment {index}\n\n{chunk_notes}")

    combined = ["# Combined Class Notes", "", "Generated from a long transcript split into multiple chunks.", ""]
    combined.extend(parts)
    return "\n\n".join(combined)


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
        max_tokens=8192,
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
    markdown = create_class_notes_for_long_transcript(transcript_text, model=model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path