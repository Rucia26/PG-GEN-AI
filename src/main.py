from __future__ import annotations

import argparse
import logging
from pathlib import Path

from services.llm_calling import create_class_notes

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class ClassNotesCreator:
    def __init__(self, notes_folder: Path | str = Path(__file__).resolve().parent.parent / "Notes"):
        self.notes_folder = Path(notes_folder).resolve()
        self.notes_folder.mkdir(parents=True, exist_ok=True)

    def generate_from_transcript_file(self, transcript_path: Path) -> Path:
        transcript_path = Path(transcript_path)
        transcript_text = transcript_path.read_text(encoding="utf-8", errors="ignore")
        markdown = create_class_notes(transcript_text)
        output_path = self.notes_folder / f"{transcript_path.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown class notes from a meeting transcript using the LLM service."
    )
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt file")
    parser.add_argument(
        "--notes-folder",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Notes",
        help="Folder where generated notes will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    creator = ClassNotesCreator(args.notes_folder)
    output_path = creator.generate_from_transcript_file(args.transcript)
    logger.info("Saved class note: %s", output_path)


if __name__ == "__main__":
    main()
