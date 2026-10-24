"""Generate normalized embeddings for the cleaned job descriptions."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
JOBS_INPUT = ROOT / "data" / "cleaned_jobs.csv"
EMBEDDINGS_OUTPUT = ROOT / "data" / "embeddings" / "job_embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16


def load_job_texts(path: Path) -> list[str]:
    """Read and validate the combined_text field from the jobs CSV."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "combined_text" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a 'combined_text' column")

            texts = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"{path} has extra CSV fields on line {line_number}")

                text = row.get("combined_text", "")
                if not text or not text.strip():
                    raise ValueError(f"{path} has empty combined_text on line {line_number}")
                texts.append(text)

    except FileNotFoundError as error:
        raise FileNotFoundError(f"Jobs CSV not found: {path}") from error
    except csv.Error as error:
        raise ValueError(f"Unable to parse jobs CSV {path}: {error}") from error

    if not texts:
        raise ValueError(f"No jobs found in {path}")

    return texts


def main() -> int:
    try:
        job_texts = load_job_texts(JOBS_INPUT)
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(
            job_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        EMBEDDINGS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_OUTPUT, embeddings)

    except (OSError, ValueError, RuntimeError) as error:
        print(f"Failed to generate job embeddings: {error}", file=sys.stderr)
        return 1

    print(f"Processed {len(job_texts)} jobs.")
    print(f"Final embedding shape: {embeddings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
