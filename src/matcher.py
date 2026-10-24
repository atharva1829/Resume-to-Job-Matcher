"""Match resume text against the project's precomputed job embeddings."""

from pathlib import Path
from textwrap import shorten

try:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"Missing required dependency: {exc.name}. "
        "Activate the project's existing Python environment and check its dependencies."
    ) from exc


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384


def load_matcher():
    """Load jobs and embeddings, then prepare the model and nearest-neighbor index."""
    project_root = Path(__file__).resolve().parent.parent
    jobs_path = project_root / "data" / "cleaned_jobs.csv"
    embeddings_path = project_root / "data" / "embeddings" / "job_embeddings.npy"

    for path in (jobs_path, embeddings_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required project file was not found: {path}")

    jobs = pd.read_csv(jobs_path)
    if "combined_text" not in jobs.columns:
        raise ValueError(
            "The cleaned jobs file must contain the required 'combined_text' column."
        )

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_SIZE:
        raise ValueError(
            f"Job embeddings must be two-dimensional with {EMBEDDING_SIZE} columns; "
            f"found shape {embeddings.shape}."
        )
    if len(jobs) != len(embeddings):
        raise ValueError(
            f"Job/embedding row count mismatch: {len(jobs)} jobs and "
            f"{len(embeddings)} embeddings."
        )
    if len(jobs) == 0:
        raise ValueError("The cleaned jobs file contains no jobs to match.")

    # Keep CSV row order unchanged so each embedding stays aligned with its job.
    model = SentenceTransformer(MODEL_NAME)
    nearest_neighbors = NearestNeighbors(metric="cosine")
    nearest_neighbors.fit(embeddings)

    return jobs, embeddings, model, nearest_neighbors


def match_resume(resume_text, jobs, model, nearest_neighbors, top_k=5):
    """Return the top matching job details, cosine distances, and similarity scores."""
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text must be a non-empty string.")
    if not isinstance(jobs, pd.DataFrame):
        raise TypeError("'jobs' must be a pandas DataFrame.")
    if "combined_text" not in jobs.columns:
        raise ValueError("The jobs DataFrame must contain 'combined_text'.")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(
        top_k, (int, np.integer)
    ):
        raise ValueError("'top_k' must be a positive integer.")
    if top_k < 1 or top_k > len(jobs):
        raise ValueError(f"'top_k' must be between 1 and {len(jobs)}.")

    fitted_count = getattr(nearest_neighbors, "n_samples_fit_", None)
    if fitted_count is not None and fitted_count != len(jobs):
        raise ValueError(
            f"Job/index row count mismatch: {len(jobs)} jobs and "
            f"{fitted_count} indexed embeddings."
        )

    resume_embedding = np.asarray(
        model.encode(resume_text, normalize_embeddings=True)
    )
    if resume_embedding.ndim == 1:
        resume_embedding = resume_embedding.reshape(1, -1)
    if resume_embedding.shape != (1, EMBEDDING_SIZE):
        raise ValueError(
            f"Resume embedding must have shape (1, {EMBEDDING_SIZE}); "
            f"found {resume_embedding.shape}."
        )

    distances, indices = nearest_neighbors.kneighbors(
        resume_embedding, n_neighbors=int(top_k)
    )

    # Include only columns actually present in the CSV; omit the full text field.
    detail_columns = [column for column in jobs.columns if column != "combined_text"]
    results = jobs.iloc[indices[0]][detail_columns].copy()
    results["cosine_distance"] = distances[0]
    results["similarity_score"] = 1 - distances[0]

    return results.sort_values(
        "similarity_score", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    sample_resume = """
    Fictional software engineer with experience building Python applications,
    designing REST APIs, working with SQL databases, and collaborating on
    cloud-based projects.
    """

    loaded_jobs, job_embeddings, sentence_model, nn_index = load_matcher()
    matches = match_resume(
        sample_resume, loaded_jobs, sentence_model, nn_index, top_k=5
    )
    display_columns = [
        column
        for column in (
            "position_title",
            "company_name",
            "similarity_score",
            "job_description",
        )
        if column in matches.columns
    ]
    display_matches = matches[display_columns].copy()
    if "job_description" in display_matches.columns:
        display_matches["job_description"] = display_matches[
            "job_description"
        ].map(lambda description: shorten(str(description), width=180))
    print(
        display_matches.to_string(
            index=False,
            formatters={"similarity_score": lambda score: f"{score:.2%}"},
        )
    )