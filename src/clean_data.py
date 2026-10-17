"""Create separate, reproducible cleaned job and resume datasets."""

from __future__ import annotations

import csv
import html
import re
import statistics
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
JOBS_INPUT = DATA_DIR / "Jobdesc_data.csv"
RESUMES_INPUT = DATA_DIR / "Resume.csv"
JOBS_OUTPUT = DATA_DIR / "cleaned_jobs.csv"
RESUMES_OUTPUT = DATA_DIR / "cleaned_resumes.csv"
REPORT_OUTPUT = DATA_DIR / "data_quality_report.txt"

JOB_REQUIRED_COLUMNS = {"position_title", "job_description"}
RESUME_REQUIRED_COLUMNS = {"ID", "Resume_str", "Resume_html", "Category"}
PLACEHOLDER_TITLES = {
    "asdf",
    "dummy",
    "dummy job",
    "lorem ipsum",
    "placeholder",
    "placeholder job",
    "sample",
    "sample job",
    "test",
    "test position",
}
NULL_LIKE_VALUES = {"nan", "none", "null", "n/a"}
HTML_ENTITY_PATTERN = re.compile(
    r"&(?:[A-Za-z][A-Za-z0-9]+|#(?:\d+|[xX][0-9a-fA-F]+));"
)
HTML_BLOCK_TAGS = {
    "address",
    "article",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


class _VisibleTextParser(HTMLParser):
    """Extract visible text while omitting markup and script/style content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_tag_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._ignored_tag_depth += 1
        elif not self._ignored_tag_depth and tag in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_tag_depth:
            self._ignored_tag_depth -= 1
        elif not self._ignored_tag_depth and tag in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_tag_depth:
            self.parts.append(data)


def _read_csv(path: Path, required_columns: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        missing_columns = required_columns.difference(columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path.name} has extra CSV fields on line {line_number}")
            rows.append({column: row.get(column) or "" for column in columns})
    return columns, rows


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def _extract_html_text(value: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(value)
    parser.close()
    return html.unescape("".join(parser.parts))


def _clean_resume(row: dict[str, str]) -> str:
    source_text = row["Resume_str"].strip()
    if not source_text:
        source_text = row["Resume_html"]
    return _normalize_whitespace(_extract_html_text(source_text))


def _is_missing(value: str) -> bool:
    return not value.strip() or value.strip().casefold() in NULL_LIKE_VALUES


def _column_statistics(
    rows: list[dict[str, str]], columns: list[str]
) -> dict[str, dict[str, int | float]]:
    results: dict[str, dict[str, int | float]] = {}
    for column in columns:
        lengths = [len(row.get(column, "").strip()) for row in rows]
        results[column] = {
            "min": min(lengths, default=0),
            "median": statistics.median(lengths) if lengths else 0,
            "max": max(lengths, default=0),
        }
    return results


def _exact_duplicate_rows(rows: list[dict[str, str]], columns: list[str]) -> int:
    seen: set[tuple[str, ...]] = set()
    duplicate_count = 0
    for row in rows:
        record = tuple(row.get(column, "") for column in columns)
        if record in seen:
            duplicate_count += 1
        else:
            seen.add(record)
    return duplicate_count


def _text_duplicate_groups(
    rows: list[dict[str, str]], text_getter
) -> list[list[int]]:
    grouped_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        text = text_getter(row)
        if text:
            grouped_rows[text].append(index)
    return [indices for indices in grouped_rows.values() if len(indices) > 1]


def _format_column_statistics(
    stats: dict[str, dict[str, int | float]]
) -> list[str]:
    lines = []
    for column, values in stats.items():
        lines.append(
            f"  {column}: min={values['min']}, median={values['median']}, "
            f"max={values['max']}"
        )
    return lines


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if JOBS_INPUT.resolve() in {
        JOBS_OUTPUT.resolve(),
        RESUMES_OUTPUT.resolve(),
        REPORT_OUTPUT.resolve(),
    } or RESUMES_INPUT.resolve() in {
        JOBS_OUTPUT.resolve(),
        RESUMES_OUTPUT.resolve(),
        REPORT_OUTPUT.resolve(),
    }:
        raise ValueError("Output paths must not overwrite either source CSV")

    job_columns, original_jobs = _read_csv(JOBS_INPUT, JOB_REQUIRED_COLUMNS)
    resume_columns, original_resumes = _read_csv(RESUMES_INPUT, RESUME_REQUIRED_COLUMNS)

    suspected_jobs = []
    cleaned_jobs = []
    for csv_line, row in enumerate(original_jobs, start=2):
        title = row["position_title"].strip()
        description = row["job_description"].strip()
        if not title or not description:
            continue

        if _normalize_whitespace(title).casefold() in PLACEHOLDER_TITLES:
            suspected_jobs.append((csv_line, title, len(description)))
            continue

        cleaned_jobs.append(
            {
                **row,
                "combined_text": " ".join(
                    part
                    for part in (
                        _normalize_whitespace(title),
                        _normalize_whitespace(description),
                    )
                    if part
                ),
            }
        )

    cleaned_resumes = []
    blank_resume_ids = []
    for row in original_resumes:
        cleaned_text = _clean_resume(row)
        if cleaned_text:
            cleaned_resumes.append({**row, "cleaned_resume": cleaned_text})
        else:
            blank_resume_ids.append(row["ID"])

    seen_resume_texts: set[str] = set()
    unique_resumes = []
    duplicate_resume_rows = 0
    for row in cleaned_resumes:
        text = row["cleaned_resume"]
        if text in seen_resume_texts:
            duplicate_resume_rows += 1
            continue
        seen_resume_texts.add(text)
        unique_resumes.append(row)

    _write_csv(JOBS_OUTPUT, [*job_columns, "combined_text"], cleaned_jobs)
    _write_csv(RESUMES_OUTPUT, [*resume_columns, "cleaned_resume"], unique_resumes)

    job_missing = {
        column: sum(_is_missing(row[column]) for row in original_jobs)
        for column in job_columns
    }
    resume_missing = {
        column: sum(_is_missing(row[column]) for row in original_resumes)
        for column in resume_columns
    }
    resume_categories = Counter(row["Category"].strip() for row in original_resumes)
    resume_text_groups = _text_duplicate_groups(
        original_resumes, lambda row: _normalize_whitespace(row["Resume_str"])
    )
    resume_html_markup_rows = sum(
        bool(re.search(r"<\s*[A-Za-z!/][^>]*>", row["Resume_html"]))
        for row in original_resumes
    )
    resume_str_entity_rows = sum(
        bool(HTML_ENTITY_PATTERN.search(row["Resume_str"])) for row in original_resumes
    )
    resume_html_entity_rows = sum(
        bool(HTML_ENTITY_PATTERN.search(row["Resume_html"])) for row in original_resumes
    )
    replacement_char_rows = [
        (row["ID"], sum(value.count("\ufffd") for value in row.values()))
        for row in original_resumes
        if any("\ufffd" in value for value in row.values())
    ]

    job_output_columns = [*job_columns, "combined_text"]
    resume_output_columns = [*resume_columns, "cleaned_resume"]
    report_lines = [
        "DATA QUALITY REPORT",
        "===================",
        "",
        "Source files are unchanged. Job and resume records are kept in separate outputs.",
        "",
        "JOBS",
        "----",
        f"Input: {JOBS_INPUT.relative_to(ROOT)}",
        f"Rows: {len(original_jobs)}",
        f"Columns: {', '.join(job_columns)}",
        f"Exact duplicate input rows: {_exact_duplicate_rows(original_jobs, job_columns)}",
        "Missing/null-like values by column: "
        + (
            ", ".join(f"{column}={count}" for column, count in job_missing.items())
            if any(job_missing.values())
            else "none"
        ),
        "Character lengths (min / median / max):",
        *_format_column_statistics(_column_statistics(original_jobs, job_columns)),
        "Repeated nonblank values:",
    ]

    for column in ("company_name", "position_title", "job_description"):
        if column not in job_columns:
            continue
        value_counts = Counter(row[column].strip() for row in original_jobs if row[column].strip())
        repeated_values = [count for count in value_counts.values() if count > 1]
        report_lines.append(
            f"  {column}: {len(repeated_values)} distinct values repeated; "
            f"{sum(repeated_values)} rows involved"
        )

    short_descriptions = [
        len(row["job_description"].strip()) for row in original_jobs
    ]
    report_lines.extend(
        [
            f"Descriptions shorter than 50 characters: "
            f"{sum(length < 50 for length in short_descriptions)}",
            f"Descriptions shorter than 100 characters: "
            f"{sum(length < 100 for length in short_descriptions)}",
            "Suspected placeholder records excluded (CSV line, title, description length):",
        ]
    )
    if suspected_jobs:
        report_lines.extend(
            f"  line {line_number}: {title!r}, {description_length} characters"
            for line_number, title, description_length in suspected_jobs
        )
    else:
        report_lines.append("  none")
    report_lines.extend(
        [
            f"Output: {JOBS_OUTPUT.relative_to(ROOT)}",
            f"Output columns: {', '.join(job_output_columns)}",
            f"Output rows: {len(cleaned_jobs)}",
            "",
            "RESUMES",
            "-------",
            f"Input: {RESUMES_INPUT.relative_to(ROOT)}",
            f"Rows: {len(original_resumes)}",
            f"Columns: {', '.join(resume_columns)}",
            f"Exact duplicate input rows: {_exact_duplicate_rows(original_resumes, resume_columns)}",
            "Missing/null-like values by column: "
            + (
                ", ".join(f"{column}={count}" for column, count in resume_missing.items())
                if any(resume_missing.values())
                else "none"
            ),
            "Character lengths (min / median / max):",
            *_format_column_statistics(_column_statistics(original_resumes, resume_columns)),
            f"Rows with HTML markup in Resume_html: {resume_html_markup_rows}",
            f"Rows with HTML entities in Resume_str: {resume_str_entity_rows}",
            f"Rows with HTML entities in Resume_html: {resume_html_entity_rows}",
            "Duplicate nonblank Resume_str groups after whitespace normalization: "
            f"{len(resume_text_groups)}",
            "Duplicate Resume_str record IDs:",
        ]
    )
    for group in resume_text_groups:
        ids = [original_resumes[line_number - 2]["ID"] for line_number in group]
        report_lines.append(f"  {', '.join(ids)}")
    if not resume_text_groups:
        report_lines.append("  none")
    report_lines.extend(
        [
            f"Resume rows with Unicode replacement characters: {len(replacement_char_rows)}",
            "Replacement-character record IDs: "
            + (", ".join(record_id for record_id, _ in replacement_char_rows) or "none"),
            "Category counts:",
        ]
    )
    report_lines.extend(
        f"  {category}: {count}"
        for category, count in sorted(resume_categories.items())
    )
    report_lines.extend(
        [
            f"Blank resumes after HTML fallback: {len(blank_resume_ids)}",
            "IDs with no readable text after fallback: "
            + (", ".join(blank_resume_ids) or "none"),
            f"Exact cleaned-resume duplicates removed: {duplicate_resume_rows}",
            f"Output: {RESUMES_OUTPUT.relative_to(ROOT)}",
            f"Output columns: {', '.join(resume_output_columns)}",
            f"Output rows: {len(unique_resumes)}",
            "",
            "Cleaning notes:",
            "- Original columns are retained; cleaned text is added in a separate column.",
            "- Job matching text contains the title followed by the description.",
            "- Resume text uses Resume_str when present and parsed Resume_html as fallback.",
            "- No resume text or contact details are included in this report.",
            "- The source records do not establish that the jobs are verified vacancies.",
        ]
    )

    REPORT_OUTPUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(
        f"Created {JOBS_OUTPUT.relative_to(ROOT)} ({len(cleaned_jobs)} rows), "
        f"{RESUMES_OUTPUT.relative_to(ROOT)} ({len(unique_resumes)} rows), and "
        f"{REPORT_OUTPUT.relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
