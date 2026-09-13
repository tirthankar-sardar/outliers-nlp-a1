from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

from .outliers_porter import PorterStemmer


BASE_DIR = Path(__file__).resolve().parent
GROUP_NAME = "outliers"
DEFAULT_INPUT = BASE_DIR / "cran.all.1400"
DEFAULT_STOPWORDS = BASE_DIR / "stopwords.txt"
DEFAULT_OUTPUT = BASE_DIR / f"{GROUP_NAME}_processed.all"
EXPECTED_DOCUMENT_IDS = list(range(1, 1401))
DOCUMENT_ID_RE = re.compile(r"\.I\s+(\d+)")
THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
RAW_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*['’]?|\d+(?:\.\d+)?")
WORD_RE = re.compile(r"[a-z]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
FIELD_TAGS = {".T", ".A", ".B", ".W"}


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    # Preserve numeric value before punctuation becomes a boundary.
    text = THOUSANDS_COMMA_RE.sub("", text)
    return RAW_TOKEN_RE.findall(text)


def normalize(tokens: Iterable[str]) -> list[str]:
    normalized = []
    for token in tokens:
        value = unicodedata.normalize("NFKC", token).lower().replace("’", "'")
        if value.endswith("'s") and value[:-2].isalpha():
            value = value[:-2]
        elif value.endswith("'"):
            value = value[:-1]
        value = value.replace("'", "")
        if WORD_RE.fullmatch(value) or NUMBER_RE.fullmatch(value):
            normalized.append(value)
    return normalized


def stem_tokens(tokens: Iterable[str], stemmer: PorterStemmer) -> list[str]:
    return [stemmer.stem(token) if WORD_RE.fullmatch(token) else token for token in tokens]


def remove_stopwords(tokens: Iterable[str], stemmed_stopwords: set[str]) -> list[str]:
    return [token for token in tokens if token not in stemmed_stopwords]


def load_stemmed_stopwords(path: Path, stemmer: PorterStemmer) -> set[str]:
    words = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            raw_word = line.strip()
            if not raw_word:
                raise ValueError(f"empty stop word at {path}:{line_number}")
            values = normalize(tokenize(raw_word))
            if len(values) != 1 or not WORD_RE.fullmatch(values[0]):
                raise ValueError(f"invalid stop word at {path}:{line_number}: {raw_word!r}")
            word = values[0]
            if word in words:
                raise ValueError(f"duplicate stop word at {path}:{line_number}: {word}")
            words.add(word)

    # Documents and stop words must be compared after the same stemming step.
    return set(stem_tokens(words, stemmer))


def _finish_document(
    doc_id: int,
    title_lines: list[str],
    abstract_lines: list[str],
    saw_title: bool,
    saw_abstract: bool,
) -> tuple[int, str]:
    if not saw_title:
        raise ValueError(f"document {doc_id} has no .T section")
    if not saw_abstract:
        raise ValueError(f"document {doc_id} has no .W section")
    return doc_id, " ".join(title_lines + abstract_lines)


def parse_documents(path: Path) -> list[tuple[int, str]]:
    documents = []
    current_id: int | None = None
    section: str | None = None
    title_lines: list[str] = []
    abstract_lines: list[str] = []
    saw_title = False
    saw_abstract = False

    with path.open(encoding="ascii") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            marker = line.strip()
            id_match = DOCUMENT_ID_RE.fullmatch(marker)

            if id_match:
                if current_id is not None:
                    documents.append(
                        _finish_document(
                            current_id,
                            title_lines,
                            abstract_lines,
                            saw_title,
                            saw_abstract,
                        )
                    )
                current_id = int(id_match.group(1))
                section = None
                title_lines = []
                abstract_lines = []
                saw_title = False
                saw_abstract = False
                continue

            if current_id is None:
                if marker:
                    raise ValueError(f"text before first document at {path}:{line_number}")
                continue

            if section == "abstract":
                # Stray tag-only lines in three abstracts do not end the section.
                if marker in FIELD_TAGS:
                    continue
                abstract_lines.append(line)
                continue

            if marker == ".T":
                section = "title"
                saw_title = True
            elif marker in {".A", ".B"}:
                section = "ignored"
            elif marker == ".W":
                section = "abstract"
                saw_abstract = True
            elif section == "title":
                title_lines.append(line)

    if current_id is not None:
        documents.append(
            _finish_document(
                current_id,
                title_lines,
                abstract_lines,
                saw_title,
                saw_abstract,
            )
        )

    document_ids = [doc_id for doc_id, _ in documents]
    if document_ids != EXPECTED_DOCUMENT_IDS:
        raise ValueError("expected sequential document IDs from 1 through 1400")
    return documents


def preprocess_text(
    text: str,
    stemmer: PorterStemmer,
    stemmed_stopwords: set[str],
) -> list[str]:
    tokens = tokenize(text)
    tokens = normalize(tokens)
    tokens = stem_tokens(tokens, stemmer)
    return remove_stopwords(tokens, stemmed_stopwords)


def preprocess_collection(
    input_path: Path,
    stopwords_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    stemmer = PorterStemmer()
    stemmed_stopwords = load_stemmed_stopwords(stopwords_path, stemmer)
    documents = parse_documents(input_path)

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for doc_id, text in documents:
            tokens = preprocess_text(text, stemmer, stemmed_stopwords)
            output.write(f".I {doc_id}\n.S\n")
            output.write(" ".join(tokens) + "\n")

    return len(documents), len(stemmed_stopwords)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the Cranfield collection.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stopwords", type=Path, default=DEFAULT_STOPWORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document_count, stopword_count = preprocess_collection(
            args.input,
            args.stopwords,
            args.output,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"processed {document_count} documents with {stopword_count} stop-word stems "
        f"into {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
