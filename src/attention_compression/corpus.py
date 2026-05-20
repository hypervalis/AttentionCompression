from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CorpusInspection:
    data_dir: str
    file_glob: str
    file_count: int
    total_bytes: int
    sample_file: str | None
    sample_keys: list[str]
    text_field: str
    text_present: bool
    metadata_keys: list[str]


def discover_corpus_files(data_dir: str | Path, file_glob: str = "*.json") -> list[Path]:
    paths = sorted(Path(data_dir).glob(file_glob))
    if not paths:
        raise FileNotFoundError(f"No corpus files matching {file_glob!r} under {data_dir}")
    return paths


def iter_jsonl_documents(
    paths: Iterable[str | Path],
    *,
    text_field: str = "text",
) -> Iterable[tuple[Path, int, dict[str, Any], str]]:
    """Yield JSONL documents as `(path, line_number, object, text)`.

    Dolma sample shards are newline-delimited JSON with the document body under
    `text`. Keeping this as a small generator gives the tokenizer a stable
    streaming reader and makes format assumptions explicit.
    """
    for path_like in paths:
        path = Path(path_like)
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get(text_field)
                if not isinstance(text, str):
                    continue
                yield path, line_number, obj, text


def inspect_jsonl_corpus(
    *,
    data_dir: str | Path,
    file_glob: str = "*.json",
    text_field: str = "text",
) -> CorpusInspection:
    paths = discover_corpus_files(data_dir, file_glob)
    total_bytes = sum(path.stat().st_size for path in paths)
    sample_obj: dict[str, Any] | None = None
    for _, _, obj, _ in iter_jsonl_documents(paths[:1], text_field=text_field):
        sample_obj = obj
        break

    sample_keys = sorted(sample_obj.keys()) if sample_obj else []
    metadata = sample_obj.get("metadata") if sample_obj else None
    metadata_keys = sorted(metadata.keys()) if isinstance(metadata, dict) else []
    return CorpusInspection(
        data_dir=str(data_dir),
        file_glob=file_glob,
        file_count=len(paths),
        total_bytes=total_bytes,
        sample_file=str(paths[0]) if paths else None,
        sample_keys=sample_keys,
        text_field=text_field,
        text_present=bool(sample_obj and isinstance(sample_obj.get(text_field), str)),
        metadata_keys=metadata_keys,
    )


def write_inspection(path: str | Path, inspection: CorpusInspection) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(asdict(inspection), f, indent=2)
