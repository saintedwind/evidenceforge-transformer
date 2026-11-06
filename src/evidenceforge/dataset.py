from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

_HUB = "https://huggingface.co/datasets"
_SCIFACT = "BeIR/scifact/resolve/b3b5335604bf5ee3c4447671af975ea25143d4f5"
_QRELS = "BeIR/scifact-qrels/resolve/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
_FILES = (
    (
        f"{_HUB}/{_SCIFACT}/corpus/corpus-00000-of-00001.parquet",
        "243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45",
        "corpus.jsonl",
        "f4a47c54e01f9dfe7f0cbbcae0e1df17c5a90d6553f3a9d5b1ba32184545b3d8",
    ),
    (
        f"{_HUB}/{_SCIFACT}/queries/queries-00000-of-00001.parquet",
        "1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034",
        "queries.jsonl",
        "d893cb3883e50906995d70ac86650b034f6231ca2cc4ba93ca75e0fa36b78686",
    ),
    (
        f"{_HUB}/{_QRELS}/test.tsv",
        "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
        "qrels/test.tsv",
        "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
    ),
)


def _verify(path: Path, expected: str) -> None:
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError(f"checksum mismatch for {path.name}")


def download_scifact(target: Path) -> Path:
    """Fetch pinned BEIR files and install the verified dataset as one directory."""
    if target.exists():
        for _, _, name, digest in _FILES:
            _verify(target / name, digest)
        return target

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Install evidenceforge[data] to fetch SciFact") from error

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".scifact-", dir=target.parent) as temporary:
        work = Path(temporary)
        stage = work / "dataset"
        (stage / "qrels").mkdir(parents=True)
        for url, source_digest, name, output_digest in _FILES:
            with urlopen(url, timeout=40) as response:
                payload = response.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise ValueError("dataset file exceeds download limit")
            source = work / "download"
            source.write_bytes(payload)
            _verify(source, source_digest)
            output = stage / name
            if url.endswith(".parquet"):
                rows = parquet.read_table(source).to_pylist()
                with output.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                output.write_bytes(payload)
            _verify(output, output_digest)
        stage.rename(target)
    return target
