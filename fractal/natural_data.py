"""Reproducible Natural Cortex tokenizer and sharded English data pipeline.

The pipeline pins every upstream revision, splits documents by a stable hash independently for
each source, exact-deduplicates normalized documents, and writes aligned token, loss-mask, source,
and document-boundary artifacts. Existing output directories and tokenizers are never replaced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
import unicodedata

import numpy as np
import torch

from fractal import chat_format as cf
from fractal import tokenizer as tk


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260716
DEFAULT_TRAIN_TOKENS = 240_000_000
DEFAULT_SHARD_TOKENS = 5_000_000


@dataclass(frozen=True)
class SourceSpec:
    source_id: int
    key: str
    dataset: str
    config: str | None
    revision: str
    kind: str
    weight: float
    license: str
    attribution: str


NATURAL_SOURCES = (
    SourceSpec(
        0, "cosmopedia_v2", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
        "3ba9d605774198c5868892d7a8deda78031a781f", "text", 0.40, "ODC-By-1.0",
        "HuggingFaceTB SmolLM-Corpus, Cosmopedia v2",
    ),
    SourceSpec(
        1, "fineweb_edu_4_5", "HuggingFaceFW/fineweb-edu", None,
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "fineweb", 0.30, "ODC-By-1.0",
        "HuggingFaceFW FineWeb-Edu; Common Crawl terms also apply",
    ),
    SourceSpec(
        2, "wikipedia_en", "wikimedia/wikipedia", "20231101.en",
        "b04c8d1ceb2f5cd4588862100d08de323dccfbaa", "wikipedia", 0.15,
        "CC-BY-SA-3.0 and GFDL", "Wikimedia Foundation English Wikipedia dump 20231101",
    ),
    SourceSpec(
        3, "smol_smoltalk", "HuggingFaceTB/smol-smoltalk", None,
        "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc", "chat", 0.15, "Apache-2.0",
        "HuggingFaceTB Smol-SmolTalk",
    ),
)


@dataclass(frozen=True)
class RenderedDocument:
    text: str
    pieces: tuple[tuple[str, bool], ...]
    is_chat: bool


_SPACE = re.compile(r"[^\S\n]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(_SPACE.sub(" ", line).strip() for line in value.split("\n"))
    return _NEWLINES.sub("\n\n", value).strip()


def render_record(record: dict, spec: SourceSpec) -> RenderedDocument | None:
    """Render one upstream record and apply source-specific quality filtering."""
    if spec.kind == "fineweb":
        score = record.get("int_score")
        if score is None:
            score = record.get("score")
        try:
            if float(score) < 4.0:
                return None
        except (TypeError, ValueError):
            return None
        text = normalize_text(record.get("text") or "")
        return RenderedDocument(text, ((text, True),), False) if text else None
    if spec.kind == "wikipedia":
        title = normalize_text(record.get("title") or "")
        body = normalize_text(record.get("text") or "")
        text = (title + "\n\n" + body).strip()
        return RenderedDocument(text, ((text, True),), False) if body else None
    if spec.kind == "chat":
        segments = cf.turns_from_messages(record.get("messages"))
        text = cf.render(segments)
        pieces = tuple(cf.render_pieces(segments))
        return RenderedDocument(text, pieces, True) if text and pieces else None
    text = normalize_text(record.get("text") or record.get("content") or "")
    return RenderedDocument(text, ((text, True),), False) if text else None


def document_digest(source_key: str, text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_for_digest(source_key: str, digest: str, val_permille: int = 10) -> str:
    bucket = int(hashlib.sha256((source_key + ":" + digest).encode()).hexdigest()[:8], 16) % 1000
    return "val" if bucket < val_permille else "train"


def encode_document(tokenizer, document: RenderedDocument) -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = []
    mask: list[int] = []
    for text, trainable in document.pieces:
        encoded = tokenizer.encode(text, add_special_tokens=False).ids
        ids.extend(encoded)
        mask.extend([1 if trainable else 0] * len(encoded))
    eot = tokenizer.token_to_id(tk.EOT)
    if eot is None:
        raise ValueError(f"tokenizer is missing {tk.EOT}")
    ids.append(eot)
    mask.append(0 if document.is_chat else 1)
    if len(ids) != len(mask):
        raise AssertionError("token/mask alignment failed")
    if not ids or max(ids) >= 65536:
        raise ValueError("Natural Cortex shards require a vocabulary below 65,536 tokens")
    return np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


class ExactDeduper:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS documents (digest TEXT PRIMARY KEY)")
        self.connection.commit()
        self.duplicates = 0

    def add(self, digest: str) -> bool:
        try:
            self.connection.execute("INSERT INTO documents(digest) VALUES (?)", (digest,))
            return True
        except sqlite3.IntegrityError:
            self.duplicates += 1
            return False

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class DatasetSource:
    """Restartable pinned streaming source."""

    def __init__(self, spec: SourceSpec, retries: int = 12):
        self.spec = spec
        self.retries = retries
        self.records_seen = 0
        self.documents_emitted = 0
        self._iterator = None

    def _open(self) -> None:
        from datasets import load_dataset
        kwargs = {
            "path": self.spec.dataset,
            "split": "train",
            "streaming": True,
            "revision": self.spec.revision,
        }
        if self.spec.config:
            kwargs["name"] = self.spec.config
        stream = load_dataset(**kwargs)
        if self.records_seen:
            stream = stream.skip(self.records_seen)
        self._iterator = iter(stream)

    def next_document(self) -> RenderedDocument | None:
        for attempt in range(self.retries + 1):
            try:
                if self._iterator is None:
                    self._open()
                for record in self._iterator:
                    self.records_seen += 1
                    document = render_record(record, self.spec)
                    if document is not None:
                        self.documents_emitted += 1
                        return document
                return None
            except Exception as exc:
                if attempt >= self.retries:
                    raise RuntimeError(f"source {self.spec.key} failed after retries") from exc
                self._iterator = None
                wait = min(30.0, 1.5 * (2 ** min(attempt, 5)))
                print(f"[{self.spec.key}] reconnect after {type(exc).__name__}; wait {wait:.1f}s",
                      flush=True)
                time.sleep(wait)
        return None


class ShardWriter:
    def __init__(self, root: Path, split: str, shard_tokens: int):
        self.root = root
        self.split = split
        self.shard_tokens = int(shard_tokens)
        self.index = -1
        self.position = 0
        self.total_tokens = 0
        self.total_documents = 0
        self.shards: list[dict] = []
        self._tokens = self._mask = self._source = self._docs = None

    def _open(self) -> None:
        self.close_shard()
        self.index += 1
        stem = f"{self.split}-{self.index:05d}"
        token_path = self.root / f"{stem}.tokens.bin"
        mask_path = self.root / f"{stem}.mask.bin"
        source_path = self.root / f"{stem}.source.bin"
        docs_path = self.root / f"{stem}.docs.jsonl"
        self._tokens = open(token_path, "wb")
        self._mask = open(mask_path, "wb")
        self._source = open(source_path, "wb")
        self._docs = open(docs_path, "w", encoding="utf-8")
        self.position = 0
        self.shards.append({
            "tokens": token_path.name,
            "mask": mask_path.name,
            "source": source_path.name,
            "documents": docs_path.name,
            "count": 0,
            "document_count": 0,
        })

    def write(self, ids: np.ndarray, mask: np.ndarray, source_id: int, digest: str) -> None:
        if self._tokens is None or (self.position and self.position + len(ids) > self.shard_tokens):
            self._open()
        ids.tofile(self._tokens)
        mask.tofile(self._mask)
        np.full(len(ids), source_id, dtype=np.uint8).tofile(self._source)
        self._docs.write(json.dumps({
            "start": self.position,
            "end": self.position + len(ids),
            "source_id": int(source_id),
            "digest": digest,
        }, sort_keys=True) + "\n")
        self.position += len(ids)
        self.total_tokens += len(ids)
        self.total_documents += 1
        self.shards[-1]["count"] = self.position
        self.shards[-1]["document_count"] += 1

    def close_shard(self) -> None:
        for handle in (self._tokens, self._mask, self._source, self._docs):
            if handle is not None:
                handle.close()
        self._tokens = self._mask = self._source = self._docs = None

    def close(self) -> None:
        self.close_shard()


def _refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_documents(max_chars: int, seed: int = DEFAULT_SEED):
    sources = [DatasetSource(spec) for spec in NATURAL_SOURCES]
    targets = {source.spec.key: source.spec.weight * max_chars for source in sources}
    counts = {source.spec.key: 0 for source in sources}
    rng = random.Random(seed)
    total = 0
    while total < max_chars:
        deficits = [
            (targets[source.spec.key] - counts[source.spec.key], rng.random(), source)
            for source in sources
        ]
        source = max(deficits, key=lambda row: (row[0], row[1]))[2]
        document = source.next_document()
        if document is None:
            sources.remove(source)
            if not sources:
                break
            continue
        counts[source.spec.key] += len(document.text)
        total += len(document.text)
        yield document.text


def train_tokenizer(output: str, vocab_size: int = 24_000, max_chars: int = 120_000_000,
                    seed: int = DEFAULT_SEED) -> None:
    tk.train_bpe(tokenizer_documents(max_chars, seed), vocab_size, output)


def build_dataset(output_dir: str, tokenizer_path: str,
                  train_tokens: int = DEFAULT_TRAIN_TOKENS,
                  shard_tokens: int = DEFAULT_SHARD_TOKENS,
                  val_permille: int = 10, seed: int = DEFAULT_SEED) -> dict:
    root = Path(output_dir)
    _refuse_nonempty(root)
    tokenizer_path = Path(tokenizer_path)
    tokenizer = tk.load(str(tokenizer_path))
    tk.assert_atomic_special_tokens(tokenizer)
    if tokenizer.get_vocab_size() != 24_000:
        raise ValueError(f"Natural Cortex requires a 24,000-token vocabulary, got "
                         f"{tokenizer.get_vocab_size()}")

    sources = [DatasetSource(spec) for spec in NATURAL_SOURCES]
    source_counts = {
        spec.key: {"train_tokens": 0, "val_tokens": 0, "documents": 0}
        for spec in NATURAL_SOURCES
    }
    targets = {spec.key: int(train_tokens * spec.weight) for spec in NATURAL_SOURCES}
    writers = {
        "train": ShardWriter(root, "train", shard_tokens),
        "val": ShardWriter(root, "val", shard_tokens),
    }
    deduper = ExactDeduper(root / "dedup.sqlite3")
    rng = random.Random(seed)
    try:
        while any(source_counts[key]["train_tokens"] < target for key, target in targets.items()):
            live = [
                source for source in sources
                if source_counts[source.spec.key]["train_tokens"] < targets[source.spec.key]
            ]
            deficits = [
                (
                    (targets[source.spec.key] - source_counts[source.spec.key]["train_tokens"])
                    / max(targets[source.spec.key], 1),
                    rng.random(),
                    source,
                )
                for source in live
            ]
            source = max(deficits, key=lambda row: (row[0], row[1]))[2]
            document = source.next_document()
            if document is None:
                raise RuntimeError(f"source exhausted before its token target: {source.spec.key}")
            digest = document_digest(source.spec.key, document.text)
            if not deduper.add(digest):
                continue
            split = split_for_digest(source.spec.key, digest, val_permille)
            ids, mask = encode_document(tokenizer, document)
            if split == "train":
                remaining = targets[source.spec.key] - source_counts[source.spec.key]["train_tokens"]
                if remaining <= 0:
                    continue
            writers[split].write(ids, mask, source.spec.source_id, digest)
            row = source_counts[source.spec.key]
            row[f"{split}_tokens"] += len(ids)
            row["documents"] += 1
            if writers["train"].total_tokens and writers["train"].total_tokens % 10_000_000 < len(ids):
                print(f"[natural-data] {writers['train'].total_tokens:,}/{train_tokens:,} train tokens",
                      flush=True)
    finally:
        deduper.close()
        for writer in writers.values():
            writer.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "natural-cortex-corpus",
        "seed": seed,
        "val_permille": val_permille,
        "requested_train_tokens": train_tokens,
        "tokenizer": {
            "file": tokenizer_path.name,
            "sha256": _sha256_file(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": list(tk.SPECIAL_TOKENS),
        },
        "sources": [
            {**asdict(spec), **source_counts[spec.key]} for spec in NATURAL_SOURCES
        ],
        "dedup": {"algorithm": "sha256-normalized-exact", "duplicates": deduper.duplicates},
        "splits": {
            split: {
                "tokens": writer.total_tokens,
                "documents": writer.total_documents,
                "shards": writer.shards,
            }
            for split, writer in writers.items()
        },
    }
    temporary = root / ".manifest.json.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, root / "manifest.json")
    return manifest


class NaturalCorpus:
    """Memory-mapped sampler for Natural Cortex shards."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        with open(self.root / "manifest.json", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("kind") != "natural-cortex-corpus":
            raise ValueError(f"not a Natural Cortex corpus: {self.root}")
        self._splits: dict[str, list[dict]] = {}
        for split in ("train", "val"):
            rows = []
            for shard in self.manifest["splits"][split]["shards"]:
                tokens = np.memmap(self.root / shard["tokens"], dtype=np.uint16, mode="r")
                mask = np.memmap(self.root / shard["mask"], dtype=np.uint8, mode="r")
                source = np.memmap(self.root / shard["source"], dtype=np.uint8, mode="r")
                if not (len(tokens) == len(mask) == len(source) == int(shard["count"])):
                    raise ValueError(f"misaligned shard: {shard['tokens']}")
                rows.append({"meta": shard, "tokens": tokens, "mask": mask, "source": source})
            self._splits[split] = rows

    def batch(self, split: str, batch_size: int, seq_len: int, device,
              rng: np.random.RandomState, *, require_loss: bool = True,
              source_id: int | None = None, max_attempts: int = 512):
        shards = [row for row in self._splits[split] if len(row["tokens"]) > seq_len + 1]
        if not shards:
            raise ValueError(f"no {split} shard is long enough for sequence length {seq_len}")
        weights = np.asarray([len(row["tokens"]) - seq_len - 1 for row in shards], dtype=np.float64)
        weights /= weights.sum()
        xs, ys, ws, labels = [], [], [], []
        for _ in range(batch_size):
            for _attempt in range(max_attempts):
                row = shards[int(rng.choice(len(shards), p=weights))]
                start = int(rng.randint(0, len(row["tokens"]) - seq_len - 1))
                target_mask = row["mask"][start + 1:start + seq_len + 1]
                sources = row["source"][start:start + seq_len]
                # The label is the document/source at the sampled start. A window may cross an EOT
                # into the next interleaved document, which is intentional for the continuous stream.
                label = int(sources[0])
                if require_loss and not target_mask.any():
                    continue
                if source_id is not None and label != int(source_id):
                    continue
                xs.append(row["tokens"][start:start + seq_len].astype(np.int64))
                ys.append(row["tokens"][start + 1:start + seq_len + 1].astype(np.int64))
                ws.append(target_mask.astype(np.float32))
                labels.append(label)
                break
            else:
                raise RuntimeError("failed to sample a qualifying Natural Cortex window")
        return (
            torch.from_numpy(np.stack(xs)).to(device),
            torch.from_numpy(np.stack(ys)).to(device),
            torch.from_numpy(np.stack(ws)).to(device),
            torch.tensor(labels, dtype=torch.long, device=device),
        )


def verify_dataset(root: str, tokenizer_path: str) -> dict:
    corpus = NaturalCorpus(root)
    tokenizer = tk.load(tokenizer_path)
    tk.assert_atomic_special_tokens(tokenizer)
    if corpus.manifest["tokenizer"]["sha256"] != _sha256_file(Path(tokenizer_path)):
        raise ValueError("tokenizer checksum does not match the corpus manifest")
    rng = np.random.RandomState(corpus.manifest["seed"])
    x1 = corpus.batch("val", 2, 32, "cpu", rng)
    rng = np.random.RandomState(corpus.manifest["seed"])
    x2 = corpus.batch("val", 2, 32, "cpu", rng)
    if not all(torch.equal(left, right) for left, right in zip(x1, x2)):
        raise AssertionError("validation sampling is not reproducible")
    return {
        "tokenizer_atomic": True,
        "validation_reproducible": True,
        "train_tokens": corpus.manifest["splits"]["train"]["tokens"],
        "val_tokens": corpus.manifest["splits"]["val"]["tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    tokenizer_parser = sub.add_parser("tokenizer")
    tokenizer_parser.add_argument("--out", default="natural_tokenizer_24k.json")
    tokenizer_parser.add_argument("--vocab-size", type=int, default=24_000)
    tokenizer_parser.add_argument("--max-chars", type=int, default=120_000_000)
    tokenizer_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--out-dir", default="natural_data_240m")
    build_parser.add_argument("--tokenizer", required=True)
    build_parser.add_argument("--train-tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    build_parser.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    build_parser.add_argument("--val-permille", type=int, default=10)
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--data-dir", required=True)
    verify_parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    if args.command == "tokenizer":
        train_tokenizer(args.out, args.vocab_size, args.max_chars, args.seed)
    elif args.command == "build":
        build_dataset(args.out_dir, args.tokenizer, args.train_tokens, args.shard_tokens,
                      args.val_permille, args.seed)
    else:
        print(json.dumps(verify_dataset(args.data_dir, args.tokenizer), indent=2))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # datasets/pyarrow streaming workers can still be tearing down after a successful command.
    # Exit after all files and manifests are closed to avoid a known interpreter-finalization race.
    os._exit(0)
