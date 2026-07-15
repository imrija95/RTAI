"""Offline Hugging Face bundle export, download, and loading.

The exporter never uploads. It converts a clean project model checkpoint into
code-free Safetensors plus JSON metadata. Runtime memory, optimizer resumes,
and telemetry are rejected by schema rather than trusted by filename.

Examples:

    uv run --extra hub python -m rtai.model_hub export \
        --checkpoint ckpt.pt --out dist/rtai-poc --name rtai-poc \
        --license apache-2.0

    uv run --extra hub python -m rtai.model_hub download \
        --repo-id ACCOUNT/MODEL --revision COMMIT_SHA --out models/MODEL
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

SCHEMA_VERSION = 1
CONFIG_NAME = "rtai_config.json"
WEIGHTS_NAME = "model.safetensors"


def _require_safetensors():
    try:
        from safetensors.torch import load_model, save_model
    except ImportError as exc:
        raise RuntimeError("install the Hub tools with: uv sync --extra hub") from exc
    return load_model, save_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("cfg"), dict) \
            or not isinstance(payload.get("model"), dict):
        raise ValueError("only clean model checkpoints with cfg/model mappings can be exported")
    forbidden = {"states", "W", "conv", "opt", "optimizer", "replay", "memory"}
    if forbidden.intersection(payload):
        raise ValueError("runtime memory and optimizer state must never enter a model release")
    return payload


def _build_model(family: str, cfg: dict):
    if family == "rtai-srwm":
        from rtai.model import GPTConfig, RTAIModel
        return RTAIModel(GPTConfig(**cfg))
    if family == "fractal":
        from fractal.model import Config, FractalLM
        return FractalLM(Config(**cfg))
    raise ValueError(f"unsupported model family: {family}")


def _family(payload: dict) -> str:
    cfg = payload["cfg"]
    if "n_layer" in cfg:
        return "rtai-srwm"
    if "depth" in cfg and "n_scales" in cfg:
        return "fractal"
    raise ValueError("checkpoint architecture is not recognized")


def _load_legacy_model(payload: dict, family: str):
    model = _build_model(family, payload["cfg"])
    state = dict(payload["model"])
    if family == "fractal" and not any(key.startswith("blocks.") for key in state):
        state.update({"blocks.0." + key[len("block."):]: value
                      for key, value in state.items() if key.startswith("block.")})
    model.load_state_dict(state)
    if family == "fractal" and payload.get("beta_gain") is not None:
        for block, gains in zip(model.blocks, payload["beta_gain"]):
            for cell, gain in zip(block.unit.cells, gains):
                cell.set_beta_gain(gain)
    return model.eval()


def _model_card(name: str, family: str, license_id: str, source_hash: str) -> str:
    return f"""---
library_name: pytorch
license: {license_id}
tags:
  - fast-weights
  - persistent-memory
  - research
---

# {name}

This is an RTAI research checkpoint in the `{family}` family. It uses code-free
Safetensors weights and requires the matching RTAI source revision.

## Status

This generated card is a release draft. Before publication, add the exact training
data, revision, evaluation results, intended uses, limitations, and hardware.

Runtime memory/state files are deliberately excluded because they can encode inputs
observed during operation.

Source checkpoint SHA-256: `{source_hash}`
"""


def export_bundle(checkpoint: str, out: str, name: str, license_id: str,
                  tokenizer: str | None = None) -> Path:
    source = Path(checkpoint)
    destination = Path(out)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    payload = _legacy_payload(source)
    family = _family(payload)
    model = _load_legacy_model(payload, family)
    _, save_model = _require_safetensors()
    weights = destination / WEIGHTS_NAME
    save_model(model, weights)
    weights.chmod(0o644)

    source_hash = _sha256(source)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_family": family,
        "architecture": payload["cfg"],
        "beta_gain": payload.get("beta_gain"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "weights": WEIGHTS_NAME,
        "weights_sha256": _sha256(weights),
        "source_checkpoint_sha256": source_hash,
        "tokenizer": None,
    }
    if tokenizer:
        token_source = Path(tokenizer)
        if not token_source.is_file():
            raise FileNotFoundError(token_source)
        token_name = "tokenizer.json"
        shutil.copy2(token_source, destination / token_name)
        metadata["tokenizer"] = token_name
        metadata["tokenizer_sha256"] = _sha256(destination / token_name)

    (destination / CONFIG_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        _model_card(name, family, license_id, source_hash), encoding="utf-8")
    return destination


def load_bundle(path: str, device):
    directory = Path(path)
    metadata = json.loads((directory / CONFIG_NAME).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported model bundle schema: {metadata.get('schema_version')}")
    weights = directory / metadata["weights"]
    if _sha256(weights) != metadata.get("weights_sha256"):
        raise ValueError("model weights checksum mismatch")
    model = _build_model(metadata["model_family"], metadata["architecture"]).to(device)
    load_model, _ = _require_safetensors()
    missing, unexpected = load_model(model, weights, strict=True, device=str(device))
    if missing or unexpected:
        raise ValueError(f"bundle state mismatch: missing={missing}, unexpected={unexpected}")
    if metadata["model_family"] == "fractal" and metadata.get("beta_gain") is not None:
        for block, gains in zip(model.blocks, metadata["beta_gain"]):
            for cell, gain in zip(block.unit.cells, gains):
                cell.set_beta_gain(gain)
    return model.eval()


def download_bundle(repo_id: str, revision: str, out: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install the Hub tools with: uv sync --extra hub") from exc
    if not revision:
        raise ValueError("--revision is required for reproducible downloads")
    result = snapshot_download(repo_id=repo_id, revision=revision, local_dir=out,
                               allow_patterns=[CONFIG_NAME, WEIGHTS_NAME, "tokenizer.json", "README.md"])
    return Path(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or fetch safe RTAI model bundles")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="convert a clean .pt checkpoint to Safetensors")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--name", required=True)
    export.add_argument("--license", required=True, dest="license_id")
    export.add_argument("--tokenizer")
    download = commands.add_parser("download", help="download a pinned Hugging Face model bundle")
    download.add_argument("--repo-id", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "export":
        result = export_bundle(args.checkpoint, args.out, args.name, args.license_id, args.tokenizer)
    else:
        result = download_bundle(args.repo_id, args.revision, args.out)
    print(result)


if __name__ == "__main__":
    main()
