"""Local Natural Cortex runtime with explicit, auditable skill teaching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import torch

from fractal import agent
from fractal import chat_format as cf
from fractal import persist
from fractal import tokenizer as tk
from fractal.growing_cortex import (
    expert_snapshot,
    restore_expert,
    teach_expert,
)


SKILL_BANK_SCHEMA = 1


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


class SkillBank:
    """Versioned append-only expert tensors plus mutable lifecycle pointers."""

    def __init__(self, root: str | os.PathLike[str], model, base_checkpoint: str):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.audit_path = self.root / "audit.jsonl"
        self.experts_path = self.root / "experts"
        self.address_path = self.root / "address"
        self.model = model
        self.base_checkpoint = str(base_checkpoint)
        if model.skill_cortex is None or model.skill_cortex.compiler is not None:
            raise ValueError("Natural Cortex requires a compiler-free skill bank")
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as handle:
                self.manifest = json.load(handle)
            self._validate()
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise FileExistsError(f"refusing to initialize a skill bank in non-empty {self.root}")
            self.experts_path.mkdir(parents=True, exist_ok=True)
            self.address_path.mkdir(parents=True, exist_ok=True)
            cortex = model.skill_cortex
            self.manifest = {
                "schema_version": SKILL_BANK_SCHEMA,
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "router": {
                    "threshold": cortex.router_threshold,
                    "address_dim": cortex.address_dim,
                    "compiler": cortex.compiler_mode,
                    "auto_route": False,
                },
                "address_encoder_revision": None,
                "experts": [],
                "created_at": time.time(),
            }
            self._save_manifest()
        self.load_into_model()

    def _validate(self) -> None:
        if self.manifest.get("schema_version") != SKILL_BANK_SCHEMA:
            raise ValueError("unsupported Natural Cortex skill-bank schema")
        if self.manifest.get("base_checkpoint_sha256") != _sha256(self.base_checkpoint):
            raise ValueError("skill bank belongs to a different immutable base checkpoint")
        router = self.manifest.get("router") or {}
        cortex = self.model.skill_cortex
        if router.get("compiler") != "none" or int(router.get("address_dim", -1)) != cortex.address_dim:
            raise ValueError("skill-bank address configuration does not match the model")

    def _save_manifest(self) -> None:
        _atomic_json(self.manifest_path, self.manifest)

    def audit(self, operation: str, **payload) -> None:
        event = {
            "event_id": uuid.uuid4().hex,
            "time": time.time(),
            "operation": operation,
            **payload,
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.audit_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_into_model(self) -> None:
        cortex = self.model.skill_cortex
        if len(cortex.experts):
            raise ValueError("base checkpoint already contains skill experts")
        router = self.manifest["router"]
        cortex.router_threshold = float(router["threshold"])
        cortex.auto_route = False
        address_revision = self.manifest.get("address_encoder_revision")
        if address_revision:
            state = torch.load(
                self.address_path / address_revision, map_location="cpu", weights_only=True)
            cortex.query_proj.load_state_dict(state)
        cortex.restore_structure([
            {
                "expert_id": entry["expert_id"],
                "status": entry["status"],
                "parent_id": entry.get("parent_id"),
                "created_step": entry.get("created_step", 0),
                "name": entry["name"],
                "synopsis": entry["synopsis"],
                "confidence": entry.get("confidence", 0.0),
                "usage": entry.get("usage", 0),
                "last_score": entry.get("last_score", 0.0),
            }
            for entry in self.manifest["experts"]
        ])
        for entry in self.manifest["experts"]:
            revision = entry["revisions"][entry["current_revision"]]
            state = torch.load(
                self.experts_path / revision["file"], map_location="cpu", weights_only=True)
            cortex.expert(entry["expert_id"]).load_state_dict(state)
        cortex.to(next(self.model.parameters()).device)

    def save_address_encoder(self) -> str:
        revision = f"address-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.pt"
        persist.atomic_torch_save(
            {name: tensor.detach().cpu() for name, tensor in
             self.model.skill_cortex.query_proj.state_dict().items()},
            self.address_path / revision,
        )
        self.manifest["address_encoder_revision"] = revision
        self._save_manifest()
        self.audit("address_encoder_saved", revision=revision)
        return revision

    def append_expert(self, expert_id: int) -> dict:
        cortex = self.model.skill_cortex
        expert = cortex.expert(expert_id)
        if expert_id != len(self.manifest["experts"]):
            raise ValueError("durable skill experts must be appended in order")
        entry = {
            **expert.metadata(),
            "revisions": [],
            "current_revision": 0,
        }
        self.manifest["experts"].append(entry)
        self._write_revision(entry, expert)
        self._save_manifest()
        self.audit(
            "skill_appended", expert_id=expert_id, status=expert.status,
            name=expert.name, synopsis=expert.synopsis)
        return entry

    def _write_revision(self, entry: dict, expert) -> dict:
        revision_index = len(entry["revisions"])
        filename = f"skill-{expert.expert_id:06d}-r{revision_index:04d}.pt"
        persist.atomic_torch_save(expert_snapshot(expert), self.experts_path / filename)
        revision = {
            "file": filename,
            "status": expert.status,
            "confidence": float(expert.confidence),
            "created_at": time.time(),
        }
        entry["revisions"].append(revision)
        entry["current_revision"] = revision_index
        entry.update(expert.metadata())
        return revision

    def save_revision(self, expert_id: int, operation: str) -> dict:
        entry = self.manifest["experts"][expert_id]
        revision = self._write_revision(entry, self.model.skill_cortex.expert(expert_id))
        self._save_manifest()
        self.audit(operation, expert_id=expert_id, revision=entry["current_revision"])
        return revision

    def rollback(self, expert_id: int) -> dict:
        entry = self.manifest["experts"][expert_id]
        current = int(entry["current_revision"])
        if current <= 0:
            raise ValueError("skill has no earlier durable revision")
        previous = entry["revisions"][current - 1]
        expert = self.model.skill_cortex.expert(expert_id)
        state = torch.load(self.experts_path / previous["file"], map_location="cpu", weights_only=True)
        expert.load_state_dict({name: tensor.to(expert.down.device) for name, tensor in state.items()})
        expert.status = previous["status"]
        expert.confidence = float(previous["confidence"])
        entry["current_revision"] = current - 1
        entry.update(expert.metadata())
        self._save_manifest()
        self.audit("rollback", expert_id=expert_id, revision=current - 1)
        return entry


class NaturalRuntimeSession:
    """Public operations for chat, proposal, teaching, rating, lifecycle, and restart checks."""

    def __init__(self, model, tokenizer, bank: SkillBank, device,
                 states=None, state_path: str | None = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.bank = bank
        self.device = device
        self.states = states or model.init_states(1, device)
        self.state_path = state_path
        self.active_skill: int | None = None
        self.confirmed_activations: set[int] = set()
        self.pending: dict | None = None
        self.last_teaching: dict | None = None

    def _save_state(self) -> None:
        if self.state_path:
            persist.save_states(self.state_path, self.states)

    def chat(self, message: str, **generation) -> dict:
        transcript, self.states = agent.run_turn(
            self.model, self.tokenizer, self.states, message, self.device,
            skill_id=self.active_skill, **generation)
        self._save_state()
        return {
            "transcript": transcript,
            "active_skill": self.active_skill,
            "fast_weight_state_saved": bool(self.state_path),
        }

    def propose_skill(self, text: str) -> dict:
        ids = self.tokenizer.encode(f"{cf.USER}\n{text}").ids
        expert_id, confidence = self.model.route_skill_from_ids(
            torch.tensor([ids], dtype=torch.long, device=self.device))
        if expert_id is None:
            return {"expert_id": None, "confidence": confidence, "requires_confirmation": False}
        expert = self.model.skill_cortex.expert(expert_id)
        return {
            "expert_id": expert_id,
            "name": expert.name,
            "synopsis": expert.synopsis,
            "confidence": confidence,
            "requires_confirmation": expert_id not in self.confirmed_activations,
        }

    def suggest_skill(self, user: str, assistant: str) -> dict:
        """Ask the base model for an editable short name and one-sentence synopsis."""
        prompt = (
            f"{cf.TEACH}\nPropose a short skill name on the first line and a one-sentence "
            f"synopsis on the second line.\nDemonstration user: {user}\n"
            f"Demonstration assistant: {assistant}")
        raw = self._greedy_reply(prompt, None, max_new=64)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        name = " ".join((lines[0] if lines else "").split()[:6])
        synopsis = " ".join(lines[1:]) if len(lines) > 1 else ""
        return {
            "name": name,
            "synopsis": synopsis,
            "raw": raw,
            "requires_confirmation": True,
        }

    def activate(self, expert_id: int, *, confirmed: bool = False) -> dict:
        expert = self.model.skill_cortex.expert(expert_id)
        if expert.status not in ("juvenile", "mature"):
            raise ValueError(f"skill {expert_id} is not active")
        if expert_id not in self.confirmed_activations and not confirmed:
            return {"activated": False, "requires_confirmation": True, "expert_id": expert_id}
        self.confirmed_activations.add(expert_id)
        self.active_skill = expert_id
        self.bank.audit("activate", expert_id=expert_id, confirmed=confirmed)
        return {"activated": True, "requires_confirmation": False, "expert_id": expert_id}

    def _training_batch(self, user: str, assistant: str):
        pieces = cf.render_pieces([("user", user), ("assistant", assistant)])
        ids, mask = [], []
        for text, trainable in pieces:
            encoded = self.tokenizer.encode(text, add_special_tokens=False).ids
            ids.extend(encoded)
            mask.extend([1.0 if trainable else 0.0] * len(encoded))
        if len(ids) < 2 or not any(mask[1:]):
            raise ValueError("demonstration did not produce an assistant loss span")
        sequence = torch.tensor([ids], dtype=torch.long, device=self.device)
        weights = torch.tensor([mask[1:]], dtype=torch.float32, device=self.device)
        return sequence[:, :-1], sequence[:, 1:], weights

    @torch.no_grad()
    def _greedy_reply(self, prompt: str, expert_id: int | None, max_new: int = 80) -> str:
        prime = self.tokenizer.encode(f"{cf.USER}\n{prompt}\n{cf.ASSISTANT}\n").ids
        states = self.model.init_states(1, self.device)
        cortex = self.model.skill_cortex
        context = cortex.force(expert_id) if expert_id is not None else cortex.suspend()
        generated = []
        with context:
            logits, states = self.model.forward_stream(
                torch.tensor([prime], device=self.device), states)
            for _ in range(max_new):
                token = int(logits[:, -1].argmax(dim=-1))
                text = self.tokenizer.decode(generated + [token])
                if any(marker in text for marker in (cf.END, cf.USER, cf.SYSTEM, cf.TOOL_RESULT)):
                    break
                generated.append(token)
                logits, states = self.model.forward_stream(
                    torch.tensor([[token]], device=self.device), states)
        return self.tokenizer.decode(generated).strip()

    def teach(self, name: str, synopsis: str, demonstrations: list[dict], *,
              confirmed: bool, anchors: list[str | dict] | None = None,
              steps: int = 64, lr: float = 1e-2) -> dict:
        if not confirmed:
            raise ValueError("the skill name and synopsis must be confirmed before teaching")
        if self.pending is not None:
            raise ValueError("rate the current candidate before teaching another skill")
        if not 1 <= len(demonstrations) <= 3:
            raise ValueError("a skill requires one to three confirmed demonstrations")
        address_ids = self.tokenizer.encode(f"{cf.SKILL}\n{name}\n{synopsis}").ids
        features = self.model.tok_emb(
            torch.tensor([address_ids], dtype=torch.long, device=self.device))
        expert_id = self.model.skill_cortex.birth(
            task_features=features, name=name.strip(), synopsis=synopsis.strip(),
            created_step=len(self.bank.manifest["experts"]))
        expert = self.model.skill_cortex.expert(expert_id)
        rollback = expert_snapshot(expert)
        batches = [
            self._training_batch(str(row["user"]), str(row["assistant"]))
            for row in demonstrations
        ]
        anchor_batches = []
        for anchor in (anchors or []):
            if isinstance(anchor, dict):
                rendered = cf.render([
                    ("user", anchor.get("user", "")),
                    ("assistant", anchor.get("assistant", "")),
                ])
            else:
                rendered = f"{cf.USER}\n{anchor}\n{cf.ASSISTANT}\n"
            anchor_batches.append(torch.tensor(
                [self.tokenizer.encode(rendered).ids],
                dtype=torch.long, device=self.device))
        before = self._greedy_reply(str(demonstrations[0]["user"]), expert_id)
        result = teach_expert(
            self.model, expert_id, batches, steps=min(max(steps, 16), 64),
            min_steps=16, patience=8, lr=lr, anchor_batches=anchor_batches,
            anchor_weight=1.0)
        after = self._greedy_reply(str(demonstrations[0]["user"]), expert_id)
        self.pending = {
            "expert_id": expert_id,
            "rollback": rollback,
            "demonstrations": demonstrations,
            "result": result,
        }
        self.last_teaching = {
            "expert_id": expert_id,
            "name": expert.name,
            "synopsis": expert.synopsis,
            "before": before,
            "after": after,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "anchor_kl": result.anchor_kl,
            "update_norm": result.update_norm,
            "steps": result.steps,
            "gradient_norm": float(expert.up.grad.norm()) if expert.up.grad is not None else 0.0,
            "routing_confidence": float(torch.dot(
                self.model.skill_cortex.address(features)[0],
                torch.nn.functional.normalize(expert.key, dim=0)).detach()),
        }
        self.bank.audit(
            "teach_candidate", expert_id=expert_id, name=expert.name, synopsis=expert.synopsis,
            demonstrations=demonstrations, result=self.last_teaching)
        return self.last_teaching

    def rate(self, rating: int) -> dict:
        if self.pending is None:
            raise ValueError("there is no pending candidate")
        if rating not in (1, 2, 3, 4, 5):
            raise ValueError("rating must be an integer from 1 to 5")
        expert_id = self.pending["expert_id"]
        expert = self.model.skill_cortex.expert(expert_id)
        if rating >= 4:
            self.model.skill_cortex.commit(
                expert_id, confidence=rating / 5.0, mature=rating == 5)
            self.bank.append_expert(expert_id)
            action = "committed"
        elif rating <= 2:
            restore_expert(expert, self.pending["rollback"])
            self.model.skill_cortex.quarantine(expert_id)
            self.bank.append_expert(expert_id)
            action = "quarantined"
        else:
            self.model.skill_cortex.discard_candidate(expert_id)
            action = "discarded"
        self.bank.audit("rate", expert_id=expert_id, rating=rating, action=action)
        self.pending = None
        return {"expert_id": expert_id, "rating": rating, "action": action}

    def quarantine(self, expert_id: int) -> dict:
        self.model.skill_cortex.quarantine(expert_id)
        if self.active_skill == expert_id:
            self.active_skill = None
        self.bank.save_revision(expert_id, "quarantine")
        return {"expert_id": expert_id, "status": "quarantined"}

    def rollback(self, expert_id: int) -> dict:
        entry = self.bank.rollback(expert_id)
        return {"expert_id": expert_id, "revision": entry["current_revision"],
                "status": entry["status"]}

    def calibrate_addresses(self, examples: list[dict], *, steps: int = 200,
                            lr: float = 1e-3) -> dict:
        """Calibrate only the factorized address encoder against confirmed expert keys."""
        if len(examples) < 2:
            raise ValueError("address calibration requires at least two labeled descriptions")
        cortex = self.model.skill_cortex
        active = [expert for expert in cortex.experts if expert.status in ("juvenile", "mature")]
        positions = {expert.expert_id: index for index, expert in enumerate(active)}
        if len(active) < 2:
            raise ValueError("address calibration requires at least two active skills")
        rows = []
        for example in examples:
            expert_id = int(example["expert_id"])
            if expert_id not in positions:
                raise ValueError(f"calibration label is not an active skill: {expert_id}")
            ids = self.tokenizer.encode(str(example["text"])).ids
            rows.append((torch.tensor([ids], device=self.device), positions[expert_id]))
        required = [(parameter, parameter.requires_grad) for parameter in self.model.parameters()]
        trainable = {id(parameter) for parameter in cortex.query_proj.parameters()}
        optimizer = torch.optim.AdamW(cortex.query_proj.parameters(), lr=lr, weight_decay=0.0)
        keys = torch.stack([
            torch.nn.functional.normalize(expert.key.detach(), dim=0) for expert in active])
        initial = final = 0.0
        try:
            for parameter, _old in required:
                parameter.requires_grad_(id(parameter) in trainable)
            for step in range(min(max(int(steps), 1), 2000)):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for ids, target in rows:
                    features = self.model.tok_emb(ids).detach()
                    query = cortex.address(features)
                    logits = query @ keys.T / 0.07
                    losses.append(torch.nn.functional.cross_entropy(
                        logits, torch.tensor([target], device=self.device)))
                loss = torch.stack(losses).mean()
                if step == 0:
                    initial = float(loss.detach())
                loss.backward()
                optimizer.step()
                final = float(loss.detach())
        finally:
            for parameter, old in required:
                parameter.requires_grad_(old)
        revision = self.bank.save_address_encoder()
        result = {
            "examples": len(rows),
            "steps": min(max(int(steps), 1), 2000),
            "initial_loss": initial,
            "final_loss": final,
            "revision": revision,
        }
        self.bank.audit("address_calibration", result=result)
        return result

    def restart_verification(self, base_checkpoint: str, state_path: str | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            runtime_state = Path(state_path or (Path(directory) / "state.pt"))
            persist.save_states(str(runtime_state), self.states)
            reloaded_model = persist.load_model(base_checkpoint, self.device).eval()
            reloaded_bank = SkillBank(self.bank.root, reloaded_model, base_checkpoint)
            reloaded_states = persist.load_states(str(runtime_state), self.device)
            experts_equal = all(
                torch.equal(left, right)
                for old, new in zip(
                    self.model.skill_cortex.experts, reloaded_model.skill_cortex.experts)
                for left, right in zip(old.state_dict().values(), new.state_dict().values())
            )
            states_equal = all(
                torch.equal(left_w, right_w)
                for left_state, right_state in zip(self.states, reloaded_states)
                for left_w, right_w in zip(left_state.W, right_state.W)
            )
            result = {
                "complete": experts_equal and states_equal,
                "experts_equal": experts_equal,
                "fast_weights_equal": states_equal,
                "experts": len(reloaded_bank.manifest["experts"]),
            }
            self.bank.audit("restart_verification", result=result)
            return result

    def snapshot(self) -> dict:
        return {
            "active_skill": self.active_skill,
            "pending_candidate": None if self.pending is None else self.pending["expert_id"],
            "last_teaching": self.last_teaching,
            "cortex": self.model.skill_cortex.snapshot(),
        }


def load_runtime(checkpoint: str, tokenizer_path: str, bank_dir: str, state_path: str,
                 device=None) -> NaturalRuntimeSession:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = persist.load_model(checkpoint, device).eval()
    tokenizer = tk.load(tokenizer_path)
    tk.assert_atomic_special_tokens(tokenizer)
    if tokenizer.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError("runtime tokenizer/model vocabulary mismatch")
    bank = SkillBank(bank_dir, model, checkpoint)
    states = (
        persist.load_states(state_path, device)
        if os.path.exists(state_path)
        else model.init_states(1, device)
    )
    return NaturalRuntimeSession(model, tokenizer, bank, device, states, state_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--skill-bank", required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    runtime = load_runtime(args.checkpoint, args.tokenizer, args.skill_bank, args.state)
    print("Natural Cortex runtime ready. Empty input exits.")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            break
        proposal = runtime.propose_skill(text)
        if proposal["expert_id"] is not None:
            print(f"skill proposal> {proposal['name']}: {proposal['synopsis']} "
                  f"({proposal['confidence']:.3f}; activation requires confirmation)")
        result = runtime.chat(text)
        for role, content in result["transcript"]:
            if role == "assistant":
                print(f"model> {content}")


if __name__ == "__main__":
    main()
