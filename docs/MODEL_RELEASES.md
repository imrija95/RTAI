# Model release policy

No model is currently published. Code publication and model publication are separate decisions.

## Release format

Public models use a pinned Hugging Face repository revision containing:

- `model.safetensors` with code-free tensor weights;
- `rtai_config.json` with architecture, parameter count, checksums, and schema version;
- the exact tokenizer when the model uses one;
- a complete model card covering data, revision, evaluation, intended use, limitations, hardware,
  and a separately selected model license.

Create an offline release draft with:

```bash
uv sync --extra hub
uv run --extra hub python -m rtai.model_hub export \
  --checkpoint MODEL.pt --tokenizer TOKENIZER.json \
  --out dist/MODEL --name MODEL --license LICENSE_ID
```

The command does not upload. Downloading a future published bundle requires an immutable revision:

```bash
uv run --extra hub python -m rtai.model_hub download \
  --repo-id ACCOUNT/MODEL --revision COMMIT_SHA --out models/MODEL
```

## Mandatory release gate

1. Start from a clean training checkpoint; reject runtime memory, chat state, optimizer resume,
   telemetry, and search state.
2. Re-run the declared evaluation and process-restart memory check from a clean environment.
3. Record dataset identifiers, licenses, immutable revisions, tokenizer checksum, seed, source
   revision, and hardware class.
4. Scan every release file for credentials, personal paths, and embedded user data.
5. Verify the Safetensors round trip and every manifest checksum.
6. Review the model license independently from the source-code license.
7. Upload only after an explicit maintainer approval; never upload automatically from training.
