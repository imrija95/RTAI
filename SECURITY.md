# Security policy

## Supported version

Security fixes target the current `main` branch. This is experimental research software; no
production-security guarantee is made.

## Reporting a vulnerability

Do not open a public issue for an unpatched vulnerability or leaked credential. Use GitHub's
private vulnerability reporting for the repository. Include the affected revision, a minimal
reproduction, impact, and any suggested mitigation.

## Model and state files

- Project loaders accept tensor-only checkpoints and do not permit arbitrary pickle code.
- Hugging Face releases use Safetensors and a checksum-verified JSON manifest.
- Runtime memory/state files can encode information observed during operation. They are local,
  owner-readable artifacts and must never be attached to issues or model releases without an
  explicit privacy review.
- The local dashboards bind to `127.0.0.1`. Exposing them through a proxy is outside the supported
  threat model.

## Secrets

Credentials belong in environment variables or an external secret manager, never in repository
files. The public audit script and CI catch common credential signatures, but scanners are not a
substitute for credential rotation after an accidental disclosure.
