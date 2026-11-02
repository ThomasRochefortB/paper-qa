# Security of this fork

This repository is ThomasRochefortB's fork of PaperQA, not the upstream
FutureHouse release or authority to publish its PyPI package. The inherited
publication workflow is restricted to the upstream repository and cannot publish
from this fork.

Use the reviewed default branch and its `uv.lock` when reproducing this fork.
Older tags and the previous lockfile have not been patched retroactively. An
upgrade here does not replace already installed environments or upstream PyPI
artifacts.

## Dependency updates

The dependency-security workflow audits an export of every locked group and extra
without installing those dependencies. It runs on pull requests, main pushes and
a weekly schedule, with no advisory ignores. The Linux audit is not a complete
SBOM for other platforms, downloaded model weights or remote services.

LiteLLM is pinned to the patched `1.100.4` backport. Later lines require newer
Pydantic than `fh-llm-client` permits, or expose type definitions incompatible
with the client's Pydantic 2.10 requirement. The current upstream LiteLLM line also
caps FileLock below the patched version needed here. Upgrade this pin only after
the strict audit and offline client/router compatibility tests pass. Never use
the compromised LiteLLM `1.107.0` release.

PaperQA uses LiteLLM through its client and router APIs; this review found no
LiteLLM Proxy server entrypoint. An affected dependency version does not establish
that a proxy-specific vulnerability is reachable. If you separately deploy the
proxy, review its configuration and advisories independently.

The focused compatibility job uses fake provider responses and no credentials.
The inherited provider tests replay cassettes with recording disabled and
use dummy keys instead of stored provider secrets. Local model/dataset tests may require separate model artifacts and are not
covered by the lightweight compatibility job.

## Reporting and repository controls

Do not put credentials or sensitive exploit details into public issues. Contact
the fork maintainer privately; vulnerabilities in upstream PaperQA should also be
reported through the upstream project's security process. Repository settings
(secret scanning, push protection, dependency alerts and branch protection) need
administrator configuration separately from these source changes.
