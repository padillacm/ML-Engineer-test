# Spec Delta

## ADDED Requirements

### Requirement: Containers run with least privilege
The service and vector-store containers SHALL run with a read-only root filesystem (writable scratch on tmpfs and the data volume only), every Linux capability dropped, no privilege escalation, a process-count limit and a memory limit, and the service SHALL work fully under them.

#### Scenario: Hardened stack
- **WHEN** the stack runs with the defaults
- **THEN** the smoke suite passes, the service container reports a read-only root, no capabilities and `no-new-privileges`, and a write outside tmpfs and the volumes fails

### Requirement: The detector model is built in with a secret
The image SHALL include the detector model at a pinned revision with sha256-checked files and its licence notice, downloaded at build time with a build secret that never enters a layer; a build without the secret or with the model opted out SHALL succeed and produce an image whose service runs with injection detection off.

#### Scenario: Build with the token
- **WHEN** the image is built with the Hugging Face token secret
- **THEN** the model files are present read-only under the models directory, their hashes match, the NOTICE carries the licence attribution, and no layer or environment variable contains the token

#### Scenario: Build without the token
- **WHEN** the image is built with the model opted out
- **THEN** the build succeeds and the service starts with injection detection off

### Requirement: Dependencies are audited before a push
The known-vulnerability audit of the locked dependencies SHALL run as part of the pre-push checks, failing the push on a known vulnerability without an accepted exception.

#### Scenario: Audit on push
- **WHEN** the pre-push hooks run
- **THEN** the audit runs against the lock file and its result decides the push
