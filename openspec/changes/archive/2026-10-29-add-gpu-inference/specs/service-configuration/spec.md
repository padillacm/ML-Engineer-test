# Spec Delta

## ADDED Requirements

### Requirement: Concurrency defaults follow the device
When the models run on the GPU, unset pool sizes SHALL be derived for the GPU (a measured small pool per model) rather than from the CPUs, and the CPU oversubscription warning and `TORCH_THREADS` SHALL apply only to the CPU path; set values SHALL still win, and the concurrency line at start-up SHALL name the device.

#### Scenario: GPU defaults
- **WHEN** the service starts on the GPU with no pool size set
- **THEN** the NER, embedder and detector pools take the GPU defaults and the start-up line reports them with the device

#### Scenario: CPU unchanged
- **WHEN** the service starts on the CPU
- **THEN** the derived sizes are those of the CPU rule, as before
