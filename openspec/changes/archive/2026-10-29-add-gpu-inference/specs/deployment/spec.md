# Spec Delta

## ADDED Requirements

### Requirement: The CUDA image is a build option
The image SHALL be built with the CPU torch build by default and with the CUDA build when a build argument selects it; both SHALL contain the same service code and models, and the CPU image SHALL NOT contain CUDA libraries.

#### Scenario: Default build
- **WHEN** the image is built without the argument
- **THEN** torch reports no CUDA support and no NVIDIA library is installed

#### Scenario: CUDA build
- **WHEN** the image is built with the CUDA variant and run with a GPU
- **THEN** torch sees the GPU and the service starts on it

### Requirement: The service shares the GPU under the gpu profile
With the `gpu` profile the service SHALL be built with the CUDA variant and SHALL reserve the GPU next to the language model server, and the whole stack's GPU memory SHALL stay within the device with the measured margin recorded.

#### Scenario: Stack on the GPU
- **WHEN** the stack starts with `COMPOSE_PROFILES=gpu`
- **THEN** readiness passes, the service reports the GPU as its device, and the smoke suite passes
