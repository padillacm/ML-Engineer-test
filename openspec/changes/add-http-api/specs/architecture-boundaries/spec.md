# Spec Delta

## ADDED Requirements

### Requirement: The HTTP layer depends on use cases only
Modules under `app/api` SHALL import only the web framework, the application use cases and ports, core types and errors, the HTTP schemas and the settings and observability modules; they SHALL NOT import adapters or vendor model, OCR, vector-store or LLM libraries. Adapters SHALL be built only in the composition root (`app/bootstrap.py`), which `app/main.py` calls.

#### Scenario: Router imports an adapter
- **WHEN** a module in `app/api` imports from `app.adapters`
- **THEN** the architecture guard fails naming the file and line

#### Scenario: Tests replace the core
- **WHEN** the application is created with a container of fake use cases
- **THEN** every endpoint works without any model, engine or service installed
