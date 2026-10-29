# Spec Delta

## ADDED Requirements

### Requirement: The audit viewer can be turned off
`AUDIT_VIEWER_ENABLED` SHALL control whether the viewer page and its files are served; it SHALL default to enabled. When disabled, `/audit` and every file under it SHALL answer 404 and the API SHALL be unaffected.

#### Scenario: Disabled viewer
- **WHEN** the service starts with `AUDIT_VIEWER_ENABLED=false`
- **THEN** `GET /audit` answers 404 and `POST /api/extract` works as before
