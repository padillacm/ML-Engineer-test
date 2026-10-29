# Spec Delta

## ADDED Requirements

### Requirement: Search excludes flagged chunks
Each stored chunk SHALL carry a flagged mark in its payload (absent means not flagged, so existing collections keep working), and search SHALL never return a flagged chunk, whatever its similarity; counting and deleting a document's chunks SHALL include flagged ones.

#### Scenario: Flagged best match
- **WHEN** the vector closest to the query belongs to a flagged chunk
- **THEN** the search returns the closest unflagged chunks only

#### Scenario: Collection written before the mark existed
- **WHEN** a collection whose points have no flagged field is searched
- **THEN** every point is searchable, as before

#### Scenario: Chunk count includes flagged
- **WHEN** a document with 10 chunks, 2 of them flagged, is counted
- **THEN** the count is 10
