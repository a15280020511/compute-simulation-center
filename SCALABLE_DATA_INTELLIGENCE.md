# Scalable Data Intelligence

`large_scale_data_intelligence` adds bounded extended analysis without changing
existing exact small-data modes.

Modes:

- `complexity_preflight`: estimates work and selects standard, extended or batch profile.
- `blocked_entity_collision`: up to 50,000 records with mandatory blocking and a hard candidate-pair budget.
- `sorted_event_collision`: up to 50,000 events using entity-partitioned sort-and-sweep.
- `chunked_numeric_profile`: up to 25,000 rows by 100 numeric columns using one-pass aggregates.
- `large_graph_summary`: up to 10,000 nodes and 100,000 edges using sparse components, degree and bounded PageRank.

All modes are offline, deterministic, output-bounded, and reject arbitrary code,
URLs and unbounded Cartesian work.  Large payloads still require the governance
material-package path; the public Issue receipt contains only summaries and
Artifact references.
