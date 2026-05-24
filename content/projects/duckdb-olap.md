## The problem

Our Redis-backed historical data path was straining under range-scan queries from the algo-trading backtester. The cluster's memory bill grew faster than the value the cache delivered, and a stateful cluster meant late-night pages whenever a node misbehaved.

## What I did

Designed an embedded analytics layer using DuckDB inside our existing Java pods, reading partitioned Parquet files from GCS mounted via NFS. No coordination, no leader election — just a query engine running next to the API code. Wrote a thin SQL builder and ported the existing endpoints over.

## What it taught me

Cache-shaped problems and analytics-shaped problems look identical until you measure them. Separating the two unlocked a 10× reduction in operational complexity and let us right-size each path independently.
