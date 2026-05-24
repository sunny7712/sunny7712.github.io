Sometime in early 2025 we noticed the algo-trading platform's Redis cluster was being asked to do something Redis was never built for: range scans across hundreds of gigabytes of historical candle data. Backtests would issue a query for, say, every minute of NIFTY 50 between 2018 and 2024, and the cache would dutifully assemble the answer at the cost of holding most of its working set in RAM.

It worked. It worked the way a hammer works for screws — loud, expensive, and the screws are bent afterward. We were paying for memory we didn't really need and pretending a key-value store was an analytics database.

## The shape of the workload

What backtests actually want is depressingly simple. Given a symbol and a date range, return the candles. Maybe at a different resolution. Maybe enriched with corporate actions. Everything is read-only, everything is range-scoped, and the dataset grows by exactly one trading day per trading day.

Once you say it out loud it sounds like the columnar-Parquet-and-OLAP brochure, which is what we ended up building.

## What we built

The service is an embedded DuckDB instance running inside an ordinary Java API pod. The data lives in partitioned Parquet on GCS, mounted into the container via NFS. DuckDB does the heavy lifting; the Java layer handles auth, request shaping, and translating into SQL.

```java
// roughly
String sql = "SELECT timestamp, open, high, low, close, volume "
          + "FROM read_parquet('/mnt/candles/symbol=' || ? || '/year=*/month=*/data.parquet') "
          + "WHERE timestamp BETWEEN ? AND ? "
          + "ORDER BY timestamp";
return duckdb.query(sql, symbol, from, to);
```

Each pod has its own DuckDB process. No coordination, no cluster, no leader election, no node-down panic at 2 a.m. Scaling out is the same as scaling out any other stateless API: add pods.

## What we measured

On a typical workload — 500 GB+ of candle data, queries spanning a few years — we landed at sub-second p95 latency. Throughput scales linearly with the number of API pods, which is what you'd hope for from an embedded engine on a shared filesystem.

More important than the latency numbers: the operational footprint collapsed. The team page for the Redis cluster has been quiet for months.

## What I'd say to past me

If your cache is the database, just admit it and use a database. Cleanly separating the hot transactional path (Redis, fine) from the historical analytical path (DuckDB + Parquet, very fine) made the system both faster and cheaper. The hard part was the cultural shift — convincing a team that "just a Java service" could do analytics work.
