There is a specific kind of engineer who, upon being handed a 100 GB analytics workload, reaches first for a Snowflake account or a Spark cluster. I sympathize — those are extraordinary tools — but they are also the wrong tools, most of the time, for the size of problem most of us actually have.

An embedded OLAP engine (DuckDB, Polars, chDB) running in the same process as your API can do startling things. The full table-scan-and-aggregate machinery, ready to go, no coordination overhead, deployed exactly the same way you deploy any other library.

## When it stops working

When the data outgrows a single machine's storage budget, or when you need true concurrency over writes, or when you want a SQL-accessible warehouse for ad-hoc analysts. Those are real reasons. "It feels too simple" is not one of them.
