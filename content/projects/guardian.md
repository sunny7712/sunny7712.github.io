## The problem

Rate limiters look trivial until concurrency and floating-point arithmetic get involved. A limiter that's subtly wrong is worse than none — you don't find out until real users are being throttled below the number in the config.

## What I did

Built a distributed rate limiter for Spring Boot: Token Bucket and Sliding Window Counter, both pushed into Redis Lua scripts so the check-and-decrement is atomic across instances — no distributed locks, no races. Config lives in Redis and hot-reloads without a restart; a Redis outage degrades to a configurable fail-open or fail-closed per endpoint. Verified with concurrency tests (8k+ requests hammering one key, exactly the configured limit gets through) and k6 load tests against the running stack.

## What it taught me

The naive integer token-bucket math silently rounds every fractional refill down to zero on sub-second traffic, so the system throttles harder than the config says — no error, no crash, just quietly stricter than what you configured. Scaling to milli-token units before doing the arithmetic fixed it. That bug, and a Redis TTL unit mismatch I found while writing this up (milliseconds handed to a seconds-based `EXPIRE`), taught me the same lesson twice: the dangerous bugs in infra code don't crash, they just make the system a little wrong.
