A rate limiter that's subtly wrong is worse than one that doesn't exist. If there's no limiter, everyone knows to be careful. If there's a limiter that quietly enforces the wrong number, nobody finds out until a customer opens a ticket asking why they're being throttled at 30 requests a second when the plan says 50. I spent a few months building [Guardian](/#/projects/guardian), a Redis-backed rate-limiting library for Spring Boot, mostly to find out how many ways that could happen. This isn't a tour of the library — the README covers the annotations and the algorithms. This is the list of places it almost lied to me, and what it cost to make it stop.

## What makes this hard

Counting requests is not the hard part. The hard part is doing the count-and-decide atomically, under concurrency, across every instance of your app, against a store that can fail mid-request. Three failure classes, really: races between concurrent requests, arithmetic that's wrong in a way that doesn't crash, and a dependency (Redis) that goes down while your API doesn't. Everything below is one of those three.

## Annotate it and forget it — mostly

The annotation is the whole interface: `@GuardianRateLimit(key = "#userId", plan = "pro_plan", quota = "read_limit")` on a method, and the calling code never mentions Guardian again. No manual `if (!limiter.allow(...))` scattered through every controller, no coupling between "is this endpoint rate limited" and "how is rate limiting done." A `RateLimitAspect` intercepts anything carrying the annotation, resolves `key`/`plan`/`quota` via SpEL against the method's actual arguments (`#user.tier`, `#request.getRemoteAddr()`, whatever's in scope), picks the target `RateLimiter` bean by name, and either lets the call through or throws before it ever executes. Add the annotation, remove it, change the plan — none of it touches the method body.

The mechanism that makes this non-intrusive is the same one that gives it away for free: this is proxy-based Spring AOP, not compile-time weaving, so interception only happens when a call arrives through the Spring proxy. Call an annotated method on `this` from another method in the same bean and you've bypassed the proxy entirely — the annotation silently becomes a no-op. It's a well-known Spring AOP limitation in general, not specific to Guardian, but it's exactly the kind of thing that's invisible until someone refactors two endpoints into "one method calling the other" and one of them quietly stops being rate limited.

Extensibility runs on the same annotation. `algorithm` is just a Spring bean name, resolved against a `Map<String, RateLimiter>` that Spring populates automatically from every bean implementing the interface — register `@Component("myCustomLimiter")` implementing `RateLimiter`, point `@GuardianRateLimit(algorithm = "myCustomLimiter")` at it, and it's live. No core-library change, no registry to update by hand; `RateLimitAspectTest` proves it end to end — a second mock bean, wired in under a different name, and the aspect routes to it without knowing it exists at compile time.

That flexibility is also exactly where the cost shows up. The routing is a string lookup, so a typo in `algorithm` isn't caught by the compiler — it's caught by an `IllegalStateException` the first time that endpoint gets hit in production. And the interface only buys you the decision logic; `TokenBucketRateLimiter` and `SlidingWindowCounterRateLimiter` each independently reimplement the same fail-open/fail-closed handling and the same Micrometer timer boilerplate around their `allow()` calls — extending Guardian with a new algorithm means writing that wrapper a third time, not inheriting it.

## Atomicity: why check-and-decrement can't be two steps

The obvious way to rate limit is: read the counter, check it against the limit, write the counter back. That's two round trips with a gap in between, and the gap is where the bug lives. Two requests read "4 out of 5 used" at the same instant, both decide they're under the limit, both write "5 used." Your limit of 5 just became 6, and it gets worse with more concurrency, not better.

The instinct if you've only ever worked on one machine is to reach for a lock. That doesn't help here — a `synchronized` block guards one JVM's heap; it has no idea another instance of your app, on another pod, is doing the same read-check-write against the same Redis key right now. The lock needs to live somewhere all instances can see it, which means it needs to live in Redis, which means you're back to the same round-trip problem, just with extra steps.

Guardian's answer is to not do it in Java at all. The whole decision — read state, compute refill, check the limit, write state back — is one Lua script, executed by `EVAL`. Redis runs Lua scripts single-threaded and start-to-finish; nothing else touches that key while the script is running. The atomicity isn't a lock you're managing, it's a property of how Redis executes scripts.

```lua
-- token_bucket_hash.lua, abbreviated
local current_tokens = tonumber(fields[1])
local tokens_to_add = time_elapsed * refill_rate
current_tokens = math.min(bucket_capacity, current_tokens + tokens_to_add)
if current_tokens >= cost then
    allowed = 1
    current_tokens = current_tokens - cost
end
redis.call("HSET", key, F_TOKENS, current_tokens, F_LAST_REFILL, now)
```

The proof is boring, which is the point: a JUnit test fires 50 threads at one key with a bucket capacity of 10, and exactly 10 get through, every run. Under real load it holds too — I pointed a k6 scenario at a single sliding-window key with 10,000 concurrent virtual users; 8,191 of them landed inside the test's 5-second window, and exactly 5 were allowed through, the configured limit, no more.

Guardian also ships a second implementation of the same guarantee, built specifically to compare against the Lua path: `RedisTransactionTokenBucketStore` uses `WATCH`/`MULTI`/`EXEC` instead — optimistic locking, retry-on-conflict, rather than single-threaded execution. Same atomicity guarantee, a completely different mechanism to get there. It's not exposed as a real choice for integrators — Lua is faster and there's less to reason about, one round trip instead of a watch-then-maybe-retry loop — but building the optimistic version was the only way to actually know that, instead of assuming it.

## The bug in the arithmetic that nobody would have noticed

This is the one worth slowing down for, because it doesn't look like a bug at all — it looks like a rate limiter doing its job, just slightly too well.

Token bucket refill is `tokens_to_add = elapsed_time * refill_rate`. Say your refill rate is 50 tokens/second and requests arrive every 15ms, which is a completely normal request pattern. If you store tokens as an integer — a defensible choice, nobody wants floating-point drift in a value that gates whether a request is allowed — then `15ms * 50/sec` works out to well under one whole token, and an integer truncates that to zero. Every refill on a sub-second cadence rounds down to nothing. Over a 900ms burst of 60 requests, the bucket should refill 45 tokens (0.9s × 50/sec). Store it as an integer and it refills 0, because each individual step was too small to survive truncation, and the errors don't average out — they compound downward, every single time.

The system doesn't throw, doesn't log, doesn't 500. It just enforces a limit meaningfully stricter than the number in your YAML, forever, silently. That's a worse bug than a crash, because a crash pages someone. This one shows up as a slow, unexplained rise in support tickets from your highest-frequency, best-paying customers — the ones whose traffic pattern is exactly the sub-second bursts that trigger it.

The fix is to stop rounding at the token level. Guardian scales everything — bucket capacity, cost per request — by 1000 before doing the arithmetic, so a "token" internally is a milli-token, and the same math that used to truncate to zero now truncates to the nearest thousandth of a token instead:

```java
long bucketCapacityUnits = quota.getBucketCapacity() * TOKEN_RESOLUTION_MULTIPLIER; // ×1000
long costTokensUnits = costTokens * TOKEN_RESOLUTION_MULTIPLIER;
// refill rate is left unscaled on purpose — elapsed time is already
// tracked in milliseconds, so the /1000 falls out of the unit conversion for free
```

That last line is the part I'd flag in review if I saw it cold: the refill rate looks unscaled, like someone forgot a multiplier. It isn't forgotten — elapsed time being in milliseconds already supplies the implicit division by 1000, so scaling the rate too would double-correct. It's the kind of line that's correct and looks wrong, which is exactly the kind of line that gets "fixed" by someone who didn't do the unit algebra first. It has a comment now.

## The same rounding bug, mirrored, in the other algorithm

Sliding Window Counter has its own place where truncation matters: `estimated_count = math.floor((previous_count * previous_window_weight) + current_count)`. The weighted estimate — the previous window's count scaled down by how much of it still "counts," plus the current window's raw count — gets floored before it's compared against the limit. Flooring pushes the estimate down, and the admission check is `estimated_count + cost > limit`: a lower estimate trips that check less often, so the algorithm ends up slightly more permissive than the true smooth interpolation, not less.

It's the same species of bug as the token-bucket one — an integer operation quietly moving the enforced number away from the configured one — just biased in the opposite direction. Token bucket rounds toward too strict. Sliding window rounds toward too lenient. Small in practice at reasonable request volumes, but the interesting part isn't the magnitude — it's that neither rounding direction was a deliberate choice. Both just fell out of reaching for `math.floor`/integer math without first asking which way truncation should break.

## Fail-open or fail-closed is not my call to make

When Redis is unreachable, Guardian has to decide what a request means in the absence of any state to check it against. There are exactly two honest answers — let it through (fail-open) or block it (fail-closed) — and the correct one depends entirely on what's behind the limiter, which I don't know and can't know from inside the library. A public read endpoint wants fail-open: a Redis blip shouldn't take down a page nobody's abusing. A billing or write endpoint wants fail-closed: better to serve a 503 for thirty seconds than let unmetered traffic through while nobody's counting. So it's a config flag, `failure-mode: open | closed`, decided per integration, not baked into the library. Every fallback trip also increments a `guardian.ratelimit.fallback` counter, tagged by mode — an outage that silently degrades your rate limiting is worse than one that pages someone.

Neither choice is free, and the honest version of this section says so: fail-open means anyone who can knock over your Redis has just bypassed every rate limit in the system for as long as the outage lasts. Fail-closed means a transient Redis blip becomes a full outage on every protected endpoint. I tested the logic with mocked failures — `TokenBucketRateLimiterResiliencyTest` throws inside the store and asserts each mode does what it says — but I never actually killed Redis under live load and watched the fallback path take over in practice. That's a real gap, not a rhetorical one; "the unit test passed" and "I watched it work" are different claims and I only get to make the first one.

## Changing a limit without a redeploy

The other operational reality: rate limits change during incidents, and "edit the YAML, cut a release, wait for the rollout" is not a fast enough loop when someone's actively hammering an endpoint. `GuardianConfigScheduler` polls a Redis hash (`guardian:config:*`) on an interval, and on success does an atomic reference swap — `AtomicReference::set` — so readers never see a half-updated config. If the poll fails or the payload doesn't parse, the reference is simply never touched, so the last good config keeps serving without any special-case "rollback" logic; it falls out of only writing on success.

The tradeoff is staleness: a 60-second poll interval means a limit you change now takes up to a minute to actually apply, which is a real cost during an active incident. I picked polling over something event-driven like Redis pub/sub because polling degrades to "config is briefly stale," which is recoverable and boring, whereas a missed pub/sub message degrades to "config never updates until someone notices and pokes it," which is a worse failure to debug at 3 a.m. Simple and occasionally slow beat clever and occasionally silent.

## How I know any of this is true

Everything above is falsifiable, so I built a way to falsify it: Testcontainers-backed integration tests for the Lua scripts against real Redis, and a k6 scenario per claim rather than one generic load test — atomicity gets a thundering-herd scenario, isolation gets a noisy-neighbor scenario, throughput gets a ramping baseline, all running against the same `docker-compose` stack with Prometheus and Grafana wired in so the numbers are pulled from real metrics, not printed by the test runner.

The headline numbers: Guardian's own decision logic (the Redis round trip plus Lua execution, timed via Micrometer, with the HTTP and Spring layers excluded) runs at p95 0.98ms and averages 0.67ms. Under a noisy-neighbor scenario — one key hammering at 3,000 req/s while another sends a legitimate 5 req/s — the legitimate key got exactly the 150 requests it should have over 30 seconds, at p95 1.28ms, with zero measurable bleed-through from the attacker's load. A short soak (four minutes, 800 req/s, high-cardinality keys) showed heap flat within noise and zero full GCs. Full numbers, including the throughput ceiling and the soak caveats, are in the [README](https://github.com/sunny7712/guardian#load-testing-k6).

## What I'd do differently

A few things I only found because I went looking for this post, which is itself the finding: none of these were on a dashboard, because nothing was watching for them.

The token bucket's Redis key gets a flat `EXPIRE 3600` regardless of the plan's actual refill rate — a slow trickle-refill plan and a fast-burst plan get the same one-hour TTL, when the TTL should really be derived from how long the bucket takes to matter again. Worse, and this one I only caught while writing this up: the sliding-window Lua script sets its window key's TTL to `window_size_ms * 2` — but Redis `EXPIRE` takes seconds, not milliseconds. A 60-second window ends up with a TTL of 120,000 *seconds*, about 33 hours, instead of 120. It doesn't break correctness — old windows are never read again — but it means every distinct key you've ever rate-limited sits in memory for a day and a half. In a four-minute soak run with high-cardinality keys, that's the difference between a Redis instance holding a few thousand keys and one holding 277,000.

The k6 test matrix has a gap I never filled: `high_cardinality.js` exists as a filename in the `tests/` directory and nothing else — an empty file, a scenario I named and never wrote. And `thundering_herd.js`'s own assertions don't reconcile with its own load shape: it fires 10,000 virtual users at one key and asserts exactly 95 get blocked, but under real load only 8,191 of those 10,000 even complete inside the test's window — the other 1,809 never finish, saturated at the Tomcat thread pool before they reach Guardian at all. The atomicity claim still holds — exactly 5 get through regardless — but the blocked-count assertion was checked against a number that was never going to be true.

And earlier on, before any of the above existed, I built a consistent-hashing ring for spreading rate-limit state across multiple Redis nodes — sharding infrastructure for a scaling problem I didn't have yet. I deleted it a few weeks later once the Lua-atomicity approach made it obvious that one well-used Redis instance was the actual bottleneck I needed to solve, not node distribution. Building it wasn't wasted exactly — it's the fastest way I know to find out a problem isn't the one you thought it was — but it's the kind of thing worth naming instead of quietly deleting and pretending the design was linear from the start.

## The lesson

None of the interesting bugs here were in the parts of the system that are hard to write. The happy path — annotate a method, atomically check a counter, return 429 — is maybe a day of work and would pass every demo. Everything that took real time was in the parts that don't show up in a demo: what integer division does to a refill rate nobody stress-tested at sub-second intervals, what a millisecond value does when it's handed to a function that expects seconds, what happens to the request in front of you when the thing behind you goes down. That's most of what infrastructure work actually is, underneath the annotations and the YAML: finding the arithmetic that's wrong in a way that doesn't crash, and building something that tells you before your users do.
