RestTemplate works, but every call site grows tentacles: timeouts, retries, deserialization, error mapping, the works. Eight services later, you have a hundred slightly-different ways to call a hundred slightly-different endpoints.

Feign collapses all of that into an interface plus an annotation. The discipline is enforced by the type system. The retry policy lives in one place. The deserialization is consistent. The diff was, frankly, embarrassing in our favor.
