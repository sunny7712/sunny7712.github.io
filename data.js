// All portfolio content. Edit me to change what shows up across the site.
window.SITE = {
  identity: {
    name: "Vamsi Katta",
    short: "Vamsi K.",
    role: "Software Engineer",
    location: "Bengaluru, India",
    tagline: "Building infrastructure for the markets at Groww. Interested in databases, distributed systems, and the messy seams between them.",
    intro: "I'm a software engineer working on trading infrastructure. Most days I'm thinking about query latency, storage formats, and how to make systems easier to operate. Before that I was at IIIT Allahabad studying electronics, where I got sidetracked into ML and never quite went back.",
    email: "sunny77katta2002@gmail.com",
    socials: [
      { label: "GitHub", handle: "sunny7712", url: "https://github.com/sunny7712" },
      { label: "Twitter", handle: "vamsik77", url: "https://x.com/vamsik77" },
      { label: "LinkedIn", handle: "vamsi-k77", url: "https://linkedin.com/in/vamsi-k77" },
      { label: "Email", handle: "sunny77katta2002@gmail.com", url: "mailto:sunny77katta2002@gmail.com" }
    ]
  },

  experience: [
    {
      company: "Groww",
      role: "Software Development Engineer",
      period: "Jul 2025 — Present",
      location: "Bengaluru",
      blurb: "Core maintainer for the primary trading API services. Building the historical data layer powering algorithmic backtests.",
      bullets: [
        {
          title: "Historical Data API for Algo Trading",
          body: "Designed and shipped a dedicated OLAP service to replace Redis for long-range historical backtests. Embedded analytics on DuckDB, querying partitioned Parquet on GCS mounted via NFS on Kubernetes. Avoided the overhead of a distributed cluster while delivering sub-second latency on 500GB+ of candle data, with linear throughput scaling across API pods."
        },
        {
          title: "Feature delivery & reliability",
          body: "Shipped option chain APIs on Cloudflare Workers, greeks data, and Static IPs for partner integrations. Owned observability, monitoring, and primary on-call for the trading API surface."
        }
      ],
      stack: ["Java", "Spring Boot", "DuckDB", "Parquet", "Kubernetes", "GCS", "Cloudflare Workers"]
    },
    {
      company: "Groww",
      role: "Software Engineering Intern",
      period: "Jan 2025 — Jul 2025",
      location: "Bengaluru",
      blurb: "Worked across aggregation, auth, and subscription services. Led a top-to-bottom refactor of the trading API core.",
      bullets: [
        {
          title: "Trading API core refactor",
          body: "Comprehensive overhaul of the backend behind trading endpoints. Modularized components, replaced REST clients with Feign, integrated service SDKs, and applied the Factory pattern across handlers. Drove unit test coverage past 80%, halving release-cycle anxiety."
        },
        {
          title: "Python SDK & documentation",
          body: "Took ownership of the public Python SDK. Rewrote documentation, added clear usage examples, and built a daily integration test suite that catches regressions before users do."
        },
        {
          title: "Cross-service contributions",
          body: "Fixed bugs and shipped features in aggregation, auth, and subscription microservices. Better error messages, automated email flows for API subscriptions, and architecture docs for onboarding."
        }
      ],
      stack: ["Java", "Python", "Feign", "Spring Boot", "MySQL"]
    },
    {
      company: "DeepGrid",
      role: "Software Engineering Intern",
      period: "May 2024 — Dec 2024",
      location: "Remote",
      blurb: "Worked on natural-language interfaces for analytics — turning English questions into SQL and charts.",
      bullets: [
        {
          title: "Text-to-SQL pipeline",
          body: "Built a planner that decomposes complex natural-language cross-database queries into simpler SQL sub-queries and aggregates the results."
        },
        {
          title: "NL-to-chart generation",
          body: "Shipped a pipeline that produces plots directly from natural-language input, removing a step from the analyst workflow."
        }
      ],
      stack: ["Python", "LangChain", "FastAPI", "Matplotlib"]
    }
  ],

  education: {
    school: "Indian Institute of Information Technology, Allahabad",
    degree: "B.Tech, Electronics and Communication Engineering",
    period: "Dec 2021 — May 2025",
    detail: "CGPA 8.08 / 10.0"
  },

  skills: {
    Languages: ["Java", "Python", "SQL", "C++"],
    "Frameworks & Libraries": ["Spring Boot", "PyTorch", "FastAPI", "Scikit-Learn", "Streamlit", "NumPy", "Pandas", "Matplotlib"],
    Infrastructure: ["Kubernetes", "Docker", "ArgoCD", "GCS", "Cloudflare Workers"],
    Data: ["MySQL", "DuckDB", "Parquet", "Redis", "FAISS"]
  },

  projects: [
    {
      id: "duckdb-olap",
      title: "Embedded OLAP on Parquet",
      year: "2025",
      role: "Lead engineer",
      tags: ["infrastructure", "data", "work"],
      blurb: "Replaced a Redis-backed historical price cache with an embedded DuckDB service over partitioned Parquet on GCS. Sub-second on 500GB+ of candle data, no cluster to babysit.",
      stack: ["DuckDB", "Parquet", "Kubernetes", "GCS", "Java"],
      links: [],
      featured: true
    },
    {
      id: "retrieval-search",
      title: "Retrieval Search for E-Commerce",
      year: "2024",
      role: "Solo",
      tags: ["ml", "side-project"],
      blurb: "Semantic search system for 20,000+ products using FAISS and the gte-base-1.5 embedding model. Includes a text-to-image search path with GPU-accelerated PyTorch inference, served behind a Streamlit UI.",
      stack: ["Python", "FAISS", "Transformers", "PyTorch", "Streamlit"],
      links: [{ label: "GitHub", url: "https://github.com/sunny7712" }],
      featured: true
    },
    {
      id: "dcgan",
      title: "DCGAN from Scratch",
      year: "2024",
      role: "Solo",
      tags: ["ml", "side-project"],
      blurb: "Implemented Deep Convolutional GANs from the seminal paper — generator, discriminator, transposed convolutions, batch norm — plus a full training pipeline with TensorBoard visualizations.",
      stack: ["PyTorch", "TensorBoard"],
      links: [{ label: "GitHub", url: "https://github.com/sunny7712" }]
    },
    {
      id: "text-to-chart",
      title: "Natural Language → Chart",
      year: "2024",
      role: "Engineer",
      tags: ["ml", "work"],
      blurb: "Pipeline that converts plain-English questions about a dataset into matplotlib charts. Built during my internship at DeepGrid.",
      stack: ["Python", "LangChain", "FastAPI", "Matplotlib"],
      links: []
    },
    {
      id: "trading-sdk",
      title: "Groww Python SDK",
      year: "2025",
      role: "Maintainer",
      tags: ["work", "tooling"],
      blurb: "Rewrote documentation, added usage examples, and built an integration test suite that catches regressions before users do.",
      stack: ["Python", "pytest"],
      links: []
    },
    {
      id: "blog-engine",
      title: "This site",
      year: "2026",
      role: "Solo",
      tags: ["side-project", "tooling"],
      blurb: "The site you're reading. A small hash-routed React app with three switchable typographic systems and a cmd-K palette. Built because I'm tired of static-site generators.",
      stack: ["React", "Vanilla CSS", "Google Fonts"],
      links: []
    }
  ],

  posts: [
    {
      slug: "redis-to-duckdb",
      title: "Why we replaced Redis with DuckDB for historical backtests",
      date: "2026-03-14",
      tags: ["databases", "infrastructure", "work"],
      summary: "A read-mostly workload, a growing dataset, and a cache that started feeling more like a database. The case for treating analytics as a separate animal.",
      featured: true,
      body: [
        { type: "p", text: "Sometime in early 2025 we noticed the algo-trading platform's Redis cluster was being asked to do something Redis was never built for: range scans across hundreds of gigabytes of historical candle data. Backtests would issue a query for, say, every minute of NIFTY 50 between 2018 and 2024, and the cache would dutifully assemble the answer at the cost of holding most of its working set in RAM." },
        { type: "p", text: "It worked. It worked the way a hammer works for screws — loud, expensive, and the screws are bent afterward. We were paying for memory we didn't really need and pretending a key-value store was an analytics database." },
        { type: "h2", text: "The shape of the workload" },
        { type: "p", text: "What backtests actually want is depressingly simple. Given a symbol and a date range, return the candles. Maybe at a different resolution. Maybe enriched with corporate actions. Everything is read-only, everything is range-scoped, and the dataset grows by exactly one trading day per trading day." },
        { type: "p", text: "Once you say it out loud it sounds like the columnar-Parquet-and-OLAP brochure, which is what we ended up building." },
        { type: "h2", text: "What we built" },
        { type: "p", text: "The service is an embedded DuckDB instance running inside an ordinary Java API pod. The data lives in partitioned Parquet on GCS, mounted into the container via NFS. DuckDB does the heavy lifting; the Java layer handles auth, request shaping, and translating into SQL." },
        { type: "code", lang: "java", text: "// roughly\nString sql = \"SELECT timestamp, open, high, low, close, volume \"\n          + \"FROM read_parquet('/mnt/candles/symbol=' || ? || '/year=*/month=*/data.parquet') \"\n          + \"WHERE timestamp BETWEEN ? AND ? \"\n          + \"ORDER BY timestamp\";\nreturn duckdb.query(sql, symbol, from, to);" },
        { type: "p", text: "Each pod has its own DuckDB process. No coordination, no cluster, no leader election, no node-down panic at 2 a.m. Scaling out is the same as scaling out any other stateless API: add pods." },
        { type: "h2", text: "What we measured" },
        { type: "p", text: "On a typical workload — 500 GB+ of candle data, queries spanning a few years — we landed at sub-second p95 latency. Throughput scales linearly with the number of API pods, which is what you'd hope for from an embedded engine on a shared filesystem." },
        { type: "p", text: "More important than the latency numbers: the operational footprint collapsed. The team page for the Redis cluster has been quiet for months." },
        { type: "h2", text: "What I'd say to past me" },
        { type: "p", text: "If your cache is the database, just admit it and use a database. Cleanly separating the hot transactional path (Redis, fine) from the historical analytical path (DuckDB + Parquet, very fine) made the system both faster and cheaper. The hard part was the cultural shift — convincing a team that 'just a Java service' could do analytics work." }
      ]
    },
    {
      slug: "embedded-olap",
      title: "The case for embedded OLAP",
      date: "2026-02-02",
      tags: ["databases", "thoughts"],
      summary: "Distributed query engines are wonderful and necessary. Most workloads don't need one.",
      body: [
        { type: "p", text: "There is a specific kind of engineer who, upon being handed a 100 GB analytics workload, reaches first for a Snowflake account or a Spark cluster. I sympathize — those are extraordinary tools — but they are also the wrong tools, most of the time, for the size of problem most of us actually have." },
        { type: "p", text: "An embedded OLAP engine (DuckDB, Polars, chDB) running in the same process as your API can do startling things. The full table-scan-and-aggregate machinery, ready to go, no coordination overhead, deployed exactly the same way you deploy any other library." },
        { type: "h2", text: "When it stops working" },
        { type: "p", text: "When the data outgrows a single machine's storage budget, or when you need true concurrency over writes, or when you want a SQL-accessible warehouse for ad-hoc analysts. Those are real reasons. 'It feels too simple' is not one of them." }
      ]
    },
    {
      slug: "feign",
      title: "Notes on Feign vs RestTemplate",
      date: "2025-11-09",
      tags: ["java", "work"],
      summary: "Why we deleted the RestTemplates and what we got back.",
      body: [
        { type: "p", text: "RestTemplate works, but every call site grows tentacles: timeouts, retries, deserialization, error mapping, the works. Eight services later, you have a hundred slightly-different ways to call a hundred slightly-different endpoints." },
        { type: "p", text: "Feign collapses all of that into an interface plus an annotation. The discipline is enforced by the type system. The retry policy lives in one place. The deserialization is consistent. The diff was, frankly, embarrassing in our favor." }
      ]
    },
    {
      slug: "sdk-design",
      title: "How to write a Python SDK people actually use",
      date: "2025-09-21",
      tags: ["python", "tooling"],
      summary: "A short list of things I wish someone had handed me when I inherited an SDK that nobody was opening.",
      body: [
        { type: "p", text: "Most SDKs are written by the team that wrote the API, which is exactly the team least equipped to write an SDK. They already know what the API does. They already have the mental model. They write a thin wrapper around HTTP and call it a day." },
        { type: "p", text: "The trick is to pretend you're a user who has never heard of your company before. What do they want to type first? What's the smallest end-to-end example that would make them say 'oh, I see'? Put that on page one of the README and work backward from there." }
      ]
    },
    {
      slug: "first-six-months",
      title: "Things I learned in my first six months as an SDE",
      date: "2025-12-30",
      tags: ["career", "thoughts"],
      summary: "Mostly humbling, occasionally funny. Notes from the transition out of school.",
      body: [
        { type: "p", text: "The first lesson is that nobody is going to give you a Jira ticket that says 'figure out what's important'. That's the job. The job is mostly choosing which fire to put out and which to let burn." },
        { type: "p", text: "The second lesson is that the senior engineers you admire are not, in fact, smarter than you. They've just developed an almost theological aversion to clever code, and that aversion has saved them — and the system — a thousand times." }
      ]
    },
    {
      slug: "embeddings-quick-note",
      title: "A short note on dot-product similarity",
      date: "2025-08-15",
      tags: ["ml", "notes"],
      summary: "Why cosine similarity is just normalized dot product, and when you should care about the distinction.",
      body: [
        { type: "p", text: "Almost every retrieval system I've worked on uses one of two similarity metrics: dot product $a \\cdot b$ or cosine similarity. People sometimes treat them as different things; mostly they aren't." },
        { type: "h2", text: "The two definitions" },
        { type: "p", text: "Dot product is the obvious one. For two vectors $a, b \\in \\mathbb{R}^n$:" },
        { type: "math", text: "a \\cdot b = \\sum_{i=1}^{n} a_i b_i" },
        { type: "p", text: "Cosine similarity normalizes by magnitude:" },
        { type: "math", text: "\\cos(\\theta) = \\frac{a \\cdot b}{\\|a\\| \\, \\|b\\|}" },
        { type: "p", text: "If $\\|a\\| = \\|b\\| = 1$ — i.e., your embeddings are L2-normalized — then $\\cos(\\theta) = a \\cdot b$. They are the same number." },
        { type: "h2", text: "When the distinction matters" },
        { type: "p", text: "If your embedding model produces unit-norm vectors (most modern sentence encoders do), reach for raw dot product. It's one less square root per query and FAISS will thank you." },
        { type: "p", text: "If your vectors have unbounded magnitude — counts, term frequencies, anything you built by hand — use cosine. Otherwise, a long document with high term counts looks 'more similar' to everything than a short one, which is rarely what you want." },
        { type: "h2", text: "What this looks like in practice" },
        { type: "p", text: "For our product-search system at $20{,}000$ items, L2-normalizing once at index time and then using inner product as the FAISS metric gave us a ~$15\\%$ latency improvement over computing cosine on the fly. Same recall, fewer cycles. The kind of micro-optimization that's only worth it when the normalization step has nothing to do with the rest of your hot path — which, conveniently, it doesn't." }
      ]
    }
  ],

  now: {
    updated: "May 24, 2026",
    sections: [
      {
        title: "Working on",
        items: [
          "Historical data API at Groww — making backtests faster and the engine cheaper to operate.",
          "Internal tooling around our embedded DuckDB layer — query plan introspection, cache warming, dataset versioning.",
          "Slowly writing a longer piece on storage formats for time series."
        ]
      },
      {
        title: "Reading",
        items: [
          "Designing Data-Intensive Applications — third re-read, still finding new things.",
          "The papers behind DuckDB. The implementation is half the fun.",
          "Less work-shaped: Annie Dillard, The Writing Life."
        ]
      },
      {
        title: "Trying to get better at",
        items: [
          "Writing. Specifically: shorter. Specifically: cutting the second draft in half.",
          "Cooking biryani that doesn't end in either crunchy rice or charred onions.",
          "Sleeping eight hours when there's an on-call rotation."
        ]
      },
      {
        title: "Not working on",
        items: [
          "Any new side projects until the current ones ship.",
          "Twitter, mostly.",
          "Convincing anyone of anything."
        ]
      }
    ]
  }
};
