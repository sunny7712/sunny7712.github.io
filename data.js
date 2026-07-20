// Static site data. Identity, experience, education, skills, and "now" live here.
//
// Posts  → write content/posts/{slug}.md, list in content/posts/_index.json
// Projects → write content/projects/{id}.md, list in content/projects/_index.json
//
// The app fetches both _index.json files on load and populates window.SITE.posts
// and window.SITE.projects at runtime.

window.SITE = {
  identity: {
    name: "Vamsi K",
    short: "Vamsi K",
    role: "Software Engineer",
    location: "Bengaluru, India",
    tagline: "Building software for traders, at Groww. Interested in databases, distributed systems, infra and the messy seams between them.",
    intro: "I'm a software engineer working around trading infrastructure. Most days I'm thinking about software architecture, writing clean, maintainable and extensible code, and how to make systems easier to operate. Before that I was at IIIT Allahabad studying electronics.",
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
      blurb: "Core maintainer for the 915 and trading API services. Building the historical data layer powering algorithmic backtests.",
      bullets: [
        {
          title: "Historical Data API for Algo Trading",
          body: "Designed and shipped a dedicated OLAP service to replace Redis for long-range historical backtests. Embedded analytics on DuckDB, querying partitioned Parquet on GCS mounted via NFS on Kubernetes. Avoided the overhead of a distributed cluster while delivering sub-second latency on 500GB+ of candle data, with linear throughput scaling across API pods."
        },
        {
          title: "Feature delivery & reliability",
          body: "Shipped option chain APIs on Cloudflare Workers, greeks data, and Static IPs for compliance and regulation. Owned observability, monitoring, and primary on-call for the trading API surface."
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
          body: "Comprehensive overhaul of the backend behind trading endpoints. Inherited a monolithic single-module service. Decomposed it into logical modules by functionality. Applied Adapter, Strategy, and Factory patterns to make the code reusable, extensible, and easy to refactor. Drove unit test coverage past 80%, halving release-cycle anxiety."
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

  now: {
    updated: "May 24, 2026",
    sections: [
      {
        title: "Working on",
        items: [
          "Currently working on 915 at Groww. ",
          "Auth system for a side project I'm building. More on that soon.",
        ]
      },
      {
        title: "Reading",
        items: [
          "Designing Data Intensive Applications By Martin Klepmann"
        ]
      },
      {
        title: "Trying to get better at",
        items: [
          "Writing. Trying to write more, and write better. ",
          "Running. Trying to be a bit more consistent, and run a sub-25 minute 5K by the end of the year.",
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
  },

  // Populated at runtime by app.jsx from content/*/_index.json
  posts: [],
  projects: []
};
