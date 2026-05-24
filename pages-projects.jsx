// Projects index + Project detail.

const { useState: useState_pp, useMemo: useMemo_pp } = React;

function ProjectsPage({ onNavigate }) {
  const all = window.SITE.projects;
  const [activeTag, setTag] = useState_pp("all");

  const tags = useMemo_pp(() => {
    const set = new Set();
    all.forEach((p) => p.tags.forEach((t) => set.add(t)));
    return ["all", ...set];
  }, [all]);

  const filtered = useMemo_pp(() => activeTag === "all" ? all : all.filter((p) => p.tags.includes(activeTag)), [activeTag, all]);

  return (
    <div className="page">
      <PageHeader
        eyebrow={"Projects · " + all.length + " total"}
        title="Things I've built, mostly for fun."
        sub="A mix of work projects (Groww), internships, and weekend experiments. Click through for the longer story."
      />

      {/* tag filter */}
      <div className="row-h" style={{ gap: 8, marginTop: "var(--gap-6)" }}>
        <span className="label" style={{ marginRight: 6 }}>Filter</span>
        {tags.map((t) => (
          <button key={t}
                  className={"tag" + (activeTag === t ? " active" : "")}
                  onClick={() => setTag(t)}
                  style={{ cursor: "pointer" }}>
            {t}
          </button>
        ))}
        <span className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11, marginLeft: 8 }}>
          {filtered.length} {filtered.length === 1 ? "project" : "projects"}
        </span>
      </div>

      <div style={{ marginTop: "var(--gap-5)" }}>
        {filtered.map((p, i) => (
          <a key={p.id}
             href={"#/projects/" + p.id}
             onClick={(e) => { e.preventDefault(); onNavigate("/projects/" + p.id); }}
             className="proj-row">
            <div className="proj-row-inner">
              <div className="proj-meta">
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-mute)", letterSpacing: "var(--label-tracking)" }}>{p.year}</span>
                <span className="row-h" style={{ gap: 6, marginTop: 8 }}>
                  {p.tags.map((t) => <span key={t} className="tag" style={{ fontSize: 9.5 }}>{t}</span>)}
                </span>
              </div>
              <div className="proj-main">
                <div className="display" style={{ fontSize: 30, marginBottom: 6 }}>{p.title}</div>
                <p className="muted" style={{ fontSize: 15, maxWidth: 620, margin: 0 }}>{p.blurb}</p>
                <div className="row-h" style={{ gap: 6, marginTop: 10 }}>
                  {p.stack.map((s) => (
                    <span key={s} style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-mute)" }}>{s}</span>
                  )).reduce((acc, el, i) => i === 0 ? [el] : [...acc, <span key={"sep" + i} style={{ color: "var(--fg-mute)", margin: "0 6px" }}>·</span>, el], [])}
                </div>
              </div>
              <div className="proj-arrow">→</div>
            </div>
          </a>
        ))}
      </div>

      <style>{`
        .proj-row{display:block;border-top:1px solid var(--rule);transition:background var(--speed) var(--ease);}
        .proj-row:last-child{border-bottom:1px solid var(--rule);}
        .proj-row:hover{background:var(--bg-soft);}
        .proj-row-inner{display:grid;grid-template-columns:140px 1fr 40px;gap:24px;padding:var(--gap-4) 0;align-items:start;}
        .proj-row:hover .proj-arrow{transform:translateX(6px);color:var(--accent);}
        .proj-arrow{font-family:var(--font-mono);color:var(--fg-mute);transition:all var(--speed) var(--ease);padding-top:4px;}
        [data-theme="warm"] .proj-row{padding:0 var(--gap-3);border-radius:var(--radius);border-top:0;background:var(--surface);margin-bottom:var(--gap-2);}
        [data-theme="warm"] .proj-row:hover{background:var(--bg-soft);}
        [data-theme="warm"] .proj-row:last-child{border-bottom:0;}
        @media (max-width: 720px){.proj-row-inner{grid-template-columns:1fr;gap:8px;}.proj-arrow{display:none;}}
      `}</style>
    </div>
  );
}

function ProjectDetailPage({ id, onNavigate }) {
  const p = window.SITE.projects.find((x) => x.id === id);
  if (!p) {
    return (
      <div className="page">
        <PageHeader eyebrow="404" title="Project not found." sub="Probably a stale link. Head back to the list?" />
        <a className="link" href="#/projects" onClick={(e) => { e.preventDefault(); onNavigate("/projects"); }}>← all projects</a>
      </div>
    );
  }

  // build prev/next within list
  const list = window.SITE.projects;
  const idx = list.findIndex((x) => x.id === id);
  const prev = list[(idx - 1 + list.length) % list.length];
  const next = list[(idx + 1) % list.length];

  return (
    <div className="page">
      <div style={{ marginBottom: "var(--gap-4)" }}>
        <a className="link muted" href="#/projects"
           onClick={(e) => { e.preventDefault(); onNavigate("/projects"); }}>← projects</a>
      </div>

      <header style={{ marginBottom: "var(--gap-6)" }}>
        <div className="eyebrow" style={{ marginBottom: "var(--gap-3)" }}>
          {p.year} · {p.role} · {p.tags.join(" · ")}
        </div>
        <h1 className="display" style={{ fontSize: "clamp(40px, 6vw, 72px)", marginBottom: "var(--gap-3)" }}>{p.title}</h1>
        <p className="muted" style={{ maxWidth: 680, fontSize: 18 }}>{p.blurb}</p>
      </header>

      <div className="media" style={{ aspectRatio: "16 / 8", marginBottom: "var(--gap-6)" }}>
        screenshot · drop-in (1600 × 800)
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 240px", gap: "var(--gap-6)" }} className="proj-detail-grid">
        <div className="prose" style={{ maxWidth: "none" }}>
          <h2>The problem</h2>
          <p>{projectStory(p).problem}</p>
          <h2>What I did</h2>
          <p>{projectStory(p).approach}</p>
          <h2>What it taught me</h2>
          <p>{projectStory(p).lesson}</p>
        </div>
        <aside style={{ position: "sticky", top: 24, alignSelf: "start" }} className="stack">
          <div className="stack-tight">
            <div className="label">Role</div>
            <div>{p.role}</div>
          </div>
          <div className="stack-tight">
            <div className="label">Year</div>
            <div>{p.year}</div>
          </div>
          <div className="stack-tight">
            <div className="label">Stack</div>
            <div className="row-h" style={{ gap: 6 }}>
              {p.stack.map((s) => <span key={s} className="tag">{s}</span>)}
            </div>
          </div>
          {p.links.length > 0 && (
            <div className="stack-tight">
              <div className="label">Links</div>
              <div className="stack-tight">
                {p.links.map((l) => (
                  <a key={l.url} className="link" href={l.url} target="_blank" rel="noreferrer">{l.label} ↗</a>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>

      {/* prev / next */}
      <nav style={{ marginTop: "var(--gap-8)", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--gap-4)" }} className="pn-grid">
        <a className="pn-card" href={"#/projects/" + prev.id} onClick={(e) => { e.preventDefault(); onNavigate("/projects/" + prev.id); }}>
          <div className="label">← Previous</div>
          <div className="display" style={{ fontSize: 22, marginTop: 6 }}>{prev.title}</div>
        </a>
        <a className="pn-card" href={"#/projects/" + next.id} onClick={(e) => { e.preventDefault(); onNavigate("/projects/" + next.id); }} style={{ textAlign: "right" }}>
          <div className="label">Next →</div>
          <div className="display" style={{ fontSize: 22, marginTop: 6 }}>{next.title}</div>
        </a>
      </nav>
      <style>{`
        .pn-card{padding:var(--gap-4);border:1px solid var(--rule);border-radius:var(--radius);
                 transition:transform var(--speed) var(--ease), border-color var(--speed) var(--ease);}
        .pn-card:hover{border-color:var(--fg);transform:translateY(-2px);}
        @media (max-width: 720px){
          .proj-detail-grid{grid-template-columns:1fr!important;}
          .pn-grid{grid-template-columns:1fr!important;}
        }
      `}</style>
    </div>
  );
}

// invented narrative scaffolding so each project has a "story"
function projectStory(p) {
  const map = {
    "duckdb-olap": {
      problem: "Our Redis-backed historical data path was straining under range-scan queries from the algo-trading backtester. The cluster's memory bill grew faster than the value the cache delivered, and a stateful cluster meant late-night pages whenever a node misbehaved.",
      approach: "Designed an embedded analytics layer using DuckDB inside our existing Java pods, reading partitioned Parquet files from GCS mounted via NFS. No coordination, no leader election — just a query engine running next to the API code. Wrote a thin SQL builder and ported the existing endpoints over.",
      lesson: "Cache-shaped problems and analytics-shaped problems look identical until you measure them. Separating the two unlocked a 10× reduction in operational complexity and let us right-size each path independently."
    },
    "retrieval-search": {
      problem: "Keyword search over a 20,000-item product catalog was missing too many obvious matches. Users typed natural language; the catalog spoke SKU.",
      approach: "Embedded the catalog with gte-base-1.5, indexed in FAISS, and added a text-to-image path with PyTorch-on-GPU. Wrapped the whole thing in Streamlit so I could ship it before losing interest.",
      lesson: "Vector search isn't magic, but the gap between BM25 and a half-decent embedding model is bigger than I expected. Also: Streamlit is a wildly good prototyping tool."
    },
    "dcgan": {
      problem: "I wanted to understand GANs deeply enough to debug one. Reading the paper wasn't getting me there.",
      approach: "Implemented DCGAN from scratch in PyTorch — generator, discriminator, transposed convs, batch norm, the whole apparatus. Wrote the training loop, hooked up TensorBoard, watched the loss oscillate in real time.",
      lesson: "Half of understanding a paper is rebuilding it. The other half is staring at a confused discriminator at 2 a.m. wondering why it stopped learning."
    },
    "text-to-chart": {
      problem: "Analysts at DeepGrid wanted plots from English-language questions, without writing matplotlib by hand.",
      approach: "Built a small planner that maps natural-language requests to a chart spec, then renders via matplotlib. Wrapped behind a FastAPI service consumed by the UI team.",
      lesson: "Most 'natural language' problems are really 'how strict can your schema be while still feeling natural?' problems."
    },
    "trading-sdk": {
      problem: "We had a Python SDK, but adoption was lower than expected and the README hadn't been touched in a year.",
      approach: "Rewrote documentation around the user's first-five-minutes experience. Added a daily integration test that hits the real API and posts to Slack on regression. Folded examples into the test suite so they can't drift.",
      lesson: "An SDK's value is mostly its README. Code is secondary."
    },
    "blog-engine": {
      problem: "Every static-site generator I tried felt either too magical or not magical enough. I wanted something I could read end-to-end in one sitting.",
      approach: "A few hundred lines of React, three CSS theme files, a hash router, and a JSON-shaped data file. No build step, no plugins, no markdown. The constraints turned out to be the feature.",
      lesson: "If you write your own portfolio engine, you'll never run out of excuses to redesign it. This is both the upside and the curse."
    }
  };
  return map[p.id] || { problem: p.blurb, approach: "Details forthcoming.", lesson: "Details forthcoming." };
}

Object.assign(window, { ProjectsPage, ProjectDetailPage });
