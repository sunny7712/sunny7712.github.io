// Projects index + Project detail.
// Detail body is fetched from content/projects/{id}.md and rendered via marked.

const { useState: useState_pp, useEffect: useEffect_pp, useMemo: useMemo_pp } = React;

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
        title="Things I've built, mostly to learn and sometimes for fun."
        sub="A mix of side projects, and weekend experiments. Click through for the longer story."
      />

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
        {filtered.map((p) => (
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
                  {p.stack.map((s, i) => (
                    <React.Fragment key={s}>
                      {i > 0 && <span style={{ color: "var(--fg-mute)", margin: "0 2px" }}>·</span>}
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-mute)" }}>{s}</span>
                    </React.Fragment>
                  ))}
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
  const [bodyHtml, setBodyHtml] = useState_pp(null);

  useEffect_pp(() => {
    setBodyHtml(null);
    fetch("content/projects/" + id + ".md")
      .then((r) => { if (!r.ok) throw new Error(); return r.text(); })
      .then((md) => setBodyHtml(window.marked.parse(md)))
      .catch(() => {
        setBodyHtml("<p style='color:var(--fg-mute);font-family:var(--font-mono);font-size:12px'>Story not found — add <code>content/projects/" + id + ".md</code>.</p>");
      });
  }, [id]);

  if (!p) {
    return (
      <div className="page">
        <PageHeader eyebrow="404" title="Project not found." sub="Probably a stale link." />
        <a className="link" href="#/projects" onClick={(e) => { e.preventDefault(); onNavigate("/projects"); }}>← all projects</a>
      </div>
    );
  }

  const list = window.SITE.projects;
  const idx = list.findIndex((x) => x.id === id);
  const prev = list[(idx - 1 + list.length) % list.length];
  const next = list[(idx + 1) % list.length];

  return (
    <div className="page">
      <div style={{ marginBottom: "var(--gap-4)" }}>
        <a className="link muted" href="#/projects" onClick={(e) => { e.preventDefault(); onNavigate("/projects"); }}>← projects</a>
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
          {bodyHtml === null
            ? <p style={{ color: "var(--fg-mute)", fontFamily: "var(--font-mono)", fontSize: 12 }}>Loading…</p>
            : <div dangerouslySetInnerHTML={{ __html: bodyHtml }} />
          }
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
          {p.links && p.links.length > 0 && (
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
        .pn-card{padding:var(--gap-4);border:1px solid var(--rule);border-radius:var(--radius);transition:transform var(--speed) var(--ease),border-color var(--speed) var(--ease);}
        .pn-card:hover{border-color:var(--fg);transform:translateY(-2px);}
        .prose h2{font-family:var(--font-display);font-size:28px;font-weight:var(--display-weight);letter-spacing:var(--display-tracking);margin:var(--gap-6) 0 var(--gap-3);}
        .prose p{margin:0 0 var(--gap-3);}
        @media (max-width: 720px){.proj-detail-grid{grid-template-columns:1fr!important;}.pn-grid{grid-template-columns:1fr!important;}}
      `}</style>
    </div>
  );
}

Object.assign(window, { ProjectsPage, ProjectDetailPage });
