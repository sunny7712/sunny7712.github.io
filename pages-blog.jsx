// Blog index + Blog post (ToC, reading time, scroll-spy).

const { useState: useState_b, useEffect: useEffect_b, useMemo: useMemo_b, useRef: useRef_b } = React;

function BlogIndexPage({ onNavigate }) {
  const all = window.SITE.posts;
  const [activeTag, setTag] = useState_b("all");
  const tags = useMemo_b(() => {
    const s = new Set();
    all.forEach((p) => p.tags.forEach((t) => s.add(t)));
    return ["all", ...s];
  }, [all]);
  const filtered = activeTag === "all" ? all : all.filter((p) => p.tags.includes(activeTag));

  return (
    <div className="page">
      <PageHeader
        eyebrow={"Writing · " + all.length + " posts"}
        title="Mostly about databases. Sometimes about life."
        sub="Notes-to-self that escaped into the open. I write to figure out what I think — these are the residue."
      />

      <div className="row-h" style={{ gap: 8, marginTop: "var(--gap-6)" }}>
        <span className="label" style={{ marginRight: 6 }}>Topics</span>
        {tags.map((t) => (
          <button key={t} className={"tag" + (activeTag === t ? " active" : "")} onClick={() => setTag(t)}>{t}</button>
        ))}
      </div>

      <div style={{ marginTop: "var(--gap-5)" }}>
        {filtered.map((p) => (
          <a key={p.slug} href={"#/blog/" + p.slug}
             onClick={(e) => { e.preventDefault(); onNavigate("/blog/" + p.slug); }}
             className="post-row">
            <div className="pr-inner">
              <div className="pr-meta">
                <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-mute)", letterSpacing: "var(--label-tracking)" }}>{formatDate(p.date)}</div>
                <div className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11, marginTop: 4 }}>{readingTime(p)} min read</div>
              </div>
              <div className="pr-main">
                <div className="display" style={{ fontSize: 28, marginBottom: 6 }}>{p.title}</div>
                <p className="muted" style={{ fontSize: 15, maxWidth: 620, margin: 0 }}>{p.summary}</p>
                <div className="row-h" style={{ gap: 6, marginTop: 10 }}>
                  {p.tags.map((t) => <span key={t} className="tag" style={{ fontSize: 9.5 }}>{t}</span>)}
                </div>
              </div>
            </div>
          </a>
        ))}
      </div>

      <style>{`
        .post-row{display:block;border-top:1px solid var(--rule);transition:background var(--speed) var(--ease);}
        .post-row:last-child{border-bottom:1px solid var(--rule);}
        .post-row:hover{background:var(--bg-soft);}
        .pr-inner{display:grid;grid-template-columns:140px 1fr;gap:24px;padding:var(--gap-4) 0;align-items:start;}
        [data-theme="warm"] .post-row{padding:0 var(--gap-3);border-radius:var(--radius);border-top:0;background:var(--surface);margin-bottom:var(--gap-2);}
        [data-theme="warm"] .post-row:hover{background:var(--bg-soft);}
        [data-theme="warm"] .post-row:last-child{border-bottom:0;}
        @media (max-width: 720px){.pr-inner{grid-template-columns:1fr;gap:8px;}}
      `}</style>
    </div>
  );
}

function BlogPostPage({ slug, onNavigate }) {
  const post = window.SITE.posts.find((p) => p.slug === slug);
  const [activeId, setActiveId] = useState_b(null);
  const [progress, setProgress] = useState_b(0);
  const articleRef = useRef_b(null);

  // headings list
  const headings = useMemo_b(() => {
    if (!post) return [];
    return post.body.filter((b) => b.type === "h2").map((b) => ({ text: b.text, id: slugify(b.text) }));
  }, [post]);

  // scroll-spy and progress
  useEffect_b(() => {
    if (!post) return;
    const onScroll = () => {
      // progress
      const el = articleRef.current;
      if (el) {
        const top = el.getBoundingClientRect().top + window.scrollY;
        const h = el.scrollHeight;
        const scrolled = Math.min(1, Math.max(0, (window.scrollY - top + window.innerHeight * 0.3) / h));
        setProgress(scrolled * 100);
      }
      // spy
      let current = headings[0]?.id || null;
      for (const h of headings) {
        const node = document.getElementById(h.id);
        if (!node) continue;
        const top = node.getBoundingClientRect().top;
        if (top < 120) current = h.id; else break;
      }
      setActiveId(current);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, [headings, post]);

  if (!post) {
    return (
      <div className="page">
        <PageHeader eyebrow="404" title="Post not found." sub="Wrong link or a deleted draft." />
        <a className="link" href="#/blog" onClick={(e) => { e.preventDefault(); onNavigate("/blog"); }}>← all writing</a>
      </div>
    );
  }

  // build prev/next
  const posts = window.SITE.posts;
  const idx = posts.findIndex((p) => p.slug === slug);
  const older = posts[(idx + 1) % posts.length];
  const newer = posts[(idx - 1 + posts.length) % posts.length];

  return (
    <div className="page">
      {/* progress bar */}
      <div style={{
        position: "fixed", left: 0, top: 0, height: 2, width: progress + "%",
        background: "var(--accent)", zIndex: 999, transition: "width 80ms linear"
      }}></div>

      <div style={{ marginBottom: "var(--gap-4)" }}>
        <a className="link muted" href="#/blog"
           onClick={(e) => { e.preventDefault(); onNavigate("/blog"); }}>← writing</a>
      </div>

      <header style={{ marginBottom: "var(--gap-6)" }}>
        <div className="eyebrow" style={{ marginBottom: "var(--gap-3)" }}>
          {formatDate(post.date)} · {readingTime(post)} min read
        </div>
        <h1 className="display" style={{ fontSize: "clamp(36px, 5.5vw, 60px)", marginBottom: "var(--gap-3)" }}>{post.title}</h1>
        <p className="muted" style={{ maxWidth: 680, fontSize: 18 }}>{post.summary}</p>
        <div className="row-h" style={{ gap: 6, marginTop: "var(--gap-3)" }}>
          {post.tags.map((t) => <span key={t} className="tag">{t}</span>)}
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 220px", gap: "var(--gap-6)" }} className="post-grid">
        <article ref={articleRef} className="prose">
          {post.body.map((b, i) => {
            if (b.type === "p") return <p key={i}>{renderInline(b.text)}</p>;
            if (b.type === "h2") return <h2 key={i} id={slugify(b.text)}>{b.text}</h2>;
            if (b.type === "code") return <pre key={i}><code>{b.text}</code></pre>;
            if (b.type === "math") return <BlockMath key={i} tex={b.text} />;
            return null;
          })}

          {/* Signature */}
          <hr style={{ border: 0, borderTop: "1px solid var(--rule)", margin: "var(--gap-6) 0 var(--gap-3)" }} />
          <p className="muted" style={{ fontSize: 14 }}>
            Got thoughts? Reach me on <a className="link" href={window.SITE.identity.socials[1].url} target="_blank" rel="noreferrer">Twitter</a> or drop a <a className="link" href="#/contact" onClick={(e) => { e.preventDefault(); onNavigate("/contact"); }}>note</a>.
          </p>
        </article>

        <aside className="post-aside">
          {headings.length > 0 && (
            <div className="toc">
              <div className="toc-title">In this post</div>
              {headings.map((h) => (
                <a key={h.id}
                   href={"#" + h.id}
                   className={activeId === h.id ? "active" : ""}
                   onClick={(e) => {
                     e.preventDefault();
                     const node = document.getElementById(h.id);
                     if (node) window.scrollTo({ top: node.offsetTop - 24, behavior: "smooth" });
                   }}>
                  {h.text}
                </a>
              ))}
            </div>
          )}
        </aside>
      </div>

      {/* prev/next */}
      <nav style={{ marginTop: "var(--gap-8)", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--gap-4)" }} className="pn-grid">
        <a className="pn-card" href={"#/blog/" + newer.slug} onClick={(e) => { e.preventDefault(); onNavigate("/blog/" + newer.slug); }}>
          <div className="label">← Newer</div>
          <div className="display" style={{ fontSize: 20, marginTop: 6 }}>{newer.title}</div>
        </a>
        <a className="pn-card" href={"#/blog/" + older.slug} onClick={(e) => { e.preventDefault(); onNavigate("/blog/" + older.slug); }} style={{ textAlign: "right" }}>
          <div className="label">Older →</div>
          <div className="display" style={{ fontSize: 20, marginTop: 6 }}>{older.title}</div>
        </a>
      </nav>

      <style>{`
        @media (max-width: 860px){
          .post-grid{grid-template-columns:1fr!important;}
          .post-aside{order:-1;}
          .toc{position:static!important;}
        }
      `}</style>
    </div>
  );
}

function slugify(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// ── Math rendering via KaTeX ──────────────────────────────
// renderInline turns a string into an array of nodes, treating $...$ as
// inline math. Use it on any plain-string body content.
function renderInline(text) {
  if (!text || typeof text !== "string" || text.indexOf("$") === -1) return text;
  const parts = [];
  let i = 0;
  let cursor = 0;
  let key = 0;
  while (i < text.length) {
    if (text[i] === "$" && text[i + 1] !== "$") {
      // find matching $
      const end = text.indexOf("$", i + 1);
      if (end === -1) break;
      if (i > cursor) parts.push(text.slice(cursor, i));
      parts.push(<InlineMath key={key++} tex={text.slice(i + 1, end)} />);
      cursor = end + 1;
      i = cursor;
    } else {
      i++;
    }
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return parts.length ? parts : text;
}

function InlineMath({ tex }) {
  const ref = useRef_b(null);
  useEffect_b(() => {
    const render = () => {
      if (!ref.current) return;
      if (!window.katex) { setTimeout(render, 50); return; }
      try { window.katex.render(tex, ref.current, { throwOnError: false, displayMode: false }); }
      catch (e) { if (ref.current) ref.current.textContent = tex; }
    };
    render();
  }, [tex]);
  return <span ref={ref} className="math-inline"></span>;
}

function BlockMath({ tex }) {
  const ref = useRef_b(null);
  useEffect_b(() => {
    const render = () => {
      if (!ref.current) return;
      if (!window.katex) { setTimeout(render, 50); return; }
      try { window.katex.render(tex, ref.current, { throwOnError: false, displayMode: true }); }
      catch (e) { if (ref.current) ref.current.textContent = tex; }
    };
    render();
  }, [tex]);
  return <div ref={ref} className="math-block" style={{ margin: "var(--gap-4) 0", overflowX: "auto" }}></div>;
}

Object.assign(window, { BlogIndexPage, BlogPostPage, slugify });
