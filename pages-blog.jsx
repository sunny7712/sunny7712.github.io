// Blog index + Blog post (ToC, reading time, scroll-spy, markdown body).
// Body is fetched from content/posts/{slug}.md and rendered via marked + KaTeX auto-render.

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
        title="Mostly about Software Engineering and things I find interesting."
        sub=""
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
                <div className="muted" style={{ fontFamily: "var(--font-mono)", fontSize: 11, marginTop: 4 }}>{p.readingTime || 1} min read</div>
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
  const [bodyHtml, setBodyHtml] = useState_b(null);
  const [headings, setHeadings] = useState_b([]);
  const [activeId, setActiveId] = useState_b(null);
  const [progress, setProgress] = useState_b(0);
  const articleRef = useRef_b(null);

  // Fetch and parse markdown body
  useEffect_b(() => {
    setBodyHtml(null);
    setHeadings([]);
    fetch("content/posts/" + slug + ".md")
      .then((r) => { if (!r.ok) throw new Error(); return r.text(); })
      .then((md) => {
        // Render markdown and inject IDs into h2 elements for scroll-spy
        const html = window.marked.parse(md).replace(
          /<h2>(.*?)<\/h2>/gs,
          (_, inner) => `<h2 id="${slugify(inner.replace(/<[^>]+>/g, "").trim())}">${inner}</h2>`
        );
        // Extract headings from markdown source for ToC
        const h = [...md.matchAll(/^## (.+)$/gm)].map((m) => ({
          text: m[1].trim(),
          id: slugify(m[1].trim()),
        }));
        setBodyHtml(html);
        setHeadings(h);
      })
      .catch(() => {
        setBodyHtml("<p style='color:var(--fg-mute);font-family:var(--font-mono);font-size:12px'>Body not found — add <code>content/posts/" + slug + ".md</code>.</p>");
      });
  }, [slug]);

  // KaTeX auto-render after body HTML lands in DOM
  useEffect_b(() => {
    if (!bodyHtml || !articleRef.current) return;
    const run = () => {
      if (!window.renderMathInElement) { setTimeout(run, 60); return; }
      window.renderMathInElement(articleRef.current, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$",  right: "$",  display: false },
        ],
        throwOnError: false,
      });
    };
    run();
  }, [bodyHtml]);

  // Scroll-spy + progress bar
  useEffect_b(() => {
    if (!post) return;
    const onScroll = () => {
      const el = articleRef.current;
      if (el) {
        const top = el.getBoundingClientRect().top + window.scrollY;
        setProgress(Math.min(100, Math.max(0, (window.scrollY - top + window.innerHeight * 0.3) / el.scrollHeight * 100)));
      }
      let cur = headings[0]?.id || null;
      for (const h of headings) {
        const node = document.getElementById(h.id);
        if (node && node.getBoundingClientRect().top < 120) cur = h.id; else break;
      }
      setActiveId(cur);
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

  const posts = window.SITE.posts;
  const idx = posts.findIndex((p) => p.slug === slug);
  const older = posts[(idx + 1) % posts.length];
  const newer = posts[(idx - 1 + posts.length) % posts.length];

  return (
    <div className="page">
      <div style={{ position: "fixed", left: 0, top: 0, height: 2, width: progress + "%", background: "var(--accent)", zIndex: 999, transition: "width 80ms linear" }}></div>

      <div style={{ marginBottom: "var(--gap-4)" }}>
        <a className="link muted" href="#/blog" onClick={(e) => { e.preventDefault(); onNavigate("/blog"); }}>← writing</a>
      </div>

      <header style={{ marginBottom: "var(--gap-6)" }}>
        <div className="eyebrow" style={{ marginBottom: "var(--gap-3)" }}>
          {formatDate(post.date)} · {post.readingTime || 1} min read
        </div>
        <h1 className="display" style={{ fontSize: "clamp(36px, 5.5vw, 60px)", marginBottom: "var(--gap-3)" }}>{post.title}</h1>
        <p className="muted" style={{ maxWidth: 680, fontSize: 18 }}>{post.summary}</p>
        <div className="row-h" style={{ gap: 6, marginTop: "var(--gap-3)" }}>
          {post.tags.map((t) => <span key={t} className="tag">{t}</span>)}
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 220px", gap: "var(--gap-6)" }} className="post-grid">
        <article ref={articleRef} className="prose">
          {bodyHtml === null
            ? <p style={{ color: "var(--fg-mute)", fontFamily: "var(--font-mono)", fontSize: 12 }}>Loading…</p>
            : <div dangerouslySetInnerHTML={{ __html: bodyHtml }} />
          }
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
                <a key={h.id} href={"#" + h.id}
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
        .prose h2{font-family:var(--font-display);font-size:28px;font-weight:var(--display-weight);letter-spacing:var(--display-tracking);margin:var(--gap-6) 0 var(--gap-3);scroll-margin-top:80px;}
        .prose p{margin:0 0 var(--gap-3);}
        .prose pre{font-family:var(--font-mono);font-size:13px;background:var(--bg-soft);border:1px solid var(--rule);border-radius:var(--radius);padding:var(--gap-3) var(--gap-4);overflow-x:auto;white-space:pre;line-height:1.55;margin:var(--gap-4) 0;}
        .prose code:not(pre code){font-family:var(--font-mono);font-size:.9em;background:var(--bg-soft);border:1px solid var(--rule);border-radius:3px;padding:1px 5px;}
        .prose ul,.prose ol{padding-left:1.4em;margin:0 0 var(--gap-3);}
        .prose li{margin-bottom:4px;}
        .prose blockquote{margin:var(--gap-4) 0;padding:var(--gap-3) var(--gap-4);border-left:2px solid var(--accent);color:var(--fg-soft);}
        .prose blockquote p{margin:0;}
        .pn-card{padding:var(--gap-4);border:1px solid var(--rule);border-radius:var(--radius);transition:transform var(--speed) var(--ease),border-color var(--speed) var(--ease);}
        .pn-card:hover{border-color:var(--fg);transform:translateY(-2px);}
        @media (max-width: 860px){.post-grid{grid-template-columns:1fr!important;}.post-aside{order:-1;}.toc{position:static!important;}}
        @media (max-width: 720px){.pn-grid{grid-template-columns:1fr!important;}}
      `}</style>
    </div>
  );
}

function slugify(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

Object.assign(window, { BlogIndexPage, BlogPostPage, slugify });
