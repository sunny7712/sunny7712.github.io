// Main pages: Home, Career, Now, About, Contact, Themes (canvas).
// Project pages live in pages-projects.jsx, blog in pages-blog.jsx.

const { useState: useState_p, useEffect: useEffect_p, useMemo: useMemo_p } = React;

// ─── HOME ──────────────────────────────────────────────────
function HomePage({ onNavigate }) {
  const id = window.SITE.identity;
  const projects = window.SITE.projects.filter((p) => p.featured).slice(0, 3);
  const posts = window.SITE.posts.slice(0, 3);

  return (
    <div className="page">
      <section className="hero" style={{ marginBottom: "var(--gap-8)" }}>
        <div className="eyebrow" style={{ marginBottom: "var(--gap-4)" }}>
          <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", marginRight: 8, transform: "translateY(-1px)" }}></span>
          Available for interesting conversations · {id.location}
        </div>
        <h1 className="display" style={{ fontSize: "clamp(54px, 8vw, 104px)", marginBottom: "var(--gap-4)" }}>
          {id.name}.<br />
          <span style={{ color: "var(--fg-mute)" }}>{id.role.toLowerCase()}.</span>
        </h1>
        <p style={{ maxWidth: 620, fontSize: "calc(var(--body-size) + 2px)", color: "var(--fg-soft)", marginTop: "var(--gap-4)" }}>
          {id.tagline}
        </p>
        <div className="row-h" style={{ marginTop: "var(--gap-5)", gap: 18 }}>
          <a href="#/projects" className="link" onClick={(e) => { e.preventDefault(); onNavigate("/projects"); }}>See projects →</a>
          <a href="#/career" className="link muted" onClick={(e) => { e.preventDefault(); onNavigate("/career"); }}>Read career</a>
          <a href={id.socials[0].url} className="link muted" target="_blank" rel="noreferrer">GitHub ↗</a>
        </div>
      </section>

      {/* Selected work */}
      <section style={{ marginBottom: "var(--gap-7)" }}>
        <div className="sec-head">
          <h2>Selected work</h2>
          <span className="count">— {window.SITE.projects.length} total</span>
        </div>
        {projects.map((p, i) => (
          <a key={p.id} href={"#/projects/" + p.id}
             className="row"
             onClick={(e) => { e.preventDefault(); onNavigate("/projects/" + p.id); }}>
            <span className="row-meta">{p.year} · {p.role}</span>
            <span>
              <span className="row-title">{p.title}</span>
              <div className="muted" style={{ marginTop: 6, fontSize: 14, maxWidth: 560 }}>{p.blurb}</div>
            </span>
            <span className="row-aside">{p.stack.slice(0, 2).join(" · ")}</span>
          </a>
        ))}
        <div style={{ marginTop: "var(--gap-4)" }}>
          <a className="link" href="#/projects" onClick={(e) => { e.preventDefault(); onNavigate("/projects"); }}>all projects →</a>
        </div>
      </section>

      {/* Recent writing */}
      <section>
        <div className="sec-head">
          <h2>Recent writing</h2>
          <span className="count">— {window.SITE.posts.length} posts</span>
        </div>
        {posts.map((p) => (
          <a key={p.slug} href={"#/blog/" + p.slug}
             className="row"
             onClick={(e) => { e.preventDefault(); onNavigate("/blog/" + p.slug); }}>
            <span className="row-meta">{formatDate(p.date)}</span>
            <span>
              <span className="row-title">{p.title}</span>
              <div className="muted" style={{ marginTop: 6, fontSize: 14, maxWidth: 560 }}>{p.summary}</div>
            </span>
            <span className="row-aside">{readingTime(p)} min</span>
          </a>
        ))}
        <div style={{ marginTop: "var(--gap-4)" }}>
          <a className="link" href="#/blog" onClick={(e) => { e.preventDefault(); onNavigate("/blog"); }}>all writing →</a>
        </div>
      </section>
    </div>
  );
}

// ─── CAREER ──────────────────────────────────────────────────
function CareerPage({ onNavigate }) {
  const exp = window.SITE.experience;
  const edu = window.SITE.education;
  const skills = window.SITE.skills;
  return (
    <div className="page">
      <PageHeader
        eyebrow="Career"
        title="Where I've worked, what I've shipped."
        sub="Three companies, two internships, one full-time role, and a habit of leaving things better than I found them."
      />

      <div className="stack-loose" style={{ marginTop: "var(--gap-7)" }}>
        {exp.map((e, i) => (
          <article key={i} style={{ paddingTop: "var(--gap-4)", borderTop: i === 0 ? "0" : "1px solid var(--rule)" }}>
            <header style={{ display: "grid", gridTemplateColumns: "1fr auto", alignItems: "baseline", gap: 16, marginBottom: 6 }}>
              <div>
                <span className="display" style={{ fontSize: 30 }}>{e.company}</span>
                <span style={{ marginLeft: 12, color: "var(--fg-mute)", fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: "var(--label-tracking)" }}>
                  {e.role}
                </span>
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-mute)", letterSpacing: "var(--label-tracking)" }}>
                {e.period}
              </div>
            </header>
            <p className="muted" style={{ maxWidth: 720, marginTop: 4, marginBottom: "var(--gap-4)" }}>
              {e.blurb}
            </p>
            <ol style={{ margin: 0, padding: 0, listStyle: "none" }}>
              {e.bullets.map((b, j) => (
                <li key={j} style={{ display: "grid", gridTemplateColumns: "28px 1fr", gap: 12, padding: "var(--gap-2) 0", maxWidth: 760 }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-mute)", paddingTop: 4 }}>0{j+1}</span>
                  <div>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>{b.title}</div>
                    <div className="muted" style={{ fontSize: 15 }}>{b.body}</div>
                  </div>
                </li>
              ))}
            </ol>
            <div className="row-h" style={{ marginTop: "var(--gap-4)", gap: 6 }}>
              {e.stack.map((s) => <span key={s} className="tag">{s}</span>)}
            </div>
          </article>
        ))}
      </div>

      {/* Education */}
      <section style={{ marginTop: "var(--gap-8)" }}>
        <div className="sec-head"><h2>Education</h2><span className="count">— {edu.period}</span></div>
        <div className="row" style={{ borderTop: 0 }}>
          <span className="row-meta">2021 — 2025</span>
          <span>
            <span className="row-title">{edu.school}</span>
            <div className="muted" style={{ marginTop: 6, fontSize: 14 }}>{edu.degree}</div>
          </span>
          <span className="row-aside">{edu.detail}</span>
        </div>
      </section>

      {/* Skills */}
      <section style={{ marginTop: "var(--gap-7)" }}>
        <div className="sec-head"><h2>Stack</h2><span className="count">— things I reach for</span></div>
        <div className="stack" style={{ marginTop: "var(--gap-4)" }}>
          {Object.entries(skills).map(([cat, arr]) => (
            <div key={cat} style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: 24, padding: "var(--gap-3) 0", borderTop: "1px solid var(--rule)" }}>
              <span className="label">{cat}</span>
              <div className="row-h" style={{ gap: 6 }}>
                {arr.map((s) => <span key={s} className="tag">{s}</span>)}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// ─── NOW ──────────────────────────────────────────────────
function NowPage() {
  const now = window.SITE.now;
  return (
    <div className="page">
      <PageHeader
        eyebrow={"Now · updated " + now.updated}
        title="What I'm up to, right now."
        sub="Inspired by Derek Sivers's /now page. A snapshot rather than a CV — what's on my desk this month, what I'm trying to learn, what I'm deliberately ignoring."
      />
      <div className="stack-loose" style={{ marginTop: "var(--gap-7)" }}>
        {now.sections.map((s, i) => (
          <section key={s.title}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 16, marginBottom: "var(--gap-3)" }}>
              <span className="label">0{i+1}</span>
              <h2 className="display" style={{ fontSize: 28, margin: 0 }}>{s.title}</h2>
            </div>
            <ul style={{ margin: 0, padding: 0, listStyle: "none", maxWidth: 720 }}>
              {s.items.map((it, j) => (
                <li key={j} style={{ display: "grid", gridTemplateColumns: "16px 1fr", gap: 12, padding: "var(--gap-2) 0", borderTop: j === 0 ? "0" : "1px solid var(--rule)" }}>
                  <span style={{ color: "var(--accent)", paddingTop: 4, fontFamily: "var(--font-mono)" }}>—</span>
                  <span>{it}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}

// ─── ABOUT ──────────────────────────────────────────────────
function AboutPage({ onNavigate }) {
  const id = window.SITE.identity;
  return (
    <div className="page">
      <PageHeader
        eyebrow="About"
        title={"Hi, I'm Vamsi."}
        sub={id.intro}
      />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "var(--gap-6)", marginTop: "var(--gap-7)" }} className="about-grid">
        <article className="prose" style={{ maxWidth: "none" }}>
          <h2>How I work</h2>
          <p>To be honest, I'm just trying to really figure out how to be a good engineer in these times of AI. </p>
          <p>I try to find the smallest thing that could possibly work and ship it. Then I try to find the smallest improvement that could possibly help, and ship that too. It's not a thrilling philosophy but it has a remarkably good track record.</p>
          <p>Most of my favorite engineering moments have come from <em>removing</em> things — a service, a cache layer, or simplifying something overly verbose written by AI. Subtraction is usually faster than addition, and almost always cheaper to operate.</p>
          <h2>What I'm curious about</h2>
          <p>I'm curious about infra. I want to learn about networking, containerization, and distributed systems.</p>
          <h2>Outside work</h2>
          <p>Biryani, Gym, and going on Runs. I watch more films than I should and read fewer books than I'd like.</p>
        </article>
        <aside className="stack" style={{ position: "sticky", top: 24, alignSelf: "start" }}>
          <div className="media" style={{ aspectRatio: "4 / 5" }}>portrait · drop-in</div>
          <div className="stack-tight">
            <div className="label">Currently</div>
            <div>SDE at <a className="link" href="#/career" onClick={(e) => { e.preventDefault(); onNavigate("/career"); }}>Groww</a></div>
          </div>
          <div className="stack-tight">
            <div className="label">Based in</div>
            <div>{id.location}</div>
          </div>
          <div className="stack-tight">
            <div className="label">Reach me</div>
            <div className="stack-tight">
              {id.socials.map((s) => (
                <a key={s.label} className="link muted" href={s.url} target="_blank" rel="noreferrer">{s.label.toLowerCase()} / {s.handle}</a>
              ))}
            </div>
          </div>
        </aside>
      </div>
      <style>{`@media (max-width: 720px){.about-grid{grid-template-columns:1fr!important;}}`}</style>
    </div>
  );
}

// ─── CONTACT ──────────────────────────────────────────────────
function ContactPage() {
  const id = window.SITE.identity;
  const [copied, setCopied] = useState_p(false);
  const [form, setForm] = useState_p({ name: "", email: "", message: "" });
  const [sent, setSent] = useState_p(false);

  const copy = async () => {
    try { await navigator.clipboard.writeText(id.email); setCopied(true); setTimeout(() => setCopied(false), 1600); } catch {}
  };

  const submit = (e) => {
    e.preventDefault();
    if (!form.email || !form.message) return;
    setSent(true);
    // open mailto as a fallback
    const body = encodeURIComponent(form.message + "\n\n— " + (form.name || "Anonymous"));
    const subject = encodeURIComponent("hello from your site");
    window.location.href = `mailto:${id.email}?subject=${subject}&body=${body}`;
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow="Contact"
        title="Drop me a line."
        sub="Best for: interesting problems, book recommendations. Reasonable response time within a working week."
      />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: "var(--gap-6)", marginTop: "var(--gap-7)" }} className="contact-grid">
        <form className="stack" onSubmit={submit}>
          <Field label="Your name" id="name">
            <input id="name" className="inp" placeholder="Ada Lovelace" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </Field>
          <Field label="Email" id="email" required>
            <input id="email" type="email" required className="inp" placeholder="you@somewhere.com" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          </Field>
          <Field label="Message" id="msg" required>
            <textarea id="msg" required className="inp" rows="6" placeholder="What's on your mind?" value={form.message} onChange={(e) => setForm({ ...form, message: e.target.value })}></textarea>
          </Field>
          <div className="row-h" style={{ justifyContent: "space-between" }}>
            <button type="submit" className="btn-primary">{sent ? "Opening mail client…" : "Send →"}</button>
            <span className="muted" style={{ fontSize: 12 }}>This opens your email client with the message pre-filled.</span>
          </div>
        </form>

        <aside className="stack">
          <div>
            <div className="label" style={{ marginBottom: 8 }}>Email</div>
            <button onClick={copy} className="link" style={{ fontFamily: "var(--font-mono)", fontSize: 14 }}>
              {id.email} <span style={{ marginLeft: 8, color: "var(--accent)" }}>{copied ? "copied ✓" : "copy"}</span>
            </button>
          </div>
          <div>
            <div className="label" style={{ marginBottom: 8 }}>Elsewhere</div>
            <div className="stack-tight">
              {id.socials.map((s) => (
                <a key={s.label} className="link" href={s.url} target="_blank" rel="noreferrer" style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}>
                  {s.label.toLowerCase().padEnd(10, " ")} <span className="muted">/ {s.handle}</span>
                </a>
              ))}
            </div>
          </div>
          <div style={{ marginTop: "var(--gap-3)" }}>
            <div className="label" style={{ marginBottom: 8 }}>Time</div>
            <Clock />
          </div>
        </aside>
      </div>
      <style>{`
        .inp{
          width:100%;padding:12px 14px;background:var(--surface);
          border:1px solid var(--rule);border-radius:var(--radius);
          font:inherit;color:var(--fg);outline:none;
          transition:border-color var(--speed) var(--ease), background var(--speed) var(--ease);
        }
        .inp:focus{border-color:var(--accent);background:var(--bg);}
        textarea.inp{resize:vertical;min-height:140px;font-family:var(--font-body);}
        .btn-primary{
          padding:11px 22px;background:var(--fg);color:var(--bg);
          border-radius:var(--radius);font-family:var(--font-ui);font-size:13px;
          letter-spacing:.01em;transition:transform var(--speed) var(--ease), opacity var(--speed) var(--ease);
        }
        .btn-primary:hover{transform:translateY(-1px);opacity:.9;}
        @media (max-width: 720px){.contact-grid{grid-template-columns:1fr!important;}}
      `}</style>
    </div>
  );
}

function Field({ label, id, required, children }) {
  return (
    <label htmlFor={id} className="stack-tight">
      <span className="label">{label}{required ? " *" : ""}</span>
      {children}
    </label>
  );
}

function Clock() {
  const [t, setT] = useState_p(() => new Date());
  useEffect_p(() => { const i = setInterval(() => setT(new Date()), 1000); return () => clearInterval(i); }, []);
  const ist = new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: "Asia/Kolkata" }).format(t);
  return <div style={{ fontFamily: "var(--font-mono)", fontSize: 14 }}>{ist} <span className="muted" style={{ fontSize: 11 }}>IST · {window.SITE.identity.location}</span></div>;
}

// ─── THEMES (canvas) — preview all 3 directions side by side ──────────────
function ThemesPage({ onNavigate, onTheme, theme }) {
  const directions = [
    { id: "editorial", title: "Editorial", subtitle: "Serif-driven, magazine.", body: "Big display serifs, generous whitespace, warm off-white. For when you want pages to feel like printed essays." },
    { id: "swiss",     title: "Swiss",     subtitle: "Monospace, gridlines.",   body: "JetBrains Mono everywhere, technical labels, electric blue. For when precision matters more than warmth." },
    { id: "warm",      title: "Warm",      subtitle: "Soft sans, cream.",        body: "Round letterforms, cream cards, amber accents. For when you want the page to feel like a cup of coffee." }
  ];
  return (
    <div className="page">
      <PageHeader eyebrow="Theme directions" title="Three ways this site can dress." sub="Each direction is a complete swap of typography, color, and rhythm. Pick one and the entire site reorganizes. You can also toggle via ⌘K." />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "var(--gap-4)", marginTop: "var(--gap-6)" }} className="themes-grid">
        {directions.map((d) => (
          <button key={d.id}
                  onClick={() => onTheme(d.id)}
                  className="theme-card"
                  style={{
                    textAlign: "left", padding: "var(--gap-4)",
                    border: "1px solid " + (theme === d.id ? "var(--accent)" : "var(--rule)"),
                    background: "var(--surface)", borderRadius: "var(--radius)",
                    transition: "transform var(--speed) var(--ease), border-color var(--speed) var(--ease)",
                    cursor: "pointer"
                  }}>
            <div className="label" style={{ marginBottom: 12 }}>{d.id === theme ? "● Current" : "○ Try"}</div>
            <div className="display" style={{ fontSize: 32, marginBottom: 4 }}>{d.title}</div>
            <div className="muted" style={{ fontSize: 13, marginBottom: 12 }}>{d.subtitle}</div>
            <div style={{ fontSize: 14 }}>{d.body}</div>
          </button>
        ))}
      </div>
      <style>{`
        .theme-card:hover{transform:translateY(-2px);}
        @media (max-width: 800px){.themes-grid{grid-template-columns:1fr!important;}}
      `}</style>
      <div style={{ marginTop: "var(--gap-5)" }}>
        <a className="link" href="#/" onClick={(e) => { e.preventDefault(); onNavigate("/"); }}>← back to home</a>
      </div>
    </div>
  );
}

// ─── Shared: PageHeader ──────────────────────────────────────────────
function PageHeader({ eyebrow, title, sub }) {
  return (
    <header>
      <div className="eyebrow" style={{ marginBottom: "var(--gap-3)" }}>{eyebrow}</div>
      <h1 className="display" style={{ fontSize: "clamp(40px, 6vw, 64px)", maxWidth: 880 }}>{title}</h1>
      {sub && <p className="muted" style={{ maxWidth: 640, marginTop: "var(--gap-3)", fontSize: 17 }}>{sub}</p>}
    </header>
  );
}

// ─── helpers ──────────────────────────────────────────────
function formatDate(d) {
  try {
    const dt = new Date(d);
    return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" }).format(dt);
  } catch { return d; }
}
function readingTime(p) {
  if (p.readingTime != null) return p.readingTime;
  return 1;
}

Object.assign(window, {
  HomePage, CareerPage, NowPage, AboutPage, ContactPage, ThemesPage,
  PageHeader, formatDate, readingTime
});
