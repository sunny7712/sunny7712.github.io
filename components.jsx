// Shared UI components: Nav, CursorHalo, CommandPalette, Footer.
// Globals consumed: window.SITE, window.useRoute (defined in app.jsx)

const { useState, useEffect, useRef, useMemo, useCallback } = React;

const NAV_ITEMS = [
  { id: "home", label: "Home", path: "/" },
  { id: "career", label: "Career", path: "/career" },
  { id: "projects", label: "Projects", path: "/projects" },
  { id: "blog", label: "Writing", path: "/blog" },
  { id: "now", label: "Now", path: "/now" },
  { id: "about", label: "About", path: "/about" },
  { id: "contact", label: "Contact", path: "/contact" }
];

// ─── Nav ──────────────────────────────────────────────
function Nav({ route, onNavigate, dark, onToggleDark, onOpenCmdK }) {
  const id = window.SITE.identity;
  return (
    <header className="nav">
      <a href="#/" className="nav-brand" onClick={(e) => { e.preventDefault(); onNavigate("/"); }}>
        <span className="dot"></span>
        {id.short}
      </a>
      <nav className="nav-links">
        {NAV_ITEMS.map((it) => {
          const active = (it.path === "/" && route.path === "/") ||
                         (it.path !== "/" && route.path.startsWith(it.path));
          return (
            <a key={it.id}
               href={"#" + it.path}
               className={"nav-link" + (active ? " active" : "")}
               onClick={(e) => { e.preventDefault(); onNavigate(it.path); }}>
              {it.label}
            </a>
          );
        })}
      </nav>
      <div className="nav-actions">
        <button className="icon-btn" title="Command palette  (⌘K)" onClick={onOpenCmdK} aria-label="Open command palette">
          <SearchIcon />
        </button>
        <button className="icon-btn" title="Toggle theme" onClick={onToggleDark} aria-label="Toggle dark mode">
          {dark ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
    </header>
  );
}

// ─── Footer ──────────────────────────────────────────────
function Footer({ onNavigate }) {
  const id = window.SITE.identity;
  return (
    <footer className="foot">
      <div>© {new Date().getFullYear()} {id.name} — built from scratch.</div>
      <div className="row-h" style={{ gap: 16 }}>
        {id.socials.slice(0, 3).map((s) => (
          <a key={s.label} className="link" href={s.url} target="_blank" rel="noreferrer">{s.label.toLowerCase()}</a>
        ))}
        <a className="link" href="#/contact" onClick={(e) => { e.preventDefault(); onNavigate("/contact"); }}>contact</a>
      </div>
    </footer>
  );
}

// ─── Cursor halo (very subtle) ──────────────────────────────────────────────
function CursorHalo({ enabled }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!enabled) return;
    const el = ref.current; if (!el) return;
    let x = window.innerWidth / 2, y = window.innerHeight / 2;
    let tx = x, ty = y, raf;
    const onMove = (e) => { tx = e.clientX; ty = e.clientY; };
    const tick = () => {
      x += (tx - x) * 0.12;
      y += (ty - y) * 0.12;
      el.style.left = x + "px";
      el.style.top = y + "px";
      raf = requestAnimationFrame(tick);
    };
    window.addEventListener("mousemove", onMove);
    raf = requestAnimationFrame(tick);
    return () => { window.removeEventListener("mousemove", onMove); cancelAnimationFrame(raf); };
  }, [enabled]);
  if (!enabled) return null;
  return <div className="cursor-halo" ref={ref}></div>;
}

// ─── Command Palette ──────────────────────────────────────────────
function CommandPalette({ open, onClose, onNavigate, theme, onTheme, dark, onToggleDark }) {
  const [q, setQ] = useState("");
  const [idx, setIdx] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => { if (open) { setQ(""); setIdx(0); setTimeout(() => inputRef.current?.focus(), 60); } }, [open]);

  const commands = useMemo(() => {
    const nav = NAV_ITEMS.map((it) => ({
      group: "Navigate", label: it.label, hint: it.path,
      run: () => onNavigate(it.path), icon: <ArrowIcon />
    }));
    const projects = window.SITE.projects.map((p) => ({
      group: "Projects", label: p.title, hint: p.year,
      run: () => onNavigate("/projects/" + p.id), icon: <BoxIcon />
    }));
    const posts = window.SITE.posts.map((p) => ({
      group: "Writing", label: p.title, hint: p.date,
      run: () => onNavigate("/blog/" + p.slug), icon: <DocIcon />
    }));
    const actions = [
      { group: "Theme", label: "Editorial (serif)", hint: "A", run: () => onTheme("editorial"), icon: <DotIcon /> },
      { group: "Theme", label: "Swiss (mono)",     hint: "B", run: () => onTheme("swiss"),     icon: <DotIcon /> },
      { group: "Theme", label: "Warm (soft sans)", hint: "C", run: () => onTheme("warm"),      icon: <DotIcon /> },
      { group: "Theme", label: dark ? "Switch to light" : "Switch to dark", hint: "D", run: onToggleDark, icon: dark ? <SunIcon /> : <MoonIcon /> },
    ];
    return [...nav, ...actions, ...projects, ...posts];
  }, [onNavigate, onTheme, dark, onToggleDark]);

  const filtered = useMemo(() => {
    if (!q.trim()) return commands;
    const t = q.toLowerCase();
    return commands.filter((c) => (c.label + " " + c.hint).toLowerCase().includes(t));
  }, [q, commands]);

  // group filtered by group
  const grouped = useMemo(() => {
    const m = new Map();
    filtered.forEach((c) => {
      if (!m.has(c.group)) m.set(c.group, []);
      m.get(c.group).push(c);
    });
    return [...m.entries()];
  }, [filtered]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") { e.preventDefault(); setIdx((i) => Math.min(i + 1, filtered.length - 1)); }
      if (e.key === "ArrowUp")   { e.preventDefault(); setIdx((i) => Math.max(i - 1, 0)); }
      if (e.key === "Enter") {
        const item = filtered[idx];
        if (item) { item.run(); onClose(); }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, filtered, idx, onClose]);

  if (!open) return null;

  let counter = -1;
  return (
    <>
      <div className="cmdk-backdrop" onClick={onClose}></div>
      <div className="cmdk" role="dialog" aria-modal="true">
        <input
          ref={inputRef}
          className="cmdk-input"
          placeholder="Search anything — projects, posts, pages, settings…"
          value={q}
          onChange={(e) => { setQ(e.target.value); setIdx(0); }}
        />
        <div className="cmdk-list">
          {grouped.length === 0 && (
            <div className="cmdk-section" style={{ paddingTop: 16, paddingBottom: 16 }}>No results.</div>
          )}
          {grouped.map(([group, items]) => (
            <div key={group}>
              <div className="cmdk-section">{group}</div>
              {items.map((c) => {
                counter += 1;
                const i = counter;
                return (
                  <div key={c.label + i}
                       className={"cmdk-item" + (i === idx ? " sel" : "")}
                       onMouseEnter={() => setIdx(i)}
                       onClick={() => { c.run(); onClose(); }}>
                    <span className="ic">{c.icon}</span>
                    <span className="nm">{c.label}</span>
                    <span className="hint">{c.hint}</span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

// ─── Icons ──────────────────────────────────────────────
const ic = { width: 16, height: 16, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round", strokeLinejoin: "round" };
const SearchIcon = () => <svg {...ic}><circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path></svg>;
const MoonIcon   = () => <svg {...ic}><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"></path></svg>;
const SunIcon    = () => <svg {...ic}><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path></svg>;
const ArrowIcon  = () => <svg {...ic}><path d="M5 12h14M13 5l7 7-7 7"></path></svg>;
const BoxIcon    = () => <svg {...ic}><path d="M3 7l9-4 9 4-9 4-9-4z"></path><path d="M3 7v10l9 4 9-4V7"></path></svg>;
const DocIcon    = () => <svg {...ic}><path d="M6 3h9l5 5v13H6z"></path><path d="M15 3v5h5M9 13h7M9 17h7M9 9h3"></path></svg>;
const DotIcon    = () => <svg {...ic}><circle cx="12" cy="12" r="4"></circle></svg>;

// ─── reusable bits ──────────────────────────────────────────────
function MetaBlock({ items }) {
  return (
    <div className="row-h" style={{ gap: 18, fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--fg-mute)", letterSpacing: "var(--label-tracking)", textTransform: "var(--label-case)" }}>
      {items.map((it, i) => (
        <span key={i}>{it[0]} <span style={{ color: "var(--fg)" }}>{it[1]}</span></span>
      ))}
    </div>
  );
}

function StackChips({ items }) {
  return (
    <div className="row-h" style={{ gap: 6 }}>
      {items.map((s) => <span key={s} className="tag">{s}</span>)}
    </div>
  );
}

// expose to other babel scripts
Object.assign(window, {
  Nav, Footer, CursorHalo, CommandPalette, NAV_ITEMS,
  MetaBlock, StackChips,
  SearchIcon, MoonIcon, SunIcon, ArrowIcon, BoxIcon, DocIcon, DotIcon
});
