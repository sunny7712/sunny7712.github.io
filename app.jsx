// Main app: hash routing, theme/dark state, Tweaks integration.

const { useState: useState_a, useEffect: useEffect_a, useCallback: useCallback_a } = React;

// ─── tweak defaults (host rewrites this on disk) ────────────────────────────
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "editorial",
  "dark": false,
  "accentEditorial": "#c46a3c",
  "accentSwiss": "#3066ec",
  "accentWarm": "#c89146",
  "fontSize": 17,
  "density": "regular",
  "cursorHalo": true,
  "showThemes": true
}/*EDITMODE-END*/;

// ─── parse hash ───────────────────────────────────────────────────────────
function parseHash() {
  const h = (window.location.hash || "#/").replace(/^#/, "");
  const path = h || "/";
  return { path };
}

function App() {
  const [route, setRoute] = useState_a(parseHash);
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [cmdkOpen, setCmdkOpen] = useState_a(false);

  // sync hash → state
  useEffect_a(() => {
    const onHash = () => { setRoute(parseHash()); window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" }); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = useCallback_a((p) => {
    window.location.hash = "#" + p;
  }, []);

  // apply theme & dark to <html>
  useEffect_a(() => {
    document.documentElement.setAttribute("data-theme", t.theme);
    document.documentElement.setAttribute("data-dark", String(!!t.dark));
  }, [t.theme, t.dark]);

  // apply accent + body size
  useEffect_a(() => {
    const accentKey = t.theme === "swiss" ? "accentSwiss" : t.theme === "warm" ? "accentWarm" : "accentEditorial";
    document.documentElement.style.setProperty("--accent", t[accentKey]);
  }, [t.theme, t.accentEditorial, t.accentSwiss, t.accentWarm]);

  useEffect_a(() => {
    document.documentElement.style.setProperty("--body-size", t.fontSize + "px");
    const d = t.density === "compact" ? 0.85 : t.density === "spacious" ? 1.15 : 1;
    document.documentElement.style.setProperty("--gap-1", (8 * d) + "px");
    document.documentElement.style.setProperty("--gap-2", (12 * d) + "px");
    document.documentElement.style.setProperty("--gap-3", (16 * d) + "px");
    document.documentElement.style.setProperty("--gap-4", (24 * d) + "px");
    document.documentElement.style.setProperty("--gap-5", (32 * d) + "px");
    document.documentElement.style.setProperty("--gap-6", (48 * d) + "px");
    document.documentElement.style.setProperty("--gap-7", (72 * d) + "px");
    document.documentElement.style.setProperty("--gap-8", (112 * d) + "px");
  }, [t.fontSize, t.density]);

  // cmd-K shortcut
  useEffect_a(() => {
    const onKey = (e) => {
      const isMod = e.metaKey || e.ctrlKey;
      if (isMod && (e.key === "k" || e.key === "K")) { e.preventDefault(); setCmdkOpen((o) => !o); }
      // ⌘/ to flip theme quickly
      if (isMod && e.key === "/") {
        e.preventDefault();
        const order = ["editorial", "swiss", "warm"];
        const next = order[(order.indexOf(t.theme) + 1) % order.length];
        setTweak("theme", next);
      }
      if (isMod && (e.key === "j" || e.key === "J")) { e.preventDefault(); setTweak("dark", !t.dark); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [t.theme, t.dark, setTweak]);

  // route resolution
  let view;
  const p = route.path;
  if (p === "/" || p === "") view = <HomePage onNavigate={navigate} />;
  else if (p === "/career") view = <CareerPage onNavigate={navigate} />;
  else if (p === "/projects") view = <ProjectsPage onNavigate={navigate} />;
  else if (p.startsWith("/projects/")) view = <ProjectDetailPage id={p.split("/")[2]} onNavigate={navigate} />;
  else if (p === "/blog") view = <BlogIndexPage onNavigate={navigate} />;
  else if (p.startsWith("/blog/")) view = <BlogPostPage slug={p.split("/")[2]} onNavigate={navigate} />;
  else if (p === "/now") view = <NowPage />;
  else if (p === "/about") view = <AboutPage onNavigate={navigate} />;
  else if (p === "/contact") view = <ContactPage />;
  else if (p === "/themes") view = <ThemesPage onNavigate={navigate} onTheme={(v) => setTweak("theme", v)} theme={t.theme} />;
  else view = <NotFoundPage onNavigate={navigate} />;

  return (
    <>
      <CursorHalo enabled={t.cursorHalo} />
      <div className="site" key={route.path /* trigger page transition */}>
        <Nav route={route} onNavigate={navigate}
             dark={t.dark} onToggleDark={() => setTweak("dark", !t.dark)}
             onOpenCmdK={() => setCmdkOpen(true)} />
        {view}
        <Footer onNavigate={navigate} />
      </div>

      <CommandPalette
        open={cmdkOpen}
        onClose={() => setCmdkOpen(false)}
        onNavigate={navigate}
        theme={t.theme}
        onTheme={(v) => setTweak("theme", v)}
        dark={t.dark}
        onToggleDark={() => setTweak("dark", !t.dark)}
      />

      <TweaksPanel title="Tweaks">
        <TweakSection label="Direction" />
        <TweakRadio label="Theme"
          value={t.theme}
          options={["editorial", "swiss", "warm"]}
          onChange={(v) => setTweak("theme", v)} />
        <TweakToggle label="Dark mode" value={t.dark} onChange={(v) => setTweak("dark", v)} />

        <TweakSection label="Accent" />
        {t.theme === "editorial" && (
          <TweakColor label="Editorial accent" value={t.accentEditorial}
            options={["#c46a3c", "#7a5a3e", "#9b3a3a", "#4a6b3a"]}
            onChange={(v) => setTweak("accentEditorial", v)} />
        )}
        {t.theme === "swiss" && (
          <TweakColor label="Swiss accent" value={t.accentSwiss}
            options={["#3066ec", "#ff5e3a", "#10b981", "#0a0a0a"]}
            onChange={(v) => setTweak("accentSwiss", v)} />
        )}
        {t.theme === "warm" && (
          <TweakColor label="Warm accent" value={t.accentWarm}
            options={["#c89146", "#a14d3a", "#4a6b3a", "#7a4f8c"]}
            onChange={(v) => setTweak("accentWarm", v)} />
        )}

        <TweakSection label="Layout" />
        <TweakSlider label="Body size" value={t.fontSize} min={14} max={20} unit="px"
          onChange={(v) => setTweak("fontSize", v)} />
        <TweakRadio label="Density" value={t.density}
          options={["compact", "regular", "spacious"]}
          onChange={(v) => setTweak("density", v)} />

        <TweakSection label="Flourishes" />
        <TweakToggle label="Cursor halo" value={t.cursorHalo} onChange={(v) => setTweak("cursorHalo", v)} />
        <TweakButton label="Compare 3 themes →" onClick={() => navigate("/themes")} />

        <TweakSection label="Keyboard" />
        <div style={{ fontFamily: "var(--font-mono, ui-monospace)", fontSize: 10.5, color: "rgba(0,0,0,.55)", lineHeight: 1.7 }}>
          ⌘K — command palette<br />
          ⌘/ — cycle theme<br />
          ⌘J — toggle dark
        </div>
      </TweaksPanel>
    </>
  );
}

function NotFoundPage({ onNavigate }) {
  return (
    <div className="page">
      <PageHeader eyebrow="404" title="That page is hiding." sub="Try the home page, or hit ⌘K to find what you're looking for." />
      <a className="link" href="#/" onClick={(e) => { e.preventDefault(); onNavigate("/"); }}>← home</a>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
