"""Astrodrift: type a spot in the sky, see what changes between NASA SPHEREx pictures."""
import hashlib
import time
from datetime import date

import numpy as np
import streamlit as st
from astropy.coordinates import SkyCoord
from astropy.time import Time

import spherex_cache as cache
import spherex_fetch
import spherex_follow as follow
import spherex_plot as splot
import spherex_process as sp
import spherex_stack as stk

RUN_START = time.perf_counter()  # for the "full script run" time printed at the very end

st.set_page_config(page_title="Astrodrift", layout="wide")

# ---------- look and feel: one function, one block of CSS (the language that styles web pages) ----------
# Near-black page with a soft blue glow, a thin starfield, one blue accent (#4DA6FF), pill buttons and cards.
# Nothing here changes what the app does: it only changes how it looks.
def starfield(n=70):
    """A CSS box-shadow list that draws n tiny white dots at fixed random places (the same ones every time)."""
    rng = np.random.default_rng(7)
    return ",".join(f"{rng.uniform(0, 100):.1f}vw {rng.uniform(0, 100):.1f}vh 0 0 rgba(255,255,255,{rng.uniform(.15, .5):.2f})"
                    for _ in range(n))


def apply_theme():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --blue: #4DA6FF; --body: #d6d9e2; --muted: #8b92a3; --line: rgba(255,255,255,.12); }

/* ----- page: near-black, a blue glow top right, a fainter one bottom left, and a thin starfield ----- */
.stApp { background: radial-gradient(ellipse 60% 50% at 100% 0%, rgba(77,166,255,.22), transparent 70%),
                     radial-gradient(ellipse 50% 40% at 0% 100%, rgba(77,166,255,.10), transparent 70%), #04060b; color: var(--body); }
.stApp::before { content: ''; position: fixed; top: 0; left: 0; width: 1px; height: 1px; pointer-events: none;
                 z-index: 0; box-shadow: STARS; }
[data-testid="stHeader"] { background: transparent; }
/* the slim blue rail on the left edge of the main area (a background, so it needs no extra element) */
[data-testid="stMain"] { background: linear-gradient(180deg, #4DA6FF, transparent) left top / 8px 100% no-repeat; }
.block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 1400px; }
[data-testid="stVerticalBlock"] { gap: 1rem; }

/* ----- fonts: Inter for text, Space Grotesk for headings, JetBrains Mono for numbers ----- */
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.stApp textarea, .stApp [data-baseweb="select"] { font-family: 'Inter', sans-serif; }
.stApp p, .stApp li { color: var(--body); }
.stApp h1, .stApp h2, .stApp h3, .big-title { font-family: 'Space Grotesk', sans-serif; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff, #b9dcff); -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent; }
.stApp input, [data-testid="stMetricValue"], code, .mono { font-family: 'JetBrains Mono', monospace; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
[data-testid="stWidgetLabel"] p { color: var(--body); }

/* ----- the small blue kicker above each view, with a short blue line before it ----- */
.kicker { display: flex; align-items: center; gap: .7rem; font-size: 12px; font-weight: 600; letter-spacing: .2em;
          text-transform: uppercase; color: var(--blue); margin: .4rem 0 .2rem 0; }
.kicker::before { content: ''; width: 30px; height: 2px; background: var(--blue); }

/* ----- cards: the right-hand info panels, My finds, the hero and the detection card ----- */
[class*="st-key-card_"], .card { position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 22px;
    background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); padding: 1.4rem 1.3rem 1.2rem 1.3rem; }
[class*="st-key-card_"]::before, .card::before { content: ''; position: absolute; left: 0; right: 0; top: 0; height: 4px;
    background: linear-gradient(90deg, var(--blue), transparent); }

/* ----- big numbers, like the "Not detected" card in the slides ----- */
.stats { display: flex; flex-wrap: wrap; gap: .4rem 2rem; margin: .2rem 0 .6rem 0; }
.stat .num { font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 44px; line-height: 1; color: #fff; }
.stat .lab { font-size: 12px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--blue); margin-top: .3rem; }
.stat.small .num { font-size: 30px; }

/* ----- pictures: rounded, thin blue border, soft blue glow ----- */
[data-testid="stPlotlyChart"] { border-radius: 18px; border: 1px solid rgba(77,166,255,.4); overflow: hidden;
                                box-shadow: 0 0 60px rgba(77,166,255,.2); }
.st-key-plot_flip, .st-key-plot_diff, .st-key-plot_layers, .st-key-plot_stack, .st-key-plot_colour,
.st-key-plot_follow_flip, .st-key-plot_follow_stack { max-width: 820px; }

/* ----- tabs: pills. Active = solid blue with dark text, others = outlined with blue-tinted text ----- */
[role="tablist"] { gap: .5rem; border-bottom: none; box-shadow: none; flex-wrap: wrap; }
div:has(> [role="tablist"]) { border-bottom: none; box-shadow: none; }
[role="tablist"]::before, [role="tablist"]::after, div:has(> [role="tablist"])::before, div:has(> [role="tablist"])::after,
[data-testid="stTabs"] > div::after { display: none !important; content: none !important; }
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid rgba(77,166,255,.35); background: rgba(77,166,255,.08); }
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTab"], button[data-baseweb="tab"] { border: 1px solid rgba(77,166,255,.5); border-radius: 999px; background: transparent;
    padding: .35rem 1.1rem; height: auto; }
[data-testid="stTab"] p, button[data-baseweb="tab"] p { color: #9cc9f5; font-weight: 600; }
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] { background: var(--blue); border-color: var(--blue); }
[data-testid="stTab"][aria-selected="true"] p, button[data-baseweb="tab"][aria-selected="true"] p { color: #04060b; }

/* ----- buttons: pills. Primary = solid blue with dark bold text. Secondary = outlined ----- */
.stApp button { border-radius: 999px; }
[data-testid="stBaseButton-secondary"] { border: 1px solid rgba(77,166,255,.5); background: transparent; color: #cfe6ff; }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--blue); background: rgba(77,166,255,.12); }
[data-testid="stBaseButton-primary"] { background: var(--blue); border: 1px solid var(--blue); }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p { color: #04060b; font-weight: 700; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] { border-radius: 12px; border-color: rgba(255,255,255,.14); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.03); }

/* ----- sidebar: dark gradient, same pills and cards ----- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b0b0d, #04060b); border-right: 1px solid rgba(255,255,255,.08); }
.side-label { font-size: 11px; font-weight: 600; letter-spacing: .2em; text-transform: uppercase; color: var(--blue); margin: .6rem 0 .2rem 0; }
.side-note { font-size: 12px; color: var(--muted); line-height: 1.4; }
.side-mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #FFFFFF; }
.logo-row { display: flex; align-items: center; gap: 12px; margin-bottom: .6rem; }
.logo-row span { font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 22px; color: #FFFFFF; }
/* the target buttons sit in a row and wrap onto the next line, like tags */
.st-key-presets { flex-direction: row !important; flex-wrap: wrap; gap: .5rem !important; }
.st-key-presets [data-testid="stElementContainer"] { width: auto !important; }
.st-key-presets button { min-height: 2rem; padding: .15rem .8rem; }
.st-key-presets button p { font-size: 12px; font-weight: 600; }

/* the Saturn ring: a blue arc runs round it all the time (this is the loading circle) */
.ring-load { animation: ringspin 1.8s linear infinite; }
@keyframes ringspin { from { stroke-dashoffset: 0; } to { stroke-dashoffset: -100; } }
@media (prefers-reduced-motion: reduce) { .ring-load { animation: none; } }

/* ----- Follow tab: the detection card, with a thin bar of measured sigma versus the 5 needed ----- */
.detect { margin: .2rem 0 .4rem 0; }
.detect .headline { font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 40px; line-height: 1.05; color: #fff; }
.detect .headline em { font-style: normal; color: var(--blue); }
.detect .text { font-size: 14px; color: var(--body); margin: .6rem 0 .8rem 0; }
.detect .bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,.12); overflow: hidden; }
.detect .bar > div { height: 100%; border-radius: 999px; background: var(--blue); }
.detect .scale { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 12px;
                 color: var(--muted); margin-top: .35rem; }
.section-label { font-size: 12px; font-weight: 600; letter-spacing: .2em; text-transform: uppercase; color: var(--blue); margin: 1.4rem 0 .3rem 0; }

/* ----- start page ----- */
.hero { display: flex; align-items: center; justify-content: space-between; gap: 2rem; padding: 2.4rem 2.4rem; }
.hero .big-title { font-size: clamp(48px, 8vw, 96px); line-height: 1; margin: .4rem 0 .8rem 0; }
.hero .line { font-size: 22px; color: var(--body); margin: 0 0 1rem 0; }
.hero .hint { font-size: 14px; color: var(--muted); margin: 0; }
.hero .art { flex: 0 0 auto; filter: drop-shadow(0 0 40px rgba(77,166,255,.35)); }
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem; }
.steps .card { padding: 1.3rem 1.2rem 1.1rem 1.2rem; }
.steps .n { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--blue); }
.steps .t { font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 28px; color: #fff; margin: .2rem 0 .3rem 0; }
.steps .d { font-size: 13px; color: var(--muted); line-height: 1.45; }
@media (max-width: 900px) { .steps { grid-template-columns: 1fr 1fr; } .hero { flex-direction: column-reverse; align-items: flex-start; } }

/* hide the Streamlit footer */
footer, [data-testid="stFooter"] { visibility: hidden; height: 0; }
</style>
""".replace("STARS", starfield()), unsafe_allow_html=True)


apply_theme()


def stat_row(*items, small=False):
    """Big numbers with a small blue label under each. items = (number, label) pairs."""
    cells = "".join(f'<div class="stat{" small" if small else ""}"><div class="num">{num}</div><div class="lab">{label}</div></div>'
                    for num, label in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def kicker(text):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


# ---------- target presets ----------
# Each button has its name (it also fills the Designation box in "Follow a moving object") and where to look:
#   "coords" = RA and Dec in degrees, typed in directly (WISE 0855 is the challenge's own test target, converted from
#              hours to degrees)
#   "lookup" = a name to look up in the SIMBAD star catalogue
#   "size"   = the patch size in arcminutes, only for buttons that need a particular one
PRESETS = {
    "WISE 0855": dict(coords=(133.79513, -7.24515)),
    "Barnard's Star": dict(lookup="Barnard's star"),
    "Epsilon Indi B": dict(lookup="eps Ind B"),
    "RW Aurigae": dict(lookup="RW Aur"),
    "R Draconis": dict(lookup="R Dra"),
    "2026 ED138": dict(coords=(273.23375, -42.28528), size=12),  # a moving solar-system object, not a star
}
# the buttons that are fixed stars: there is no solar-system object to follow there
STAR_NAMES = {name.lower() for name in PRESETS if name != "2026 ED138"}
STAR_MESSAGE = "This is a star, not a moving solar-system object. Use the Difference tab to look for movement."


@st.cache_data(show_spinner="Looking up the star's position...", ttl=86400)
def look_up(name):
    spot = SkyCoord.from_name(name)  # needs internet
    return float(spot.ra.deg), float(spot.dec.deg)


# Patch sizes in arcminutes: 0.05, 0.1, 0.2, 0.5 and 1 degree.
SIZES = [3, 6, 12, 30, 60]


def resolve_search(requested):
    """Download the pictures for a search. If the archive refuses the patch size, step down to the next smaller
    size until one works. Returns the search with the size that really worked (and remembers the answer)."""
    resolved = st.session_state.setdefault("resolved", {})
    if requested not in resolved:
        ra, dec, size, detector = requested
        refused = None
        for try_size in [s for s in SIZES if s <= size][::-1] or [size]:  # the biggest first
            try:
                cache.ensure_pictures((ra, dec, try_size, detector))  # shows the progress bar
            except spherex_fetch.DownloadRefused as problem:
                refused = problem
                continue
            resolved[requested] = (ra, dec, try_size, detector)
            break
        else:
            raise refused
    return resolved[requested]


def refused_note(asked, got):
    return f"The archive would not give {asked} arcmin pictures, so I am showing {got} arcmin instead."


def clean_number(text):
    """Tidy a typed coordinate. Fancy minus signs (the true minus sign, en and em dashes that some keyboards,
    phones and web pages produce) become a plain '-', which Python can read. Spaces are removed, and a comma used
    as the decimal point becomes a full stop."""
    text = str(text).strip()
    for dash in "\u2212\u2012\u2013\u2014\u2010\u2011\ufe63\uff0d":
        text = text.replace(dash, "-")
    text = text.replace(" ", "")
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    return text


def run_search():
    """Remember what is typed in the sidebar as the current search."""
    # write the tidied numbers back into the boxes, so the Declination box always shows a plain minus sign
    st.session_state.ra_in = clean_number(st.session_state.ra_in)
    st.session_state.dec_in = clean_number(st.session_state.dec_in)
    st.session_state.search = (st.session_state.ra_in, st.session_state.dec_in,
                               st.session_state.get("size_in") or 6, st.session_state.det_in)


def pick_preset(label):
    """Fill RA, Dec (and the Designation box) for a preset button and run the search straight away.

    This runs as the button's on_click callback, which Streamlit calls BEFORE it redraws the page. That is the
    moment when it is allowed to change the value of a box (st.session_state), so the boxes show the new values."""
    spec = PRESETS[label]
    try:
        ra_deg, dec_deg = spec["coords"] if "coords" in spec else look_up(spec["lookup"])
    except Exception:
        st.session_state.notice = f"Couldn't look up {label}. Are you offline? Type its RA and Dec by hand instead."
        return
    st.session_state.notice = ""
    st.session_state.ra_in, st.session_state.dec_in = f"{ra_deg:.5f}", f"{dec_deg:.5f}"
    if "size" in spec:
        st.session_state.size_in = spec["size"]  # e.g. 12 arcmin for 2026 ED138
    st.session_state.follow_name = label         # the Designation box in "Follow a moving object"
    st.session_state.follow_note = ""            # forget an old "this is a star" message
    st.session_state.follow_problem = ""
    st.session_state.active_preset = label
    run_search()


def search_by_hand():
    st.session_state.active_preset = None  # typed coordinates, so no preset is "on"
    run_search()


def set_follow(name, eph, start_mjd, days, eph_key):
    """Remember the predicted path and the dates, so the Follow tab can use them."""
    st.session_state.follow = dict(name=name, eph=eph, start_mjd=start_mjd, days=days, eph_key=eph_key)
    st.session_state.pop("follow_result", None)  # pictures found for an older path are no longer valid


@st.fragment
def follow_controls():
    """Sidebar: name the object, choose the dates, get its predicted path (or paste one)."""
    name = st.text_input("Designation", key="follow_name", placeholder="Type a name, for example 2026 ED138")
    start_date = st.date_input("Start date", value=date(2026, 3, 12), key="follow_start")
    days = st.slider("Days to follow (60 at most)", 1, 60, 30, key="follow_days")
    start_mjd = int(Time(start_date.isoformat()).mjd)

    if st.button("Get predicted path", type="primary", width="stretch", key="follow_get"):
        if not name.strip():
            st.session_state.follow_note = "Type a designation first, for example 2026 ED138."
        elif name.strip().lower() in STAR_NAMES:
            # a fixed star has no predicted path. We do not even ask Horizons: a star's name could match some
            # unrelated asteroid, and we would follow the wrong thing.
            st.session_state.follow_problem = ""
            st.session_state.follow_note = STAR_MESSAGE
        else:
            st.session_state.follow_note = ""
            try:
                eph = cache.follow_path(name.strip(), start_mjd, days)
            except follow.EphemerisError as problem:
                st.session_state.follow_problem = str(problem)
            else:
                st.session_state.follow_problem = ""
                set_follow(name.strip(), eph, start_mjd, days, f"path|{name.strip().lower()}|{start_mjd}|{days}")
                st.rerun(scope="app")  # the Follow tab (outside this fragment) must update too
    if st.session_state.get("follow_note"):
        st.info(st.session_state.follow_note)
    if st.session_state.get("follow_problem"):
        st.warning(f"Could not get the path from the internet ({st.session_state.follow_problem}). "
                   "Paste a table below instead.")
    paste_on = st.toggle("Paste my own ephemeris table", key="follow_use_paste")
    if paste_on or st.session_state.get("follow_problem"):
        text = st.text_area("Or paste an ephemeris table", key="follow_paste", height=150,
                            placeholder="2026 01 01 000000 18 12 56.1 -42 17 07\n2026 01 02 000000 18 12 58.3 -42 17 40",
                            help="MPC format: the date as YYYY MM DD HHMMSS, then R.A. as h m s and Dec. as d m s.")
        if st.button("Use the pasted table", width="stretch", key="follow_use_table"):
            try:
                eph = follow.parse_mpc_table(text)
            except follow.EphemerisError as problem:
                st.error(str(problem))
            else:
                first, last = int(np.ceil(eph.start)), int(np.floor(eph.stop))
                start, stop = max(start_mjd, first), min(start_mjd + days, last)
                if stop <= start:
                    st.error(f"The table covers {cache.nice_date(eph.start)} to {cache.nice_date(eph.stop)}. "
                             "Change the start date so the dates overlap.")
                else:
                    key = hashlib.sha1(text.encode()).hexdigest()[:12]
                    set_follow(name.strip(), eph, start, stop - start, f"paste|{key}|{start}|{stop}")
                    st.rerun(scope="app")


# starting values for the inputs (set once, before the inputs exist)
st.session_state.setdefault("ra_in", "133.79513")
st.session_state.setdefault("dec_in", "-7.24515")
st.session_state.setdefault("follow_name", "")  # empty on purpose: no target is the default
st.session_state.setdefault("size_in", 6)

SATURN = ('<svg width="44" height="44" viewBox="0 0 48 48" aria-hidden="true"><g transform="rotate(-24 24 24)">'
          '<ellipse cx="24" cy="24" rx="21" ry="7" fill="none" stroke="#2a2a30" stroke-width="2.5"/>'
          '<ellipse class="ring-load" cx="24" cy="24" rx="21" ry="7" fill="none" stroke="#4DA6FF" stroke-width="2.5" '
          'stroke-linecap="round" pathLength="100" stroke-dasharray="30 70"/></g>'
          '<circle cx="24" cy="24" r="10" fill="#fff"/></svg>')

BIG_SATURN = ('<svg width="220" height="220" viewBox="0 0 48 48" aria-hidden="true"><g transform="rotate(-24 24 24)">'
              '<ellipse cx="24" cy="24" rx="22" ry="7.4" fill="none" stroke="#2a3a52" stroke-width="1.6"/>'
              '<ellipse cx="24" cy="24" rx="22" ry="7.4" fill="none" stroke="#4DA6FF" stroke-width="1.6" '
              'stroke-linecap="round" pathLength="100" stroke-dasharray="32 68"/></g>'
              '<circle cx="24" cy="24" r="10.5" fill="#fff"/></svg>')
if "search" in st.session_state:
    st.title("Astrodrift")
    st.caption("Type a spot in the sky. See what moves between NASA SPHEREx pictures.")
else:
    # the start page: shown until the first search
    st.markdown('<div class="card hero"><div><div class="kicker">NASA Space Apps · SPHEREx</div>'
                '<div class="big-title">Astrodrift</div>'
                '<p class="line">See what moves between NASA SPHEREx pictures</p>'
                '<p class="hint">Pick a spot on the left and press <b>Show me the sky</b>.</p></div>'
                f'<div class="art">{BIG_SATURN}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f'<div class="logo-row">{SATURN}<span>Astrodrift</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="side-label">Try a target</div>', unsafe_allow_html=True)
    with st.container(key="presets"):
        for label in PRESETS:
            st.button(label, key=f"preset_{label}", on_click=pick_preset, args=(label,),
                      type="primary" if st.session_state.get("active_preset") == label else "secondary")
    if st.session_state.get("notice"):
        st.warning(st.session_state.notice)

    st.markdown('<div class="side-label">Where to look</div>', unsafe_allow_html=True)
    st.text_input("Right ascension (degrees)", key="ra_in")
    st.text_input("Declination (degrees)", key="dec_in")
    st.segmented_control("Size of the patch (arcmin)", SIZES, key="size_in", format_func=str,
                         help="3, 6, 12, 30 and 60 arcmin = 0.05, 0.1, 0.2, 0.5 and 1 degree. "
                              "Bigger patches show more sky but take longer to download.")
    st.selectbox("Detector", spherex_fetch.DETECTORS, index=1, key="det_in")
    st.button("Show me the sky", type="primary", width="stretch", on_click=search_by_hand)

    # filled in further down, once the pictures are loaded (this keeps them in the right place on screen)
    colour_box = st.container()
    advanced = st.expander("Advanced")
    with st.expander("Follow a moving object"):
        follow_controls()
    footer_box = st.container()

# remember the search, so moving the slider doesn't forget it
if "search" not in st.session_state:
    st.stop()  # the start page above is all there is until the first search

requested = st.session_state.search  # what the sidebar asked for: (ra, dec, patch size, detector)
try:
    # (ra, dec, patch size that really worked, detector): the main part of every cache key
    search = resolve_search(requested)
    n_pictures, fetched_at = cache.fetch_info(search)
except spherex_fetch.NoDataError as problem:
    st.warning(str(problem))
    st.stop()
except ValueError:
    st.error("I couldn't read those coordinates. Use numbers like 133.795 and -7.245.")
    st.stop()
if search[2] != requested[2]:
    st.warning(refused_note(requested[2], search[2]))

with advanced:
    tolerance = st.slider("Colour match tolerance (micrometres)", 0.01, 0.10, 0.03, 0.01,
                          help="Smaller = only very similar colours are compared. Fewer pictures, fewer false differences.")
    blur = st.slider("Smoothing amount (pixels)", 0.0, 2.0, 0.8, 0.1,
                     help="How much to blur the pictures when 'Smooth the picture' is on. 0 = no extra blur.")

# group pictures by colour, then pick the colour seen in the most passes (remembered, so it is instant)
overview = cache.colour_overview(search, tolerance)
colours, groups, passes_per_group = overview["colours"], overview["groups"], overview["passes_per_group"]

with colour_box:
    choice = st.selectbox(
        "Infrared colour", range(len(groups)), index=overview["best"],
        format_func=lambda i: f"{colours[groups[i]].mean():.2f} \u00b5m \u00b7 {len(groups[i])} pictures \u00b7 {passes_per_group[i]} passes")
    st.markdown('<div class="side-note">Only pictures of the same colour are compared, so differences are real.</div>',
                unsafe_allow_html=True)

with footer_box:
    st.divider()
    st.markdown('<div class="side-note">Last fetched from NASA IRSA</div>'
                f'<div class="side-mono">{fetched_at.strftime("%d %b %Y, %H:%M")} \u00b7 {n_pictures} pictures</div>',
                unsafe_allow_html=True)
    if st.button("Refresh from NASA", width="stretch"):
        st.cache_data.clear()  # forget everything we downloaded, so the next run asks NASA again
        st.session_state.pop("downloaded", None)
        st.session_state.pop("resolved", None)
        st.rerun()

prep = cache.prepared(search, tolerance, choice)  # line up + clean + combine: done once per search and colour
stat_row((prep["n_chosen"], "pictures"), (passes_per_group[choice], "passes"),
         (f"{prep['mean_colour']:.2f}", "micrometres"))
st.caption(f"{prep['n_chosen']} pictures at about {prep['mean_colour']:.2f} micrometres, "
           f"{passes_per_group[choice]} pass(es), {prep['first_date']} to {prep['last_date']}.")


def picture_id(*parts):
    """A text that names one exact picture. The picture cache uses it as its key."""
    return "|".join(str(p) for p in parts)


def keep_alive(*keys):
    """Streamlit forgets a widget's value if the widget is not drawn in a run. Tabs that are not open are
    skipped, so we copy each value onto itself to keep it until you come back to the tab."""
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


# ---------- one fragment per tab ----------
# A @st.fragment function re-runs ALONE when you touch a widget inside it. The sidebar, the other tabs and the
# slow steps above are not re-run.
# In each tab the picture is on the left and the controls and notes are on the right. Streamlit puts the columns
# under each other on a narrow screen, so there the picture comes first and the controls below it.
PICTURE_COLUMNS = [5, 2]


def panel_columns(name):
    """The picture on the left and, on the right, a card (a container with a key, which the CSS turns into a card)."""
    picture_col, side = st.columns(PICTURE_COLUMNS)
    with side:
        card = st.container(key=f"card_{name}")
    return picture_col, card



@st.fragment
def flipbook_tab(search, tolerance, choice, blur):
    with cache.timed("Flipbook tab (fragment run)"):
        prep = cache.prepared(search, tolerance, choice)
        picture_col, info_col = panel_columns("flip")
        with info_col:
            mode = st.radio("Show", ["Single pictures", "Pass averages (less noise)"])
            if mode.startswith("Pass"):
                show_frames, show_dates = prep["period_images"], prep["period_dates"]
            else:
                show_frames, show_dates = list(prep["frames"]), prep["frame_dates"]
            label = "Pass" if mode.startswith("Pass") else "Picture"
            n = st.slider(label, 1, len(show_frames), 1) if len(show_frames) > 1 else 1
            smooth = st.toggle("Smooth the picture", value=True, key="smooth_flip",
                               help="Off shows the real pixels as sharp squares. On blends them.")
            palette = st.radio("Colours", list(sp.PALETTES), horizontal=True)
            shown = show_frames[n - 1]
            st.caption(f"{label} {n} of {len(show_frames)} | {show_dates[n - 1]}")
            st.caption("Fixed stars stay put. Anything that moves jumps between pictures.")
        with picture_col:
            splot.show(picture_id("flip", search, tolerance, choice, mode, n, palette),
                       sp.to_picture(shown, prep["lo"], prep["hi"], palette=palette),
                       prep["sky_ra"], prep["sky_dec"], shown, key="plot_flip", smooth=smooth, blur=blur)


@st.fragment
def difference_tab(search, tolerance, choice, blur):
    with cache.timed("Difference tab (fragment run)"):
        prep = cache.prepared(search, tolerance, choice)
        n_periods, period_dates = prep["n_periods"], prep["period_dates"]
        if n_periods < 2:
            st.info("The difference view needs at least two passes in this colour. Pick another colour in the sidebar.")
            return
        picture_col, info_col = panel_columns("diff")
        with info_col:
            first = st.selectbox("Earlier pass", range(n_periods), index=0, format_func=lambda k: period_dates[k])
            second = st.selectbox("Later pass", range(n_periods), index=n_periods - 1, format_func=lambda k: period_dates[k])
            if first == second:
                st.info("Pick two different passes.")
                return
            d = cache.difference(search, tolerance, choice, first, second)  # subtract + mover finder: remembered
            movers = d["movers"]
            stat_row((len(movers), "candidates found"), (f"{max(m['score'] for m in movers):.0f}" if movers else "0", "sigma, strongest"),
                     small=True)
            st.caption(f"The later pass was scaled by {d['scale']:.2f} so fixed stars match the earlier one.")
            smooth_diff = st.toggle("Smooth the picture", value=True, key="smooth_diff")
            rings = None  # the selected mover (white = was here, blue = now here)
            if movers:
                chosen_mover = st.selectbox(
                    "Possible movers", range(len(movers)),
                    format_func=lambda k: f"#{k + 1}: {movers[k]['sep']:.1f} pixels apart, strength {movers[k]['score']:.0f} sigma"
                                      f" (in {movers[k]['seen_in']} pair{'s' if movers[k]['seen_in'] > 1 else ''})")
                rings = (movers[chosen_mover]["old"], movers[chosen_mover]["new"])
            st.caption("Blue = brighter now (where it went). White = fainter now (where it was). "
                       "Fixed stars cancel to black. A mover shows as a white spot next to a blue one.")
            if movers:
                st.caption(f"Showing the {len(movers)} strongest of {d['n_loose']} loose candidates. A candidate is kept only if "
                           f"it is at least {sp.STRONG_SIGMA:.0f} sigma, or it repeats in another pair of passes, and it is "
                           "not near the edge or on a bright star.")
            else:
                st.write("No clear movers found.")
                st.caption(f"{d['n_loose']} loose candidate(s) were rejected as noise, edge effects or bright-star leftovers.")
        with picture_col:
            splot.show(picture_id("diff", search, tolerance, choice, first, second),
                       d["rgb"], prep["sky_ra"], prep["sky_dec"], d["diff"], key="plot_diff",
                       rings=rings, value_name="change", smooth=smooth_diff, blur=blur)


@st.fragment
def layers_tab(search, tolerance, choice, blur):
    with cache.timed("Layers tab (fragment run)"):
        prep = cache.prepared(search, tolerance, choice)
        n_periods, period_dates = prep["n_periods"], prep["period_dates"]
        if n_periods < 2:
            st.info("The layers view needs at least two passes in this colour. Pick another colour in the sidebar.")
            return
        picture_col, info_col = panel_columns("layers")
        with info_col:
            # a slider needs a range, so with exactly 2 passes we just use both
            if n_periods > 2:
                if st.session_state.get("layers_how", n_periods) > n_periods:
                    st.session_state.layers_how = n_periods  # a new search with fewer passes: keep the value in range
                how_many = st.slider("Passes to stack (newest ones)", 2, n_periods, n_periods, key="layers_how")
            else:
                how_many = 2
            cutoff = st.slider("Strictness (higher = fewer, surer dots)", 3.0, 8.0, 5.0, 0.5, key="layers_cutoff",
                               help="A spot must be this many times stronger than the random noise to be drawn.")
            smooth_layers = st.toggle("Smooth the picture", value=True, key="smooth_layers")
            st.caption(f"{how_many} passes stacked | {period_dates[n_periods - how_many]} to {period_dates[-1]}")
            st.caption("White = oldest pass, blue = newest pass. Fixed stars are subtracted, "
                       "so they vanish. A mover leaves a trail of dots that turns from white to blue.")
        layers = cache.layers(search, tolerance, choice, how_many, cutoff)  # stacking: remembered
        with picture_col:
            # dots are tiny, so use only half the blur here or they melt into blobs
            splot.show(picture_id("layers", search, tolerance, choice, how_many, cutoff),
                       layers["rgb"], prep["sky_ra"], prep["sky_dec"], layers["strongest"], key="plot_layers",
                       value_name="biggest change", smooth=smooth_layers, blur=blur / 2)


def set_all_opacity():
    """The 'All layers' slider: give every layer's own opacity slider the same value.
    (A callback runs before the page is redrawn, which is when Streamlit allows changing a slider's value.)"""
    for key in [k for k in st.session_state if str(k).startswith("stack_op_")]:
        st.session_state[key] = st.session_state.stack_master


@st.fragment
def stack_tab(search, tolerance, choice):
    """Stack tab: every pass is a see-through layer. A fixed star looks the same in every layer and cancels out;
    a mover leaves one spot per layer, and the app links them into a track you can click.
    It is a fragment, so touching a control here re-runs only this function."""
    with cache.timed("Stack tab (fragment run)"):
        prep = cache.prepared(search, tolerance, choice)
        n_periods, dates = prep["n_periods"], prep["period_dates"]
        if n_periods < 2:
            st.info("The stack view needs at least two passes in this colour. Pick another colour in the sidebar.")
            return
        colours = [sp.layer_colour(i, n_periods) for i in range(n_periods)]

        # A click on a track in the figure arrives as the chart's state. Turn it into the selected track BEFORE
        # the selectbox below is drawn (a widget's value may only be changed before the widget exists).
        hit = stk.clicked_track(st.session_state.get("plot_stack"))
        if hit is not None and hit != st.session_state.get("stack_last_click"):
            st.session_state.stack_track = hit
        st.session_state.stack_last_click = hit
        st.session_state.setdefault("stack_master", 70)
        for i in range(n_periods):
            st.session_state.setdefault(f"stack_op_{i}", 70)
            st.session_state.setdefault(f"stack_on_{i}", True)

        top_left, top_right = st.columns(2)
        remove_fixed = top_left.toggle("Remove fixed things", value=True, key="stack_remove",
                                       help="On: the median of all passes (what never changes) is subtracted from every "
                                            "layer, so fixed stars cancel. Off: the full pictures are shown.")
        top_right.slider("All layers (opacity %)", 0, 100, key="stack_master", on_change=set_all_opacity)
        for i in range(n_periods):
            swatch, box, slider = st.columns([1, 3, 8], vertical_alignment="center")
            swatch.markdown(f'<div style="width:22px;height:22px;border-radius:4px;background:rgb{colours[i]}"></div>',
                            unsafe_allow_html=True)
            box.checkbox(dates[i], key=f"stack_on_{i}")
            slider.slider(f"Opacity, {dates[i]}", 0, 100, key=f"stack_op_{i}")

        data = cache.stack(search, tolerance, choice, remove_fixed)  # remembered, so toggling is instant
        tracks = data["tracks"]
        scale = stk.pixel_scale_arcsec(prep["sky_ra"], prep["sky_dec"])

        picture_col, info_col = panel_columns("stack")
        selected = 0
        with info_col:
            if tracks:
                if st.session_state.get("stack_track", 0) > len(tracks):
                    st.session_state.stack_track = 0  # a new search with fewer tracks: keep the value in range
                stat_row((len(tracks), "mover tracks found"), (f"{tracks[0]['strength']:.0f}", "sigma, strongest"), small=True)
                st.dataframe(
                    [{"Track": n, "Passes": len(t["layers"]),
                      "Pixels per pass": round(stk.measure(t, prep["period_mjds"], scale)["px_per_pass"], 1),
                      "Strength (sigma)": round(t["strength"], 1)} for n, t in enumerate(tracks, start=1)],
                    hide_index=True, width="stretch")
                selected = st.selectbox("Selected track (or click one on the picture)", range(len(tracks) + 1),
                                        format_func=lambda k: "None" if k == 0 else f"Track {k}", key="stack_track")
                if selected:
                    t = tracks[selected - 1]
                    m = stk.measure(t, prep["period_mjds"], scale)
                    st.caption(f"Track {selected}: {m['path_px']:.1f} pixels in total, {m['px_per_pass']:.1f} pixels per pass "
                               f"(1 pixel = {scale:.2f} arcsec), about {m['arcsec_per_day']:.2f} arcsec per day. "
                               f"Seen in {len(t['layers'])} passes, strength {t['strength']:.1f} sigma.")
                    if st.button("I found a mover", type="primary", width="stretch", key="stack_found"):
                        ra, dec = prep["sky_ra"], prep["sky_dec"]
                        row = {"search": str(search[:2]), "strength_sigma": round(t["strength"], 1),
                               "pixels_per_pass": round(m["px_per_pass"], 2), "arcsec_per_day": round(m["arcsec_per_day"], 3)}
                        for layer, (x, y) in zip(t["layers"], t["xy"]):
                            r, c = int(round(y)), int(round(x))
                            row[f"date_{layer + 1}"] = dates[layer]
                            row[f"ra_{layer + 1}"] = round(float(ra[r, c]), 5)
                            row[f"dec_{layer + 1}"] = round(float(dec[r, c]), 5)
                        st.session_state.setdefault("finds", []).append(row)
                        st.toast("Saved to your finds.")
            else:
                st.write("No clear movers found.")
                st.caption("Nothing above 5 sigma in one pass could be linked to a similar spot "
                           f"{stk.MIN_STEP:g} to {stk.MAX_STEP:g} pixels away in the next pass. The plain layers are shown.")
            st.caption("Each colour is one pass. A star that stays put looks the same in every layer. "
                       "Something that moved shows up as separate dots in different colours.")
            st.markdown(" &nbsp; ".join(
                f'<span style="white-space:nowrap"><span style="display:inline-block;width:12px;height:12px;'
                f'background:rgb{c};border-radius:2px;vertical-align:-1px"></span> {d}</span>'
                for c, d in zip(colours, dates)), unsafe_allow_html=True)
        with picture_col:
            on = [st.session_state[f"stack_on_{i}"] for i in range(n_periods)]
            opacity = [st.session_state[f"stack_op_{i}"] / 100 for i in range(n_periods)]
            rgb = stk.composite(data["layers"], on, opacity)
            fig = splot.stack_figure(rgb, prep["sky_ra"], prep["sky_dec"], data["strongest"],
                                     tracks, selected, colours, dates)
            st.plotly_chart(fig, width="stretch", theme=None, config=splot.CONFIG, key="plot_stack",
                            on_select="rerun", selection_mode="points")
            st.caption("Scroll to zoom, drag to move, double-click to reset. Click a ring to pick that track.")
        finds = st.session_state.get("finds")
        if finds:
            with st.container(key="card_finds"):
                kicker(f"My finds ({len(finds)})")
                st.dataframe(finds, hide_index=True, width="stretch")
                keys = sorted({k for f in finds for k in f})
                csv = "\n".join([",".join(keys)] + [",".join(str(f.get(k, "")) for k in keys) for f in finds])
                st.download_button("Download finds (CSV)", csv, "astrodrift_finds.csv", "text/csv")


@st.fragment
def colour_tab(search, tolerance, blur):
    with cache.timed("Colour tab (fragment run)"):
        picture_col, info_col = panel_columns("colour")
        with info_col:
            # The colour picture has its own patch size, and it starts at the largest one (60 arcmin).
            size = st.segmented_control("Patch size for the colour picture (arcmin)", SIZES, default=SIZES[-1],
                                        key="colour_size", format_func=str) or SIZES[-1]
            try:
                colour_search = resolve_search((search[0], search[1], size, search[3]))  # downloads if needed
            except spherex_fetch.NoDataError as problem:
                st.warning(str(problem))
                return
            if colour_search[2] != size:
                st.warning(refused_note(size, colour_search[2]))
            n_all, all_dates = overview_for(colour_search, tolerance)
            if st.session_state.get("colour_pass", 0) >= n_all:
                st.session_state.colour_pass = 0  # a new search with fewer passes: keep the value in range
            which = st.selectbox("Pass", range(n_all), format_func=lambda k: all_dates[k], key="colour_pass")
            picture = cache.colour_picture(colour_search, which)  # line up + colour composite: remembered
            if picture["n_members"] < 3:
                st.info("This pass has fewer than 3 pictures, so I can't make a colour picture.")
                return
            smooth_c = st.toggle("Smooth the picture", value=True, key="smooth_colour")
            st.caption(f"Infrared colour picture | {all_dates[which]} | {picture['n_members']} pictures")
            ranges = picture["ranges"]
            st.caption(
                f"FALSE COLOUR. Blue = shortest wavelengths ({ranges[0][0]:.2f} to {ranges[0][1]:.2f} micrometres), "
                f"green = middle ({ranges[1][0]:.2f} to {ranges[1][1]:.2f}), "
                f"red = longest ({ranges[2][0]:.2f} to {ranges[2][1]:.2f}). "
                "Infrared cannot be seen by eye, so these are visible colours painted on. "
                "Red things are brighter at longer wavelengths. "
                "These pictures were taken hours to days apart, so something that moves can show coloured fringes. "
                "Use the Difference tab to decide what is a mover.")
        with picture_col:
            splot.show(picture_id("colour", colour_search, which), picture["rgb"], picture["ra"], picture["dec"],
                       picture["brightness"], key="plot_colour", smooth=smooth_c, blur=blur)


@st.fragment
def follow_tab(blur):
    """Follow tab: the tracked pictures, their stack, their spectrum, and an honest status line."""
    state = st.session_state.get("follow")
    if not state:
        st.info("Open **Follow a moving object** in the sidebar, type a designation and press **Get predicted path**.")
        return
    eph, start_mjd, days = state["eph"], state["start_mjd"], state["days"]
    run_key = f"{state['eph_key']}|{follow.CUTOUT_ARCMIN}"
    ra0, dec0 = (float(x[0]) for x in eph.at(start_mjd))
    st.caption(f"{state['name']} | path from {eph.source} | {cache.nice_date(start_mjd)} to "
               f"{cache.nice_date(start_mjd + days)} ({days} days) | first position RA {ra0:.5f}, Dec {dec0:.5f}")

    if st.button("Find SPHEREx pictures", type="primary", key="follow_go"):
        pictures, failed_days = [], 0
        bar = st.progress(0.0, text="Asking NASA's archive, one day at a time...")
        for k in range(days):
            try:
                pictures += cache.follow_day(state["eph_key"], eph, start_mjd + k, follow.CUTOUT_ARCMIN)
            except Exception:  # one bad day (network hiccup) should not stop the whole search
                failed_days += 1
            bar.progress((k + 1) / days, text=f"Day {k + 1} of {days} | {len(pictures)} pictures so far")
        bar.empty()
        st.session_state.follow_result = dict(run_key=run_key, pictures=pictures, failed_days=failed_days)

    result = st.session_state.get("follow_result")
    if not result or result["run_key"] != run_key:
        st.write("Press **Find SPHEREx pictures**. It searches each day in turn and can take a few minutes "
                 "the first time. Days you have searched before are instant.")
        return
    if result["failed_days"]:
        st.warning(f"{result['failed_days']} day(s) could not be searched (network problem). Press the button again to retry them.")
    if not result["pictures"]:
        st.info("No SPHEREx pictures cover the predicted path on these days. Try another start date.")
        return

    with cache.timed("Follow tab (fragment run)"):
        frames = cache.follow_frames(run_key, result["pictures"])
        if frames["n"] == 0:
            st.info("SPHEREx pictures were found, but the object is too close to their edges to measure.")
            return
        n = frames["n"]
        st.caption(f"{n} tracked pictures, {frames['dates'][0]} to {frames['dates'][-1]}. Each cutout is centred on where the "
                   "object is predicted to be at that picture's own time, so the object is always in the middle.")
        centre = (follow.CENTRE, follow.CENTRE)

        # (a) flipbook with a crosshair on the predicted position
        st.markdown('<div class="section-label">Flipbook of the tracked pictures</div>', unsafe_allow_html=True)
        picture_col, info_col = panel_columns("follow_flip")
        with info_col:
            if st.session_state.get("follow_i", 1) > n:
                st.session_state.follow_i = 1  # a new search with fewer pictures: keep the slider in range
            i = (st.slider("Picture", 1, n, 1, key="follow_i") if n > 1 else 1) - 1
            smooth = st.toggle("Smooth the picture", value=True, key="smooth_follow_flip")
            st.caption(f"Picture {i + 1} of {n} | {frames['dates'][i]} | {frames['wavelengths'][i]:.2f} \u00b5m")
        with picture_col:
            splot.show(picture_id("follow_flip", run_key, i), sp.to_picture(frames["cleaned"][i], frames["lo"], frames["hi"]),
                       frames["ra"][i], frames["dec"][i], frames["cleaned"][i], key="plot_follow_flip",
                       smooth=smooth, blur=blur, cross=centre)

        # (b) stack of one colour group, and (d) the honest status line
        st.markdown('<div class="section-label">Stack (median) of one colour</div>', unsafe_allow_html=True)
        picture_col, info_col = panel_columns("follow_stack")
        groups = follow.colour_groups(frames["wavelengths"])
        with info_col:
            if st.session_state.get("follow_group", 0) >= len(groups):
                st.session_state.follow_group = 0
            g = st.selectbox("Colour group", range(len(groups)), key="follow_group",
                             format_func=lambda k: f"{frames['wavelengths'][groups[k]].mean():.2f} \u00b5m | {len(groups[k])} pictures")
            members = groups[g]
            stacked = cache.follow_stack(run_key, tuple(members), frames)
            mean_colour = float(frames["wavelengths"][members].mean())
            detected, text = follow.status_line(stacked["significance"], len(members), mean_colour)
            sigma = (stacked["significance"] or {}).get("sigma")
            fill = 0 if sigma is None else max(0.0, min(sigma / follow.DETECTION_SIGMA, 1.0)) * 100
            headline = "<em>Detected</em>" if detected else "Not <em>detected</em>"
            sigma_text = "no number" if sigma is None else f"{sigma:.1f} sigma"
            st.markdown(f'<div class="card detect"><div class="headline">{headline}</div><div class="text">{text}</div>'
                        f'<div class="bar"><div style="width:{fill:.0f}%"></div></div>'
                        f'<div class="scale"><span>0</span><span>measured {sigma_text}</span>'
                        f'<span>{follow.DETECTION_SIGMA:g} needed</span></div></div>', unsafe_allow_html=True)
            smooth_stack = st.toggle("Smooth the picture", value=True, key="smooth_follow_stack")
            st.caption("Median of the cutouts, each with its background removed. The object adds up in the middle; stars are in "
                       "different places in each cutout, so they smear out. The sigma number compares the middle with the "
                       "scatter of the same circle placed elsewhere on this picture. Sky positions shown are for the middle picture.")
        stack = stacked["stack"]
        lo, hi = np.nanpercentile(stack, [50, 99.8])
        middle = members[len(members) // 2]
        with picture_col:
            splot.show(picture_id("follow_stack", run_key, tuple(members)), sp.to_picture(stack, lo, hi),
                       frames["ra"][middle], frames["dec"][middle], stack, key="plot_follow_stack",
                       smooth=smooth_stack, blur=blur, cross=centre)

        # (c) spectrum
        st.markdown('<div class="section-label">Spectrum at the predicted position</div>', unsafe_allow_html=True)
        wl, flux, error = frames["spectrum"]
        if len(wl) == 0:
            st.info("None of the pictures had enough good pixels around the centre to measure.")
        else:
            st.plotly_chart(splot.spectrum_figure(wl, flux, error, frames["unit"]), theme=None, key="plot_follow_spectrum",
                            config={"displaylogo": False, "scrollZoom": True})
            st.caption("Brightness in a circle of radius 2 pixels at the predicted position, with the local background "
                       "(a ring around it) subtracted. Error bars come from the VARIANCE layer of each picture. "
                       "If the object is not detected, these points just scatter around zero.")


def overview_for(search, tolerance):
    ov = cache.colour_overview(search, tolerance)
    return ov["n_all"], ov["all_dates"]


# on_change="rerun" lets each tab know if it is open (tab.open). Layers, Colour and Follow are skipped until you open them.
tab_flip, tab_diff, tab_layers, tab_stack, tab_colour, tab_follow = st.tabs(
    ["Flipbook", "Difference", "Layers", "Stack", "Colour", "Follow"], on_change="rerun", key="main_tabs")
with tab_flip:
    kicker("Flipbook · every picture in turn")
    flipbook_tab(search, tolerance, choice, blur)
with tab_diff:
    kicker("Difference · what changed between two passes")
    difference_tab(search, tolerance, choice, blur)
with tab_layers:
    kicker("Layers · passes stacked from old to new")
    if tab_layers.open:
        layers_tab(search, tolerance, choice, blur)
    else:
        keep_alive("layers_how", "layers_cutoff", "smooth_layers")
with tab_stack:
    kicker("Stack · one see-through layer per pass")
    if tab_stack.open:
        stack_tab(search, tolerance, choice)
    else:
        keep_alive("stack_remove", "stack_master", "stack_track",
                   *[f"stack_{kind}_{i}" for kind in ("on", "op") for i in range(prep["n_periods"])])
with tab_colour:
    kicker("Colour · false-colour infrared picture")
    if tab_colour.open:
        colour_tab(search, tolerance, blur)
    else:
        keep_alive("colour_pass", "smooth_colour", "colour_size")
with tab_follow:
    kicker("Follow · track one moving object")
    if tab_follow.open:
        follow_tab(blur)
    else:
        keep_alive("follow_i", "follow_group", "smooth_follow_flip", "smooth_follow_stack")

print(f"[astrodrift] FULL SCRIPT RUN: {time.perf_counter() - RUN_START:.3f} s", flush=True)
