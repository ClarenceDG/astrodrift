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

RUN_START = time.perf_counter()  # for the "full script run" time printed at the very end

st.set_page_config(page_title="Astrodrift", layout="wide")

# ---------- look and feel: one block of CSS (the language that styles web pages) ----------
# Black, white and one blue accent (#4DA6FF). Fonts come from Google Fonts.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* body text = Inter. Only text elements are listed, so the icon font is left alone */
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.stApp textarea, .stApp [data-baseweb="select"] { font-family: 'Inter', sans-serif; }

/* title and headings = Space Grotesk */
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.01em; }

/* coordinates and numbers = JetBrains Mono */
.stApp input, [data-testid="stMetricValue"], code { font-family: 'JetBrains Mono', monospace; }

/* colours of the small text */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #9A9AA3; }
[data-testid="stWidgetLabel"] p { color: #C9C9D0; }

/* lots of empty space */
.block-container { padding-top: 3rem; padding-bottom: 4rem; max-width: 1400px; }
[data-testid="stVerticalBlock"] { gap: 1rem; }

/* sidebar panel */
[data-testid="stSidebar"] { background: #0B0B0D; border-right: 1px solid #1E1E22; }

/* buttons and inputs: 6px corners, dark borders */
.stApp button { border-radius: 6px; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
    border-radius: 6px; border-color: #2A2A30; }
[data-testid="stBaseButton-secondary"] { border: 1px solid #2A2A30; background: #0B0B0D; }
/* the blue button has black text */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p { color: #000000; font-weight: 600; }

/* cards and expanders: 8px corners */
[data-testid="stExpander"] { border: 1px solid #1E1E22; border-radius: 8px; background: #0B0B0D; }

/* centre every picture */
[data-testid="stPlotlyChart"] { display: flex; justify-content: center; }

/* small grey capital labels, like TRY A TARGET */
.side-label { font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
              color: #9A9AA3; margin: 0.6rem 0 0.2rem 0; }
.side-note { font-size: 12px; color: #9A9AA3; line-height: 1.4; }
.side-mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #FFFFFF; }
.logo-row { display: flex; align-items: center; gap: 12px; margin-bottom: 0.6rem; }
.logo-row span { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 22px; color: #FFFFFF; }

/* the five target buttons sit in a row and wrap onto the next line, like tags */
.st-key-presets { flex-direction: row !important; flex-wrap: wrap; gap: 0.5rem !important; }
.st-key-presets [data-testid="stElementContainer"] { width: auto !important; }
.st-key-presets button { min-height: 2rem; padding: 0.15rem 0.7rem; }
.st-key-presets button p { font-size: 12px; font-weight: 600; }

/* the Saturn ring: a blue arc runs round it all the time (this is the loading circle) */
.ring-load { animation: ringspin 1.8s linear infinite; }
@keyframes ringspin { from { stroke-dashoffset: 0; } to { stroke-dashoffset: -100; } }
@media (prefers-reduced-motion: reduce) { .ring-load { animation: none; } }

/* the honest status line of the Follow tab: always contains a sigma number */
.status-line { border: 1px solid #1E1E22; border-radius: 8px; background: #0B0B0D; padding: 0.8rem 1rem;
               font-size: 15px; color: #C9C9D0; }
.status-line.yes { border-color: #4DA6FF; color: #FFFFFF; }
.section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
                 color: #9A9AA3; margin: 1.4rem 0 0.3rem 0; }

/* hide the Streamlit footer */
footer, [data-testid="stFooter"] { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

st.title("Astrodrift")
st.caption("Type a spot in the sky. See what moves between NASA SPHEREx pictures.")


# ---------- target presets ----------
# WISE 0855 is the challenge's own test target (converted from hours to degrees), so it needs no lookup.
# The others are looked up by name in the SIMBAD star catalogue. The dictionary maps button text -> name to look up.
PRESETS = {"WISE 0855": None, "Barnard's Star": "Barnard's star", "Epsilon Indi B": "eps Ind B",
           "RW Aurigae": "RW Aur", "R Draconis": "R Dra"}


@st.cache_data(show_spinner="Looking up the star's position...", ttl=86400)
def look_up(name):
    spot = SkyCoord.from_name(name)  # needs internet
    return float(spot.ra.deg), float(spot.dec.deg)


def run_search():
    """Remember what is typed in the sidebar as the current search."""
    st.session_state.search = (st.session_state.ra_in, st.session_state.dec_in,
                               st.session_state.get("size_in") or 6, st.session_state.det_in)


def pick_preset(label):
    """Fill RA and Dec for a preset button and run the search straight away."""
    try:
        ra_deg, dec_deg = (133.79513, -7.24515) if PRESETS[label] is None else look_up(PRESETS[label])
    except Exception:
        st.session_state.notice = f"Couldn't look up {label}. Are you offline? Type its RA and Dec by hand instead."
        return
    st.session_state.notice = ""
    st.session_state.ra_in, st.session_state.dec_in = f"{ra_deg:.5f}", f"{dec_deg:.5f}"
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
    name = st.text_input("Designation", "2026 ED138", key="follow_name")
    start_date = st.date_input("Start date", value=date(2026, 3, 12), key="follow_start")
    days = st.slider("Days to follow (60 at most)", 1, 60, 30, key="follow_days")
    start_mjd = int(Time(start_date.isoformat()).mjd)

    if st.button("Get predicted path", type="primary", width="stretch", key="follow_get"):
        try:
            eph = cache.follow_path(name.strip(), start_mjd, days)
        except follow.EphemerisError as problem:
            st.session_state.follow_problem = str(problem)
        else:
            st.session_state.follow_problem = ""
            set_follow(name.strip(), eph, start_mjd, days, f"path|{name.strip().lower()}|{start_mjd}|{days}")
            st.rerun(scope="app")  # the Follow tab (outside this fragment) must update too
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

SATURN = ('<svg width="44" height="44" viewBox="0 0 48 48" aria-hidden="true"><g transform="rotate(-24 24 24)">'
          '<ellipse cx="24" cy="24" rx="21" ry="7" fill="none" stroke="#2a2a30" stroke-width="2.5"/>'
          '<ellipse class="ring-load" cx="24" cy="24" rx="21" ry="7" fill="none" stroke="#4DA6FF" stroke-width="2.5" '
          'stroke-linecap="round" pathLength="100" stroke-dasharray="30 70"/></g>'
          '<circle cx="24" cy="24" r="10" fill="#fff"/></svg>')

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
    st.segmented_control("Size of the patch", [3, 6, 12], default=6, key="size_in",
                         format_func=lambda v: f"{v} arcmin")
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
    st.write("Pick a spot on the left and press **Show me the sky**.")
    st.stop()

search = st.session_state.search  # (ra, dec, patch size, detector): the main part of every cache key
try:
    n_pictures, fetched_at = cache.fetch_info(search)
except spherex_fetch.NoDataError as problem:
    st.warning(str(problem))
    st.stop()
except ValueError:
    st.error("I couldn't read those coordinates. Use numbers like 133.795 and -7.245.")
    st.stop()

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
        st.rerun()

prep = cache.prepared(search, tolerance, choice)  # line up + clean + combine: done once per search and colour
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
@st.fragment
def flipbook_tab(search, tolerance, choice, blur):
    with cache.timed("Flipbook tab (fragment run)"):
        prep = cache.prepared(search, tolerance, choice)
        mode = st.radio("Show", ["Single pictures", "Pass averages (less noise)"], horizontal=True)
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
        splot.show(picture_id("flip", search, tolerance, choice, mode, n, palette),
                   sp.to_picture(shown, prep["lo"], prep["hi"], palette=palette),
                   prep["sky_ra"], prep["sky_dec"], shown, key="plot_flip", smooth=smooth, blur=blur)
        st.caption("Fixed stars stay put. Anything that moves jumps between pictures.")


@st.fragment
def difference_tab(search, tolerance, choice, blur):
    with cache.timed("Difference tab (fragment run)"):
        prep = cache.prepared(search, tolerance, choice)
        n_periods, period_dates = prep["n_periods"], prep["period_dates"]
        if n_periods < 2:
            st.info("The difference view needs at least two passes in this colour. Pick another colour in the sidebar.")
            return
        c1, c2 = st.columns(2)
        first = c1.selectbox("Earlier pass", range(n_periods), index=0, format_func=lambda k: period_dates[k])
        second = c2.selectbox("Later pass", range(n_periods), index=n_periods - 1, format_func=lambda k: period_dates[k])
        if first == second:
            st.info("Pick two different passes.")
            return
        d = cache.difference(search, tolerance, choice, first, second)  # subtract + mover finder: remembered
        movers = d["movers"]
        st.caption(f"The later pass was scaled by {d['scale']:.2f} so fixed stars match the earlier one.")
        smooth_diff = st.toggle("Smooth the picture", value=True, key="smooth_diff")
        rings = None  # the selected mover (white = was here, blue = now here)
        if movers:
            chosen_mover = st.selectbox(
                "Possible movers", range(len(movers)),
                format_func=lambda k: f"#{k + 1}: {movers[k]['sep']:.1f} pixels apart, strength {movers[k]['score']:.0f}")
            rings = (movers[chosen_mover]["old"], movers[chosen_mover]["new"])
        splot.show(picture_id("diff", search, tolerance, choice, first, second),
                   d["rgb"], prep["sky_ra"], prep["sky_dec"], d["diff"], key="plot_diff",
                   rings=rings, value_name="change", smooth=smooth_diff, blur=blur)
        st.caption("Blue = brighter now (where it went). White = fainter now (where it was). "
                   "Fixed stars cancel to black. A mover shows as a white spot next to a blue one.")
        if not movers:
            st.write("No white-and-blue pair stands out between these two passes.")


@st.fragment
def layers_tab(search, tolerance, choice, blur):
    with cache.timed("Layers tab (fragment run)"):
        prep = cache.prepared(search, tolerance, choice)
        n_periods, period_dates = prep["n_periods"], prep["period_dates"]
        if n_periods < 2:
            st.info("The layers view needs at least two passes in this colour. Pick another colour in the sidebar.")
            return
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
        layers = cache.layers(search, tolerance, choice, how_many, cutoff)  # stacking: remembered
        st.caption(f"{how_many} passes stacked | {period_dates[n_periods - how_many]} to {period_dates[-1]}")
        # dots are tiny, so use only half the blur here or they melt into blobs
        splot.show(picture_id("layers", search, tolerance, choice, how_many, cutoff),
                   layers["rgb"], prep["sky_ra"], prep["sky_dec"], layers["strongest"], key="plot_layers",
                   value_name="biggest change", smooth=smooth_layers, blur=blur / 2)
        st.caption("White = oldest pass, blue = newest pass. Fixed stars are subtracted, "
                   "so they vanish. A mover leaves a trail of dots that turns from white to blue.")


@st.fragment
def colour_tab(search, tolerance, blur):
    with cache.timed("Colour tab (fragment run)"):
        n_all, all_dates = overview_for(search, tolerance)
        if st.session_state.get("colour_pass", 0) >= n_all:
            st.session_state.colour_pass = 0  # a new search with fewer passes: keep the value in range
        which = st.selectbox("Pass", range(n_all), format_func=lambda k: all_dates[k], key="colour_pass")
        picture = cache.colour_picture(search, which)  # line up + colour composite: remembered
        if picture["n_members"] < 3:
            st.info("This pass has fewer than 3 pictures, so I can't make a colour picture.")
            return
        smooth_c = st.toggle("Smooth the picture", value=True, key="smooth_colour")
        st.caption(f"Infrared colour picture | {all_dates[which]} | {picture['n_members']} pictures")
        splot.show(picture_id("colour", search, which), picture["rgb"], picture["ra"], picture["dec"],
                   picture["brightness"], key="plot_colour", smooth=smooth_c, blur=blur)
        ranges = picture["ranges"]
        st.caption(
            f"FALSE COLOUR. Blue = shortest wavelengths ({ranges[0][0]:.2f} to {ranges[0][1]:.2f} micrometres), "
            f"green = middle ({ranges[1][0]:.2f} to {ranges[1][1]:.2f}), "
            f"red = longest ({ranges[2][0]:.2f} to {ranges[2][1]:.2f}). "
            "Infrared cannot be seen by eye, so these are visible colours painted on. "
            "Red things are brighter at longer wavelengths. "
            "These pictures were taken hours to days apart, so something that moves can show coloured fringes. "
            "Use the Difference tab to decide what is a mover.")


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
        if st.session_state.get("follow_i", 1) > n:
            st.session_state.follow_i = 1  # a new search with fewer pictures: keep the slider in range
        i = (st.slider("Picture", 1, n, 1, key="follow_i") if n > 1 else 1) - 1
        smooth = st.toggle("Smooth the picture", value=True, key="smooth_follow_flip")
        st.caption(f"Picture {i + 1} of {n} | {frames['dates'][i]} | {frames['wavelengths'][i]:.2f} \u00b5m")
        splot.show(picture_id("follow_flip", run_key, i), sp.to_picture(frames["cleaned"][i], frames["lo"], frames["hi"]),
                   frames["ra"][i], frames["dec"][i], frames["cleaned"][i], key="plot_follow_flip",
                   smooth=smooth, blur=blur, cross=centre)

        # (b) stack of one colour group, and (d) the honest status line
        st.markdown('<div class="section-label">Stack (median) of one colour</div>', unsafe_allow_html=True)
        groups = follow.colour_groups(frames["wavelengths"])
        if st.session_state.get("follow_group", 0) >= len(groups):
            st.session_state.follow_group = 0
        g = st.selectbox("Colour group", range(len(groups)), key="follow_group",
                         format_func=lambda k: f"{frames['wavelengths'][groups[k]].mean():.2f} \u00b5m | {len(groups[k])} pictures")
        members = groups[g]
        stacked = cache.follow_stack(run_key, tuple(members), frames)
        mean_colour = float(frames["wavelengths"][members].mean())
        detected, text = follow.status_line(stacked["significance"], len(members), mean_colour)
        st.markdown(f'<div class="status-line{" yes" if detected else ""}">{text}</div>', unsafe_allow_html=True)
        smooth_stack = st.toggle("Smooth the picture", value=True, key="smooth_follow_stack")
        stack = stacked["stack"]
        lo, hi = np.nanpercentile(stack, [50, 99.8])
        middle = members[len(members) // 2]
        splot.show(picture_id("follow_stack", run_key, tuple(members)), sp.to_picture(stack, lo, hi),
                   frames["ra"][middle], frames["dec"][middle], stack, key="plot_follow_stack",
                   smooth=smooth_stack, blur=blur, cross=centre)
        st.caption("Median of the cutouts, each with its background removed. The object adds up in the middle; stars are in "
                   "different places in each cutout, so they smear out. The sigma number compares the middle with the "
                   "scatter of the same circle placed elsewhere on this picture. Sky positions shown are for the middle picture.")

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
tab_flip, tab_diff, tab_layers, tab_colour, tab_follow = st.tabs(
    ["Flipbook", "Difference", "Layers", "Colour", "Follow"], on_change="rerun", key="main_tabs")
with tab_flip:
    flipbook_tab(search, tolerance, choice, blur)
with tab_diff:
    difference_tab(search, tolerance, choice, blur)
with tab_layers:
    if tab_layers.open:
        layers_tab(search, tolerance, choice, blur)
    else:
        keep_alive("layers_how", "layers_cutoff", "smooth_layers")
with tab_colour:
    if tab_colour.open:
        colour_tab(search, tolerance, blur)
    else:
        keep_alive("colour_pass", "smooth_colour")
with tab_follow:
    if tab_follow.open:
        follow_tab(blur)
    else:
        keep_alive("follow_i", "follow_group", "smooth_follow_flip", "smooth_follow_stack")

print(f"[astrodrift] FULL SCRIPT RUN: {time.perf_counter() - RUN_START:.3f} s", flush=True)
