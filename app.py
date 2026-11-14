"""Astrodrift: type a spot in the sky, see what changes between NASA SPHEREx pictures."""
from datetime import datetime

import numpy as np
import streamlit as st
from astropy.coordinates import SkyCoord
from astropy.time import Time

import spherex_fetch
import spherex_process as sp

st.set_page_config(page_title="Astrodrift", layout="wide")

# width divided by height of the picture frame
SHAPES = {"Square": 1.0, "Rectangle (3:2)": 1.5, "Wide rectangle (2:1)": 2.0}

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

/* centre every picture (and its caption) */
[data-testid="stImage"] { display: flex; flex-direction: column; align-items: center; }

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

/* hide the Streamlit footer */
footer, [data-testid="stFooter"] { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)

st.title("Astrodrift")
st.caption("Type a spot in the sky. See what moves between NASA SPHEREx pictures.")


def nice_date(mjd):
    return Time(float(mjd), format="mjd").to_datetime().strftime("%d %b %Y")


@st.cache_data(show_spinner="Asking NASA's archive for pictures...", ttl=3600)
def get_pictures(ra, dec, size_deg, detector):
    # we also keep the time of the download, to show "Last fetched" in the sidebar
    return spherex_fetch.fetch_pictures(ra, dec, size_deg, detector), datetime.now()


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
    footer_box = st.container()

# remember the search, so moving the slider doesn't forget it
if "search" not in st.session_state:
    st.write("Pick a spot on the left and press **Show me the sky**.")
    st.stop()

ra_s, dec_s, size_s, det_s = st.session_state.search
try:
    pictures, fetched_at = get_pictures(float(ra_s), float(dec_s), size_s / 60, det_s)
except spherex_fetch.NoDataError as problem:
    st.warning(str(problem))
    st.stop()
except ValueError:
    st.error("I couldn't read those coordinates. Use numbers like 133.795 and -7.245.")
    st.stop()

# group pictures by colour, then pick the colour seen in the most passes
mjds = np.array([p["mjd"] for p in pictures])
colours = np.array([p["wavelength"] for p in pictures])
with advanced:
    tolerance = st.slider("Colour match tolerance (micrometres)", 0.01, 0.10, 0.03, 0.01,
                          help="Smaller = only very similar colours are compared. Fewer pictures, fewer false differences.")
    blur = st.slider("Smoothing amount (pixels)", 0.0, 2.0, 0.8, 0.1,
                     help="How much to blur the pictures when 'Smooth the picture' is on. 0 = no extra blur.")
    shape_name = st.selectbox("Frame shape", list(SHAPES), index=1,
                              help="Rectangles cut the top and bottom off the square patch.")
    pic_w = st.slider("Picture size (pixels wide)", 400, 1600, 1000, 100,
                      help="Bigger = easier to see small dots. It shrinks by itself if the window is narrower.")
groups = sp.group_by_colour(colours, tolerance)
passes_per_group = [sp.assign_periods(mjds[g]).max() + 1 for g in groups]
best = max(range(len(groups)), key=lambda i: (passes_per_group[i], len(groups[i])))

with colour_box:
    choice = st.selectbox(
        "Infrared colour", range(len(groups)), index=best,
        format_func=lambda i: f"{colours[groups[i]].mean():.2f} \u00b5m \u00b7 {len(groups[i])} pictures \u00b7 {passes_per_group[i]} passes")
    st.markdown('<div class="side-note">Only pictures of the same colour are compared, so differences are real.</div>',
                unsafe_allow_html=True)

with footer_box:
    st.divider()
    st.markdown('<div class="side-note">Last fetched from NASA IRSA</div>'
                f'<div class="side-mono">{fetched_at.strftime("%d %b %Y, %H:%M")} \u00b7 {len(pictures)} pictures</div>',
                unsafe_allow_html=True)
    if st.button("Refresh from NASA", width="stretch"):
        st.cache_data.clear()  # forget everything we downloaded, so the next run asks NASA again
        st.rerun()
chosen = sorted(groups[choice], key=lambda i: mjds[i])

frames = sp.line_up([pictures[i]["image"] for i in chosen], [pictures[i]["header"] for i in chosen])
frames = sp.crop_shape(np.array([sp.remove_background(f) for f in frames]), SHAPES[shape_name])
lo, hi = sp.brightness_limits(frames)

st.caption(f"{len(chosen)} pictures at about {colours[chosen].mean():.2f} micrometres, "
           f"{passes_per_group[choice]} pass(es), {nice_date(mjds[chosen[0]])} to {nice_date(mjds[chosen[-1]])}.")

times = mjds[chosen]
periods = sp.assign_periods(times)
n_periods = int(periods.max()) + 1
period_images = [sp.combine(frames[periods == k]) for k in range(n_periods)]
period_dates = [nice_date(times[periods == k].mean()) for k in range(n_periods)]

tab_flip, tab_diff, tab_layers, tab_colour = st.tabs(["Flipbook", "Difference", "Layers", "Colour"])

with tab_flip:
    mode = st.radio("Show", ["Single pictures", "Pass averages (less noise)"], horizontal=True)
    if mode.startswith("Pass"):
        show_frames, show_dates = period_images, period_dates
    else:
        show_frames, show_dates = list(frames), [nice_date(t) for t in times]
    label = "Pass" if mode.startswith("Pass") else "Picture"
    n = st.slider(label, 1, len(show_frames), 1) if len(show_frames) > 1 else 1
    zoom = st.slider("Zoom (centre of the patch)", 1, 4, 1)
    smooth = st.toggle("Smooth the picture", value=True, key="smooth_flip",
                       help="Off shows the real pixels as sharp squares. On blends them.")
    palette = st.radio("Colours", list(sp.PALETTES), horizontal=True)
    shown = sp.crop_centre(show_frames[n - 1], zoom)
    st.image(sp.to_picture(shown, lo, hi, smooth=smooth, palette=palette, blur=blur, width=pic_w), width=pic_w,
             caption=f"{label} {n} of {len(show_frames)} | {show_dates[n - 1]}")
    st.caption("Fixed stars stay put. Anything that moves jumps between pictures.")

with tab_diff:
    if n_periods < 2:
        st.info("The difference view needs at least two passes in this colour. Pick another colour in the sidebar.")
    else:
        c1, c2 = st.columns(2)
        first = c1.selectbox("Earlier pass", range(n_periods), index=0, format_func=lambda k: period_dates[k])
        second = c2.selectbox("Later pass", range(n_periods), index=n_periods - 1, format_func=lambda k: period_dates[k])
        if first == second:
            st.info("Pick two different passes.")
        else:
            matched, scale = sp.match_brightness(period_images[first], period_images[second])
            diff = matched - period_images[first]
            average = (matched + period_images[first]) / 2
            sigma = sp.noise_level(diff)
            movers = sp.find_movers(diff, sigma, average)
            st.caption(f"The later pass was scaled by {scale:.2f} so fixed stars match the earlier one.")
            smooth_diff = st.toggle("Smooth the picture", value=True, key="smooth_diff")
            pic = sp.render_difference(diff, sigma, width=pic_w, smooth=smooth_diff, blur=blur)
            if movers:
                chosen_mover = st.selectbox(
                    "Possible movers", range(len(movers)),
                    format_func=lambda k: f"#{k + 1}: {movers[k]['sep']:.1f} pixels apart, strength {movers[k]['score']:.0f}")
                pic = sp.draw_pair(pic, movers[chosen_mover], diff.shape)
            st.image(pic, width=pic_w)
            st.caption("Blue = brighter now (where it went). White = fainter now (where it was). "
                       "Fixed stars cancel to black. A mover shows as a white spot next to a blue one.")
            if not movers:
                st.write("No white-and-blue pair stands out between these two passes.")

with tab_layers:
    if n_periods < 2:
        st.info("The layers view needs at least two passes in this colour. Pick another colour in the sidebar.")
    else:
        # a slider needs a range, so with exactly 2 passes we just use both
        how_many = st.slider("Passes to stack (newest ones)", 2, n_periods, n_periods) if n_periods > 2 else 2
        first_pass = n_periods - how_many  # take the newest `how_many` passes
        cutoff = st.slider("Strictness (higher = fewer, surer dots)", 3.0, 8.0, 5.0, 0.5, key="layers_cutoff",
                           help="A spot must be this many times stronger than the random noise to be drawn.")
        smooth_layers = st.toggle("Smooth the picture", value=True, key="smooth_layers")
        st.image(sp.stack_layers(period_images[first_pass:], smooth=smooth_layers, blur=blur, cutoff=cutoff, width=pic_w), width=pic_w,
                 caption=f"{how_many} passes stacked | {period_dates[first_pass]} to {period_dates[-1]}")
        st.caption("White = oldest pass, blue = newest pass. Fixed stars are subtracted, "
                   "so they vanish. A mover leaves a trail of dots that turns from white to blue.")

with tab_colour:
    all_periods = sp.assign_periods(mjds)
    n_all = int(all_periods.max()) + 1
    all_dates = [nice_date(mjds[all_periods == k].mean()) for k in range(n_all)]
    which = st.selectbox("Pass", range(n_all), format_func=lambda k: all_dates[k], key="colour_pass")
    members = np.nonzero(all_periods == which)[0]
    if len(members) < 3:
        st.info("This pass has fewer than 3 pictures, so I can't make a colour picture.")
    else:
        lined = sp.line_up([pictures[i]["image"] for i in members], [pictures[i]["header"] for i in members])
        lined = sp.crop_shape(np.array([sp.remove_background(f) for f in lined]), SHAPES[shape_name])
        channels, ranges = sp.colour_composite(lined, colours[members])
        smooth_c = st.toggle("Smooth the picture", value=True, key="smooth_colour")
        st.image(sp.render_rgb(channels, width=pic_w, smooth=smooth_c, blur=blur), width=pic_w,
                 caption=f"Infrared colour picture | {all_dates[which]} | {len(members)} pictures")
        st.caption(
            f"FALSE COLOUR. Blue = shortest wavelengths ({ranges[0][0]:.2f} to {ranges[0][1]:.2f} micrometres), "
            f"green = middle ({ranges[1][0]:.2f} to {ranges[1][1]:.2f}), "
            f"red = longest ({ranges[2][0]:.2f} to {ranges[2][1]:.2f}). "
            "Infrared cannot be seen by eye, so these are visible colours painted on. "
            "Red things are brighter at longer wavelengths. "
            "These pictures were taken hours to days apart, so something that moves can show coloured fringes. "
            "Use the Difference tab to decide what is a mover.")
