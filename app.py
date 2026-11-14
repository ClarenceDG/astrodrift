"""Astrodrift: type a spot in the sky, see what changes between NASA SPHEREx pictures."""
import numpy as np
import streamlit as st
from astropy.time import Time

import spherex_fetch
import spherex_process as sp

st.set_page_config(page_title="Astrodrift", layout="wide")

st.title("Astrodrift")
st.caption("Type a spot in the sky. See what moves between NASA SPHEREx pictures.")


def nice_date(mjd):
    return Time(float(mjd), format="mjd").to_datetime().strftime("%d %b %Y")


@st.cache_data(show_spinner="Asking NASA's archive for pictures...", ttl=3600)
def get_pictures(ra, dec, size_deg, detector):
    return spherex_fetch.fetch_pictures(ra, dec, size_deg, detector)


with st.sidebar:
    st.header("Where to look")
    # WISE 0855: the challenge's own test target, converted from hours to degrees
    ra = st.text_input("Right ascension (degrees)", "133.79513")
    dec = st.text_input("Declination (degrees)", "-7.24515")
    size_arcmin = st.select_slider("Size of the patch (arcminutes)", options=[3, 6, 12], value=6)
    detector = st.selectbox("Detector", spherex_fetch.DETECTORS, index=1)
    go = st.button("Show me the sky", type="primary")

# remember the search, so moving the slider doesn't forget it
if go:
    st.session_state.search = (ra, dec, size_arcmin, detector)
if "search" not in st.session_state:
    st.write("Pick a spot on the left and press **Show me the sky**.")
    st.stop()

ra_s, dec_s, size_s, det_s = st.session_state.search
try:
    pictures = get_pictures(float(ra_s), float(dec_s), size_s / 60, det_s)
except spherex_fetch.NoDataError as problem:
    st.warning(str(problem))
    st.stop()
except ValueError:
    st.error("I couldn't read those coordinates. Use numbers like 133.795 and -7.245.")
    st.stop()

# group pictures by colour, then pick the colour seen in the most passes
mjds = np.array([p["mjd"] for p in pictures])
colours = np.array([p["wavelength"] for p in pictures])
groups = sp.group_by_colour(colours)
passes_per_group = [sp.assign_periods(mjds[g]).max() + 1 for g in groups]
best = max(range(len(groups)), key=lambda i: (passes_per_group[i], len(groups[i])))

choice = st.sidebar.selectbox(
    "Colour (micrometres)", range(len(groups)), index=best,
    format_func=lambda i: f"{colours[groups[i]].mean():.2f}  |  {len(groups[i])} pictures  |  {passes_per_group[i]} pass(es)",
)
chosen = sorted(groups[choice], key=lambda i: mjds[i])

frames = sp.line_up([pictures[i]["image"] for i in chosen], [pictures[i]["header"] for i in chosen])
frames = np.array([sp.remove_background(f) for f in frames])
lo, hi = sp.brightness_limits(frames)

st.caption(f"{len(chosen)} pictures at about {colours[chosen].mean():.2f} micrometres, "
           f"{passes_per_group[choice]} pass(es), {nice_date(mjds[chosen[0]])} to {nice_date(mjds[chosen[-1]])}.")

if len(chosen) > 1:
    n = st.slider("Picture", 1, len(chosen), 1)
else:
    n = 1
zoom = st.slider("Zoom (centre of the patch)", 1, 4, 1)
smooth = st.toggle("Smooth the picture", value=True,
                   help="Off shows the real pixels as sharp squares. On blends them.")
shown = sp.crop_centre(frames[n - 1], zoom)
st.image(sp.to_picture(shown, lo, hi, smooth=smooth), width=600,
         caption=f"Picture {n} of {len(chosen)} | {nice_date(mjds[chosen[n - 1]])}")
st.caption("Fixed stars stay put. Anything that moves jumps between pictures.")
