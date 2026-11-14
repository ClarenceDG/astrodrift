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
with st.sidebar.expander("Advanced"):
    tolerance = st.slider("Colour match tolerance (micrometres)", 0.01, 0.10, 0.03, 0.01,
                          help="Smaller = only very similar colours are compared. Fewer pictures, fewer false differences.")
groups = sp.group_by_colour(colours, tolerance)
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

times = mjds[chosen]
periods = sp.assign_periods(times)
n_periods = int(periods.max()) + 1
period_images = [sp.combine(frames[periods == k]) for k in range(n_periods)]
period_dates = [nice_date(times[periods == k].mean()) for k in range(n_periods)]

tab_flip, tab_diff, tab_colour = st.tabs(["Flipbook", "Difference", "Colour"])

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
    st.image(sp.to_picture(shown, lo, hi, smooth=smooth, palette=palette), width=600,
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
            pic = sp.render_difference(diff, sigma, smooth=smooth_diff)
            if movers:
                chosen_mover = st.selectbox(
                    "Possible movers", range(len(movers)),
                    format_func=lambda k: f"#{k + 1}: {movers[k]['sep']:.1f} pixels apart, strength {movers[k]['score']:.0f}")
                pic = sp.draw_pair(pic, movers[chosen_mover], diff.shape)
            st.image(pic, width=600)
            st.caption("Amber = brighter now (where it went). White = fainter now (where it was). "
                       "Fixed stars cancel to black. A mover shows as a white spot next to an amber one.")
            if not movers:
                st.write("No white-and-amber pair stands out between these two passes.")

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
        lined = np.array([sp.remove_background(f) for f in lined])
        channels, ranges = sp.colour_composite(lined, colours[members])
        smooth_c = st.toggle("Smooth the picture", value=True, key="smooth_colour")
        st.image(sp.render_rgb(channels, smooth=smooth_c), width=600,
                 caption=f"Infrared colour picture | {all_dates[which]} | {len(members)} pictures")
        st.caption(
            f"FALSE COLOUR. Blue = shortest wavelengths ({ranges[0][0]:.2f} to {ranges[0][1]:.2f} micrometres), "
            f"green = middle ({ranges[1][0]:.2f} to {ranges[1][1]:.2f}), "
            f"red = longest ({ranges[2][0]:.2f} to {ranges[2][1]:.2f}). "
            "Infrared cannot be seen by eye, so these are visible colours painted on. "
            "Red things are brighter at longer wavelengths. "
            "These pictures were taken hours to days apart, so something that moves can show coloured fringes. "
            "Use the Difference tab to decide what is a mover.")
