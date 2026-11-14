"""Astrodrift: type a spot in the sky, see what changes between NASA SPHEREx pictures."""
import pandas as pd
import streamlit as st
from astropy.time import Time

import spherex_fetch

st.set_page_config(page_title="Astrodrift", layout="wide")

st.title("Astrodrift")
st.caption("Type a spot in the sky. See what moves between NASA SPHEREx pictures.")


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

if go:
    try:
        pictures = get_pictures(float(ra), float(dec), size_arcmin / 60, detector)
    except spherex_fetch.NoDataError as problem:
        st.warning(str(problem))
        st.stop()
    except ValueError:
        st.error("I couldn't read those coordinates. Use numbers like 133.795 and -7.245.")
        st.stop()

    st.success(f"Found {len(pictures)} pictures of this spot.")
    table = pd.DataFrame({
        "Date": [Time(p["mjd"], format="mjd").to_datetime().strftime("%d %b %Y") for p in pictures],
        "Colour (micrometres)": [round(p["wavelength"], 3) for p in pictures],
        "Size (pixels)": [f"{p['image'].shape[1]} x {p['image'].shape[0]}" for p in pictures],
    })
    st.dataframe(table, width="stretch")
else:
    st.write("Pick a spot on the left and press **Show me the sky**.")
