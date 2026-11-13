"""Astrodrift: type a spot in the sky, see what changes between NASA SPHEREx pictures."""
import streamlit as st

st.set_page_config(page_title="Astrodrift", layout="wide")

st.title("Astrodrift")
st.caption("Type a spot in the sky. See what moves between NASA SPHEREx pictures.")

with st.sidebar:
    st.header("Where to look")
    # WISE 0855: the challenge's own test target, converted from hours to degrees
    ra = st.text_input("Right ascension (degrees)", "133.79513")
    dec = st.text_input("Declination (degrees)", "-7.24515")
    go = st.button("Show me the sky", type="primary")

if go:
    st.write(f"You asked for RA {ra}, Dec {dec}. The pictures come in the next step.")
else:
    st.write("Pick a spot on the left and press **Show me the sky**.")
