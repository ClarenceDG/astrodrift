"""Show a picture as an interactive Plotly figure: scroll to zoom, drag to pan, double-click to reset."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import spherex_process as sp

PICTURE_PX = 560  # the picture is always drawn this many pixels wide
BLUE_HEX = "#4DA6FF"

# Plotly's own settings: scroll wheel zooms, no Plotly logo, and the toolbar keeps only zoom in, zoom out and reset
CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtons": [["zoomIn2d", "zoomOut2d", "resetScale2d"]],
}


def hover_texts(ra, dec, value, value_name):
    """One line of text for every pixel: its sky position (RA, Dec) and its brightness number."""
    return [[f"RA {r:.5f}°<br>Dec {d:.5f}°<br>{value_name} {v:.3g}"
             for r, d, v in zip(ra_row, dec_row, value_row)]
            for ra_row, dec_row, value_row in zip(ra, dec, value)]


def ring_shapes(mover, radius=1.6):
    """Two rings for the mover finder: white where it WAS, blue where it is NOW.
    They are Plotly shapes in picture-pixel coordinates, so they stay on the right spot when you zoom or pan."""
    shapes = []
    for (x, y), colour in ((mover["old"], "#FFFFFF"), (mover["new"], BLUE_HEX)):
        shapes.append(dict(type="circle", xref="x", yref="y",
                           x0=x - radius, x1=x + radius, y0=y - radius, y1=y + radius,
                           line=dict(color=colour, width=2)))
    return shapes


def crosshair_shapes(cx, cy, gap=1.5, arm=4.0):
    """A thin white crosshair with a gap in the middle (so it does not hide the object) at pixel (cx, cy)."""
    line = dict(color="#FFFFFF", width=1)
    return [dict(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1, line=line)
            for x0, x1, y0, y1 in ((cx - gap - arm, cx - gap, cy, cy), (cx + gap, cx + gap + arm, cy, cy),
                                   (cx, cx, cy - gap - arm, cy - gap), (cx, cx, cy + gap, cy + gap + arm))]


def make_figure(rgb, ra, dec, value, value_name="brightness", smooth=True, blur=0.0, shapes=()):
    """Build the figure from a colour picture (rows x columns x 3 numbers) plus the numbers shown on hover."""
    h, w = rgb.shape[:2]

    if smooth:
        # ON: blur a little, then blow the picture up with a smooth (bicubic) stretch, so it has no hard squares
        factor = max(2, round(640 / max(h, w)))
        big = np.array(Image.fromarray(sp.soften(rgb, blur)).resize((w * factor, h * factor), Image.BICUBIC))
        step = 1 / factor
        first = -0.5 + step / 2  # the centre of the first small block, so the picture lines up with the pixel grid
    else:
        # OFF: the real pixels, shown as sharp squares (nearest neighbour)
        big, step, first = rgb, 1, 0

    fig = go.Figure()
    # trace 1: the picture itself
    fig.add_trace(go.Image(z=big, x0=first, dx=step, y0=first, dy=step, hoverinfo="skip"))
    # trace 2: an invisible layer, one cell per REAL pixel, that only exists to show the hover text
    fig.add_trace(go.Heatmap(
        z=np.nan_to_num(value), x=np.arange(w), y=np.arange(h),
        text=hover_texts(ra, dec, np.nan_to_num(value), value_name),
        hovertemplate="%{text}<extra></extra>", showscale=False, opacity=0, zsmooth=False))

    fig.update_layout(
        width=PICTURE_PX, height=round(PICTURE_PX * h / w),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        dragmode="pan",
        # no axis labels or ticks; y runs downwards like a picture, and pixels stay square
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", constrain="domain"),
        shapes=list(shapes),
        # "keep" = when the picture changes (next flipbook frame), stay at the same zoom and position
        uirevision="keep",
        hoverlabel=dict(bgcolor="#0B0B0D", bordercolor="#2A2A30",
                        font=dict(color="#FFFFFF", family="JetBrains Mono, monospace", size=12)),
    )
    return fig


@st.cache_data(max_entries=40)  # each saved picture is about 1.5 MB, so keep only the 40 newest
def cached_figure(picture_id, _rgb, _ra, _dec, _value, value_name, smooth, blur, rings, cross=None):
    """Build the figure once per picture. The cache key is only `picture_id` (a text that names the picture, e.g.
    'flip|search|colour|frame 2|Blue') plus the smooth/blur settings. The names starting with _ are the big
    arrays: Streamlit does not hash them (that would be slow), it trusts `picture_id` to say which picture it is."""
    shapes = list(ring_shapes({"old": rings[0], "new": rings[1]})) if rings else []
    if cross:
        shapes += crosshair_shapes(*cross)
    return make_figure(_rgb, _ra, _dec, _value, value_name=value_name, smooth=smooth, blur=blur, shapes=shapes)


def show(picture_id, rgb, ra, dec, value, key, rings=None, value_name="brightness", smooth=True, blur=0.0, cross=None):
    """Draw the picture on the page. `picture_id` names this exact picture (for the cache), `key` is the chart's
    identity on the page (so it keeps its zoom). `rings` = ((old x, old y), (new x, new y)) or None."""
    fig = cached_figure(picture_id, rgb, ra, dec, value, value_name, smooth, blur, rings, cross)
    st.plotly_chart(fig, width="content", theme=None, config=CONFIG, key=key)
    st.caption("Scroll to zoom, drag to move, double-click to reset. Hover to read the sky position and brightness.")


def spectrum_figure(wavelength, flux, error, unit=""):
    """Brightness against wavelength: blue points with error bars on black."""
    fig = go.Figure(go.Scatter(
        x=wavelength, y=flux, mode="markers", marker=dict(color=BLUE_HEX, size=8),
        error_y=dict(type="data", array=error, color=BLUE_HEX, thickness=1.2, width=3),
        customdata=error, hovertemplate="%{x:.2f} µm<br>brightness %{y:.3g} ± %{customdata:.2g}<extra></extra>"))
    axis = dict(gridcolor="#1E1E22", linecolor="#2A2A30", zerolinecolor="#2A2A30")
    fig.update_layout(
        height=340, margin=dict(l=70, r=10, t=10, b=55),
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(color="#C9C9D0", family="Inter, sans-serif", size=12),
        xaxis=dict(title="Wavelength (µm)", **axis),
        yaxis=dict(title=f"Brightness in the circle ({unit or 'summed'})", **axis),
        hoverlabel=dict(bgcolor="#0B0B0D", bordercolor="#2A2A30", font=dict(color="#FFFFFF", size=12)))
    return fig
