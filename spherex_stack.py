"""The Stack tab's maths: see-through layers (one per pass) and the mover tracks that link a bright spot
from one pass to the next. Plain numpy, no Streamlit, so it can be tested on its own."""
import numpy as np
from scipy.ndimage import gaussian_filter

import spherex_process as sp

PEAK_SIGMA = 5.0     # a spot must be this many noise-sizes above the noise to count
EDGE_MARGIN = 5      # spots closer than this many pixels to the picture's edge are ignored
MIN_STEP = 1.5       # a mover moves at least this many pixels from one pass to the next ...
MAX_STEP = 12.0      # ... and at most this many
MAX_RATIO = 2.5      # and its brightness may change by at most this factor
MAX_TRACKS = 10      # show at most this many tracks, the strongest first


# ---------- layers ----------
def remove_fixed_things(pass_images):
    """Match the brightness of every pass to the first one, then subtract the median of all passes.
    The median is what never changes (fixed stars), so what is left is only what changed.
    Returns the matched passes, the median, and one 'what is left' picture per pass (blurred a little:
    noise is one pixel wide, real dots are not)."""
    matched = [pass_images[0]] + [sp.match_brightness(pass_images[0], p)[0] for p in pass_images[1:]]
    fixed = np.nanmedian(np.array(matched), axis=0)
    left = [gaussian_filter(np.nan_to_num(m - fixed), 1.0) for m in matched]
    return matched, fixed, left


def _own_scale(values):
    """Scale one layer so its bright end is 1 (the 99.5th percentile of its positive values) and the rest is 0..1."""
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros(values.shape)
    top = np.percentile(positive, 99.5) if positive.size >= 20 else positive.max()
    return np.clip(values / (top + 1e-12), 0, 1)


def build(pass_images, remove_fixed=True, lo=None, hi=None):
    """Everything the Stack tab needs: one RGBA picture per pass (a flat colour whose alpha follows brightness),
    the mover tracks, and the biggest change per pixel (shown on hover)."""
    matched, fixed, left = remove_fixed_things(pass_images)
    n = len(matched)
    sigmas = [sp.noise_level(x) for x in left]
    if remove_fixed:
        floors = [3 * s for s in sigmas]                                   # ignore anything under 3 noise-sizes
        # every layer gets its OWN brightness scale, so a dim pass is as bright on screen as the others
        scaled = [_own_scale(x - f) for x, f in zip(left, floors)]
        strongest = np.max(left, axis=0)
    else:
        if lo is None:
            lo, hi = np.nanpercentile(np.array(matched), [50, 99.8])
        scaled = [_own_scale(np.nan_to_num(m, nan=lo) - lo) for m in matched]
        strongest = np.nanmax(np.array(matched), axis=0)
    layers = []
    for i, x in enumerate(scaled):
        # a gentle curve (brightness to the power 0.6): faint stars stay visible, the brightest reach 0.9
        alpha = 0.9 * x ** 0.6
        rgba = np.empty(x.shape + (4,), np.uint8)
        rgba[..., :3] = sp.layer_colour(i, n)
        rgba[..., 3] = np.round(alpha * 255)
        layers.append(rgba)
    tracks = find_tracks(left, sigmas, fixed)  # tracks are always looked for with the fixed things removed
    return dict(layers=layers, tracks=tracks, strongest=strongest)


def composite(layers, on, opacity):
    """Mix the switched-on layers into one picture. Colours ADD (white + blue = pale blue), so overlapping layers
    mix instead of one covering the others. `opacity` = one number 0..1 per layer. A layer with opacity 0 or
    switched off adds nothing, and the other layers are not touched."""
    out = np.zeros(layers[0].shape[:2] + (3,))
    for rgba, is_on, op in zip(layers, on, opacity):
        if is_on and op > 0:
            out += rgba[..., :3] * (rgba[..., 3:4] / 255.0 * op)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- finding movers ----------
def _centre(signal, x, y):
    """The peak's position to a fraction of a pixel: the weighted average position in its 3x3 neighbourhood."""
    ys, xs = np.mgrid[y - 1:y + 2, x - 1:x + 2]
    w = np.clip(signal[y - 1:y + 2, x - 1:x + 2], 0, None)
    return float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())


def find_peaks(signal, sigma, fixed):
    """Bright spots in one layer: above PEAK_SIGMA, not near the edge, and not on a bright fixed star."""
    h, w = signal.shape
    fixed = np.nan_to_num(fixed)
    peaks = []
    for x, y, value in sp._peaks(signal, PEAK_SIGMA * sigma):
        if x < EDGE_MARGIN or y < EDGE_MARGIN or x >= w - EDGE_MARGIN or y >= h - EDGE_MARGIN:
            continue
        nearby = fixed[max(0, y - 2):y + 3, max(0, x - 2):x + 3].max()
        if nearby > sp.BRIGHT_STAR_SIGMA * sigma and nearby > value:
            # A bright fixed star that is much brighter than the spot itself: the spot is only a leftover of the
            # star. (If the spot is brighter than what is fixed there, the fixed part is just the tail of the
            # mover's own light seen in the neighbouring passes, so we keep it.)
            continue
        cx, cy = _centre(signal, x, y)
        peaks.append((cx, cy, value / sigma))
    return peaks


def link_peaks(peaks):
    """Join peaks of neighbouring passes into chains (one list of (layer, x, y, strength) per chain).
    Two peaks are linked when they are MIN_STEP to MAX_STEP pixels apart and their brightness differs by less than
    MAX_RATIO. The best matches are linked first, each peak is used once, and a chain keeps going in the same
    direction. Returns only chains that cover 2 or more passes."""
    chains = []
    for k in range(len(peaks) - 1):
        ends = [c for c in chains if c[-1][0] == k]
        used = {(c[-1][1], c[-1][2]) for c in ends}
        for x, y, s in peaks[k]:
            if (x, y) not in used:
                chain = [(k, x, y, s)]
                chains.append(chain)
                ends.append(chain)
        options = []
        for ci, chain in enumerate(ends):
            _, x0, y0, s0 = chain[-1]
            for qi, (x1, y1, s1) in enumerate(peaks[k + 1]):
                step = np.hypot(x1 - x0, y1 - y0)
                ratio = s1 / s0
                if not (MIN_STEP <= step <= MAX_STEP and 1 / MAX_RATIO <= ratio <= MAX_RATIO):
                    continue
                cost = abs(np.log(ratio))
                if len(chain) >= 2:
                    px, py = chain[-1][1] - chain[-2][1], chain[-1][2] - chain[-2][2]
                    if (x1 - x0) * px + (y1 - y0) * py <= 0:
                        continue  # it would turn around
                    cost += abs(step - np.hypot(px, py)) / MAX_STEP
                else:
                    cost += step / MAX_STEP / 4
                options.append((cost, ci, qi))
        taken_chain, taken_peak = set(), set()
        for cost, ci, qi in sorted(options):
            if ci in taken_chain or qi in taken_peak:
                continue
            taken_chain.add(ci)
            taken_peak.add(qi)
            x1, y1, s1 = peaks[k + 1][qi]
            ends[ci].append((k + 1, x1, y1, s1))
    return [c for c in chains if len(c) >= 2]


def find_tracks(left, sigmas, fixed):
    """Mover tracks, strongest first. Each is {'layers': [...], 'xy': [(x, y), ...], 'strength': ...}."""
    peaks = [find_peaks(x, s, fixed) for x, s in zip(left, sigmas)]
    tracks = [dict(layers=[p[0] for p in c], xy=[(p[1], p[2]) for p in c], strength=min(p[3] for p in c))
              for c in link_peaks(peaks)]
    return sorted(tracks, key=lambda t: -t["strength"])[:MAX_TRACKS]


# ---------- measuring a track ----------
def pixel_scale_arcsec(ra, dec):
    """Size of one pixel on the sky in arcseconds, measured in the middle of the picture from the WCS grid."""
    h, w = ra.shape
    y, x = h // 2, w // 2
    d_ra = (ra[y, x + 1] - ra[y, x]) * np.cos(np.radians(dec[y, x]))
    d_dec = dec[y, x + 1] - dec[y, x]
    return float(np.hypot(d_ra, d_dec) * 3600)


def measure(track, mjds, scale):
    """Path length and speed of one track. `mjds` = the date of every pass, `scale` = arcseconds per pixel."""
    steps = [float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(track["xy"], track["xy"][1:])]
    days = mjds[track["layers"][-1]] - mjds[track["layers"][0]]
    path = sum(steps)
    return dict(path_px=path, px_per_pass=path / len(steps), steps=steps,
                arcsec_per_day=path * scale / days if days > 0 else float("nan"))


def clicked_track(event):
    """The track number (1, 2, ...) of the point the user clicked on the figure, or None.
    Streamlit hands us the click as {'selection': {'points': [{'customdata': ...}, ...]}}."""
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None
    for p in points:
        value = p.get("customdata")
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None
