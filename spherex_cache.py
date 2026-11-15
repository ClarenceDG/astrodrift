"""The slow steps of Astrodrift, each remembered with @st.cache_data.

Streamlit re-runs the whole script every time you touch a slider. Without a memory, that would download,
line up and combine the pictures again and again. With @st.cache_data, a function runs once for a given set of
inputs (the "key") and every later call with the same inputs just returns the saved answer.

Every function takes `search` = (ra, dec, patch size in arcmin, detector). That is the main part of every key.
Each one prints a line in the terminal when it really has to work, so you can see what was computed and how long it took.
"""
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import streamlit as st
from astropy.time import Time

import spherex_fetch
import spherex_follow as follow
import spherex_process as sp
import spherex_stack as stk

TTL = 3600  # forget saved answers after an hour (NASA may have new pictures)


@contextmanager
def timed(label):
    """Print how long the code inside the `with` took, in the terminal where Streamlit runs."""
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[astrodrift] {label}: {time.perf_counter() - start:.3f} s", flush=True)


def nice_date(mjd):
    return Time(float(mjd), format="mjd").to_datetime().strftime("%d %b %Y")


# ---------- step 1: download ----------
# The download is split in two cached steps so that the app can show a progress bar:
#   cutout_rows   = ask the archive which pictures exist (fast)
#   cutout_batch  = download 8 of them at once (slow). Each batch is remembered by itself.
BATCH = 8


@st.cache_data(ttl=TTL, show_spinner=False)
def cutout_rows(ra, dec, size_deg, detector):
    with timed("COMPUTED archive question (which pictures exist)"):
        return dict(rows=spherex_fetch.cutout_rows(ra, dec, size_deg, detector), asked_at=datetime.now())


@st.cache_data(ttl=TTL, show_spinner=False)
def cutout_batch(rows, ra, dec):
    with timed(f"COMPUTED download of {len(rows)} cutouts"):
        return spherex_fetch.download_batch(list(rows), ra, dec)


def _key(search):
    """The numbers that identify a search for the download steps."""
    ra, dec, size_arcmin, detector = search
    return float(ra), float(dec), size_arcmin / 60, detector


def ensure_pictures(search):
    """Download every picture of a search, showing a progress bar if it really has to download.

    Raises NoDataError (nothing there) or DownloadRefused (the archive would not give the cutouts, for
    example because the patch is too big). Everything is cached, so asking again is instant."""
    ra, dec, size_deg, detector = _key(search)
    first_time = search not in st.session_state.setdefault("downloaded", set())
    if first_time:
        # Asking the archive can take 10 to 45 seconds, so say what is happening (no silent waiting)
        with st.spinner(f"Asking NASA's archive which {search[2]} arcmin pictures exist (this can take up to a minute)..."):
            rows = cutout_rows(ra, dec, size_deg, detector)["rows"]
    else:
        rows = cutout_rows(ra, dec, size_deg, detector)["rows"]
    batches = [tuple(rows[i:i + BATCH]) for i in range(0, len(rows), BATCH)]
    bar = st.progress(0.0, text=f"Downloading {search[2]} arcmin pictures from NASA...") if first_time else None
    n_ok = 0
    for k, batch in enumerate(batches):
        n_ok += sum(r is not None for r in cutout_batch(batch, ra, dec))
        if bar is not None:
            bar.progress((k + 1) / len(batches),
                         text=f"Downloading {search[2]} arcmin pictures from NASA: {min((k + 1) * BATCH, len(rows))} of {len(rows)}")
    if bar is not None:
        bar.empty()
    if n_ok == 0:
        raise spherex_fetch.DownloadRefused("Found pictures but could not download them. Check your internet.")
    st.session_state["downloaded"].add(search)


def get_pictures(ra, dec, size_deg, detector):
    """All the pictures of a search, put together from the remembered batches. Returns (pictures, time of the archive question).
    Not cached itself: the batches already are, and this way the pictures are stored only once."""
    found = cutout_rows(ra, dec, size_deg, detector)
    rows = found["rows"]
    pictures = []
    for i in range(0, len(rows), BATCH):
        pictures += [r for r in cutout_batch(tuple(rows[i:i + BATCH]), ra, dec) if r is not None]
    if not pictures:
        raise spherex_fetch.DownloadRefused("Found pictures but could not download them. Check your internet.")
    return pictures, found["asked_at"]


def _pictures(search):
    return get_pictures(*_key(search))


@st.cache_data(ttl=TTL)
def fetch_info(search):
    """How many pictures NASA gave us and when. Also raises the 'no data' / 'bad coordinates' errors."""
    pictures, fetched_at = _pictures(search)
    return len(pictures), fetched_at


# ---------- step 2: sort the pictures by colour and by pass ----------
@st.cache_data(ttl=TTL)
def colour_overview(search, tolerance):
    with timed("COMPUTED colour groups"):
        pictures, _ = _pictures(search)
        mjds = np.array([p["mjd"] for p in pictures])
        colours = np.array([p["wavelength"] for p in pictures])
        groups = sp.group_by_colour(colours, tolerance)
        passes_per_group = [int(sp.assign_periods(mjds[g]).max()) + 1 for g in groups]
        best = max(range(len(groups)), key=lambda i: (passes_per_group[i], len(groups[i])))
        # passes over ALL pictures (the Colour tab uses these)
        all_periods = sp.assign_periods(mjds)
        n_all = int(all_periods.max()) + 1
        all_dates = [nice_date(mjds[all_periods == k].mean()) for k in range(n_all)]
        return dict(mjds=mjds, colours=colours, groups=groups, passes_per_group=passes_per_group, best=best,
                    all_periods=all_periods, n_all=n_all, all_dates=all_dates)


# ---------- step 3: line up, clean and combine the pictures of the chosen colour ----------
@st.cache_data(ttl=TTL)
def prepared(search, tolerance, choice):
    with timed("COMPUTED line up + remove background + combine passes"):
        pictures, _ = _pictures(search)
        ov = colour_overview(search, tolerance)
        mjds, colours = ov["mjds"], ov["colours"]
        chosen = sorted(ov["groups"][choice], key=lambda i: mjds[i])

        images = [pictures[i]["image"] for i in chosen]
        frames = sp.line_up(images, [pictures[i]["header"] for i in chosen])
        frames = np.array([sp.remove_background(f) for f in frames])
        lo, hi = sp.brightness_limits(frames)
        # all pictures are drawn on the grid of the biggest one; its header says where every pixel is on the sky
        grid_header = pictures[chosen[sp.reference_index(images)]]["header"]
        sky_ra, sky_dec = sp.sky_grid(grid_header, frames.shape[1:])

        times = mjds[chosen]
        periods = sp.assign_periods(times)
        n_periods = int(periods.max()) + 1
        period_images = [sp.combine(frames[periods == k]) for k in range(n_periods)]
        period_dates = [nice_date(times[periods == k].mean()) for k in range(n_periods)]
        period_mjds = [float(times[periods == k].mean()) for k in range(n_periods)]
        return dict(frames=frames, lo=lo, hi=hi, sky_ra=sky_ra, sky_dec=sky_dec,
                    frame_dates=[nice_date(t) for t in times], n_periods=n_periods,
                    period_images=period_images, period_dates=period_dates, period_mjds=period_mjds,
                    n_chosen=len(chosen), mean_colour=float(colours[chosen].mean()),
                    first_date=nice_date(mjds[chosen[0]]), last_date=nice_date(mjds[chosen[-1]]))


# ---------- step 4: one function per tab, run only when that tab needs it ----------
@st.cache_data(ttl=TTL)
def pair_candidates(search, tolerance, choice, i, j):
    """The loose candidates and the difference picture for one pair of passes (i = earlier, j = later)."""
    with timed(f"COMPUTED difference of passes {i} and {j}"):
        images = prepared(search, tolerance, choice)["period_images"]
        return sp.pair_candidates(images[i], images[j])


@st.cache_data(ttl=TTL)
def difference(search, tolerance, choice, first, second):
    """Difference tab: match brightness, subtract, and keep only the movers that pass the strict checks.

    A candidate can be 'confirmed' by the OTHER pairs of passes, so we also look at every other pair."""
    with timed("COMPUTED strict mover finder"):
        n_periods = prepared(search, tolerance, choice)["n_periods"]
        this = pair_candidates(search, tolerance, choice, first, second)
        others = [pair_candidates(search, tolerance, choice, i, j)
                  for i in range(n_periods) for j in range(i + 1, n_periods) if {i, j} != {first, second}]
        movers = sp.strict_movers(this, others)
        return dict(diff=this["diff"], scale=this["scale"], movers=movers, n_loose=len(this["movers"]),
                    rgb=sp.render_difference(this["diff"], this["sigma"]))


@st.cache_data(ttl=TTL)
def layers(search, tolerance, choice, how_many, cutoff):
    """Layers tab: stack the newest `how_many` passes."""
    with timed("COMPUTED layers"):
        images = prepared(search, tolerance, choice)["period_images"]
        rgb, strongest = sp.stack_layers(images[len(images) - how_many:], cutoff=cutoff)
        return dict(rgb=rgb, strongest=strongest)


@st.cache_data(ttl=TTL)
def stack(search, tolerance, choice, remove_fixed):
    """Stack tab: one see-through picture per pass, plus the mover tracks (same colour group, line-up,
    background removal and brightness matching as the Difference tab)."""
    with timed("COMPUTED stack layers + mover tracks"):
        prep = prepared(search, tolerance, choice)
        return stk.build(prep["period_images"], remove_fixed, prep["lo"], prep["hi"])


@st.cache_data(ttl=TTL)
def colour_picture(search, which):
    """Colour tab: the false-colour picture of one pass (needs 3 or more pictures in that pass)."""
    with timed("COMPUTED colour composite"):
        pictures, _ = _pictures(search)
        mjds = np.array([p["mjd"] for p in pictures])
        colours = np.array([p["wavelength"] for p in pictures])
        members = np.nonzero(sp.assign_periods(mjds) == which)[0]
        if len(members) < 3:
            return dict(n_members=len(members))
        member_images = [pictures[i]["image"] for i in members]
        lined = sp.line_up(member_images, [pictures[i]["header"] for i in members])
        lined = np.array([sp.remove_background(f) for f in lined])
        channels, ranges = sp.colour_composite(lined, colours[members])
        ra, dec = sp.sky_grid(pictures[members[sp.reference_index(member_images)]]["header"], lined.shape[1:])
        return dict(n_members=len(members), rgb=sp.render_rgb(channels), brightness=np.nanmean(channels, axis=0),
                    ranges=ranges, ra=ra, dec=dec)


# ---------- Follow a moving object ----------
@st.cache_data(ttl=TTL, show_spinner="Asking JPL Horizons and the Minor Planet Center for the predicted path...")
def follow_path(designation, start_mjd, days):
    """The predicted path of a named object (Horizons first, then the MPC). Raises EphemerisError if both fail."""
    with timed("COMPUTED predicted path"):
        return follow.get_predicted_path(designation, start_mjd, start_mjd + days)


@st.cache_data(ttl=TTL, show_spinner=False)
def follow_day(eph_key, _eph, day_mjd, size_arcmin):
    """All SPHEREx pictures of one day, downloaded around the predicted position. `eph_key` names the path
    (the path itself, _eph, is not hashed: that would be slow)."""
    with timed(f"COMPUTED follow: find + download pictures of day {day_mjd:.0f}"):
        rows = follow.find_exposures(_eph, day_mjd)
        return follow.download_tracked(_eph, rows, size_arcmin)


@st.cache_data(ttl=TTL)
def follow_frames(run_key, _pictures):
    """Line every tracked picture up on the object, and measure the brightness at the centre of each."""
    with timed("COMPUTED follow: line up tracked pictures + measure spectrum"):
        pictures = sorted(_pictures, key=lambda p: p["mjd"])
        aligned = [follow.align_on_object(p) for p in pictures]
        c = follow.CENTRE
        keep = [i for i, a in enumerate(aligned) if np.isfinite(a[0][c - 2:c + 3, c - 2:c + 3]).mean() >= 0.5]
        pictures, aligned = [pictures[i] for i in keep], [aligned[i] for i in keep]
        if not aligned:
            return dict(n=0)
        frames = np.array([a[0] for a in aligned])
        variances = [a[1] for a in aligned]
        wavelengths = np.array([p["wavelength"] for p in pictures])
        cleaned = np.array([f - np.nanmedian(f) for f in frames])
        lo, hi = np.nanpercentile(cleaned, [50, 99.8])
        wl, flux, error = follow.measure_spectrum(frames, variances, wavelengths)
        return dict(n=len(frames), cleaned=cleaned, ra=np.array([a[2] for a in aligned]), dec=np.array([a[3] for a in aligned]),
                    mjds=np.array([p["mjd"] for p in pictures]), wavelengths=wavelengths,
                    dates=[nice_date(p["mjd"]) for p in pictures], lo=float(lo), hi=float(hi),
                    spectrum=(wl, flux, error), unit=str(pictures[0]["header"].get("BUNIT", "")), frames=frames)


@st.cache_data(ttl=TTL)
def follow_stack(run_key, members, _frames):
    """Median of the tracked pictures of one colour group, and how many sigmas the middle sticks out."""
    with timed("COMPUTED follow: stack + significance"):
        stack = follow.stack_median(_frames["frames"][list(members)])
        return dict(stack=stack, significance=follow.detection_significance(stack))
