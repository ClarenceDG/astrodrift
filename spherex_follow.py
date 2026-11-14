"""Follow a known solar-system object across SPHEREx pictures.

The steps, in order:
  1. Get the predicted path (an "ephemeris": a table of where the object is at what time).
  2. For every day, ask NASA's IRSA archive which SPHEREx pictures cover the predicted position.
  3. Download a small cutout of each picture, centred on where the object should be AT THAT PICTURE'S TIME.
  4. Line the cutouts up (north up, same pixel size), so the object is in the middle of every one.
  5. Measure the brightness in a small circle at the centre. Stack the cutouts and measure how many
     "sigmas" (noise sizes) the centre sticks out above the noise.
This file has no Streamlit code in it: it is plain functions, so they are easy to test.
"""
import concurrent.futures
import re

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.interpolate import CubicSpline

import spherex_fetch
import spherex_process as sp

SPHEREX_HORIZONS_ID = "-163182"   # SPHEREx itself, as an "observer" in JPL Horizons
CUTOUT_ARCMIN = 4.0               # size of each downloaded cutout
PIXEL_ARCSEC = 6.15               # SPHEREx pixel size
FRAME = 41                        # lined-up pictures are FRAME x FRAME pixels; the object sits in the middle one
CENTRE = FRAME // 2
APERTURE_R = 2.0                  # the small circle (in pixels) that we measure the object in
RING_IN, RING_OUT = 4.0, 7.0      # the ring around it that tells us the local background
DETECTION_SIGMA = 5.0             # we only say "detected" at or above this many sigmas
MAX_PICTURES_PER_DAY = 60


class EphemerisError(Exception):
    """Raised when we cannot get a predicted path."""


# ---------- 1. the predicted path ----------
class Ephemeris:
    """Where an object is at different times. `at()` fills in the times between the table rows."""

    def __init__(self, mjd, ra, dec, source):
        mjd, ra, dec = (np.asarray(x, dtype=float) for x in (mjd, ra, dec))
        order = np.argsort(mjd)
        mjd, ra, dec = mjd[order], ra[order], dec[order]
        keep = np.concatenate([[True], np.diff(mjd) > 0])  # times must strictly increase
        self.mjd, self.dec = mjd[keep], dec[keep]
        # RA jumps from 359.9 to 0.1 at the wrap-around; 'unwrap' removes the jump so we can interpolate smoothly
        self.ra = np.rad2deg(np.unwrap(np.deg2rad(ra[keep])))
        self.source = source
        if len(self.mjd) < 2:
            raise EphemerisError("The path needs at least 2 rows.")

    @property
    def start(self):
        return float(self.mjd[0])

    @property
    def stop(self):
        return float(self.mjd[-1])

    def covers(self, mjd):
        return self.start - 1e-6 <= mjd <= self.stop + 1e-6

    def at(self, mjd):
        """RA and Dec (degrees) at the given time(s). Smooth curve through the rows (straight line if under 4 rows)."""
        mjd = np.atleast_1d(np.asarray(mjd, dtype=float))
        if mjd.min() < self.start - 1e-6 or mjd.max() > self.stop + 1e-6:
            raise EphemerisError("That time is outside the predicted path.")
        mjd = np.clip(mjd, self.start, self.stop)
        if len(self.mjd) >= 4:
            ra, dec = CubicSpline(self.mjd, self.ra)(mjd), CubicSpline(self.mjd, self.dec)(mjd)
        else:
            ra, dec = np.interp(mjd, self.mjd, self.ra), np.interp(mjd, self.mjd, self.dec)
        return ra % 360.0, dec


_ROW = re.compile(
    r"^\s*(?P<y>\d{4})\s+(?P<mo>\d{1,2})\s+(?P<d>\d{1,2})(?:\s+(?P<hms>\d{4,6}))?"
    r"\s+(?P<rh>\d{1,2})\s+(?P<rm>\d{1,2})\s+(?P<rs>\d{1,2}(?:\.\d*)?)"
    r"\s+(?P<dd>[+-]?\d{1,2})\s+(?P<dm>\d{1,2})\s+(?P<ds>\d{1,2}(?:\.\d*)?)")


def parse_mpc_table(text):
    """Read a pasted MPC ephemeris: date as `YYYY MM DD HHMMSS`, then RA as `h m s` and Dec as `d m s`.
    Lines that do not look like that (headers, notes) are skipped."""
    mjds, ras, decs = [], [], []
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        hms = (m["hms"] or "000000").ljust(6, "0")
        hour, minute, second = int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
        when = Time(f"{m['y']}-{int(m['mo']):02d}-{int(m['d']):02d}T{hour:02d}:{minute:02d}:{second:02d}", scale="utc")
        ra = 15.0 * (int(m["rh"]) + int(m["rm"]) / 60 + float(m["rs"]) / 3600)
        sign = -1.0 if m["dd"].strip().startswith("-") else 1.0  # "-00 30 00" is south of 0 even though -00 == 0
        dec = sign * (abs(int(m["dd"])) + int(m["dm"]) / 60 + float(m["ds"]) / 3600)
        mjds.append(when.mjd), ras.append(ra), decs.append(dec)
    if len(mjds) < 2:
        raise EphemerisError("I could not find at least 2 rows like `2026 01 01 000000 18 12 56.1 -42 17 07` in the text.")
    return Ephemeris(mjds, ras, decs, source="the table you pasted")


def horizons_ephemeris(designation, start_mjd, stop_mjd, step="2h"):
    """Ask JPL Horizons. We try SPHEREx as the observer first (so even close objects are right), then Earth's centre."""
    from astroquery.jplhorizons import Horizons
    epochs = {"start": Time(start_mjd, format="mjd").iso[:16], "stop": Time(stop_mjd, format="mjd").iso[:16], "step": step}
    last_problem = None
    for location, where in ((f"@{SPHEREX_HORIZONS_ID}", "the SPHEREx spacecraft"), ("500", "Earth's centre")):
        try:
            table = Horizons(id=designation, id_type="smallbody", location=location, epochs=epochs).ephemerides()
            return Ephemeris(np.array(table["datetime_jd"], dtype=float) - 2400000.5, np.array(table["RA"], dtype=float),
                             np.array(table["DEC"], dtype=float), source=f"JPL Horizons, seen from {where}")
        except Exception as problem:  # the network, an unknown name, an ambiguous name...
            last_problem = problem
    raise EphemerisError(f"JPL Horizons: {str(last_problem)[:200]}")


def mpc_ephemeris(designation, start_mjd, stop_mjd, step_hours=2):
    """Ask the Minor Planet Center. Used if Horizons fails."""
    from astroquery.mpc import MPC
    number = int((stop_mjd - start_mjd) * 24 / step_hours) + 1
    try:
        table = MPC.get_ephemeris(designation, location="500", start=Time(start_mjd, format="mjd"),
                                  step=step_hours * u.hour, number=number)
        dates = Time(table["Date"]) if not isinstance(table["Date"], Time) else table["Date"]
        ra = u.Quantity(table["RA"]).to_value(u.deg)
        dec = u.Quantity(table["Dec"]).to_value(u.deg)
        return Ephemeris(dates.mjd, ra, dec, source="the Minor Planet Center (Earth's centre)")
    except Exception as problem:
        raise EphemerisError(f"Minor Planet Center: {str(problem)[:200]}")


def get_predicted_path(designation, start_mjd, stop_mjd):
    """Try Horizons, then the MPC. Raises EphemerisError (with both reasons) if both fail."""
    problems = []
    for ask in (horizons_ephemeris, mpc_ephemeris):
        try:
            return ask(designation, start_mjd - 1, stop_mjd + 1)  # one extra day each side, so the ends interpolate well
        except EphemerisError as problem:
            problems.append(str(problem))
    raise EphemerisError(" | ".join(problems))


# ---------- 2 and 3. find and download the pictures ----------
def find_exposures(eph, day_mjd):
    """Ask IRSA for SPHEREx pictures that cover the predicted position on one day (day_mjd = that day at 00:00 UT).

    Same kind of question as spherex_fetch.find_cutouts, with one change: the picture must start within that day
    (p.time_bounds_lower between the day's start and end). IRSA does not accept 'this OR that position' in one
    question, so a slow object gets one question (its position at midday). An object that moves more than 0.02
    degrees in a day gets three (start, middle, end of the day) and the answers are merged.
    """
    from astroquery.ipac.irsa import Irsa
    lo, hi = float(day_mjd), float(day_mjd) + 1.0
    probes = np.clip([(lo + hi) / 2], eph.start, eph.stop)
    ends_ra, ends_dec = eph.at(np.clip([lo, hi], eph.start, eph.stop))
    if np.hypot((ends_ra[1] - ends_ra[0]) * np.cos(np.deg2rad(ends_dec[0])), ends_dec[1] - ends_dec[0]) > 0.02:
        probes = np.clip([lo, (lo + hi) / 2, hi], eph.start, eph.stop)
    seen, rows = set(), []
    for ra, dec in zip(*eph.at(probes)):
        query = f"""
        SELECT a.uri AS uri, p.time_bounds_lower AS t_start, p.time_bounds_upper AS t_end
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE 1 = CONTAINS(POINT('ICRS', {ra:.6f}, {dec:.6f}), p.poly)
            AND p.time_bounds_lower BETWEEN {lo} AND {hi}
        ORDER BY p.time_bounds_lower
        """
        table = Irsa.query_tap(query).to_table()
        for uri, t_start, t_end in zip(table["uri"], table["t_start"], table["t_end"]):
            if str(uri) in seen:
                continue
            seen.add(str(uri))
            t_start = float(t_start)
            t_end = float(t_end) if np.isfinite(float(t_end)) else t_start
            rows.append({"uri": str(uri), "mjd": (t_start + t_end) / 2})  # the middle of the exposure
    return sorted(rows, key=lambda r: r["mjd"])[:MAX_PICTURES_PER_DAY]


def download_tracked(eph, rows, size_arcmin=CUTOUT_ARCMIN):
    """Download one cutout per row, each centred on where the object is predicted to be at that picture's own time."""
    size_deg = size_arcmin / 60

    def one(row):
        if not eph.covers(row["mjd"]):
            return None
        ra, dec = (float(x[0]) for x in eph.at(row["mjd"]))
        uri = f"https://irsa.ipac.caltech.edu/{row['uri']}?center={ra:.6f},{dec:.6f}d&size={size_deg}"
        result = spherex_fetch._download_one(uri, SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs"),
                                            keep_variance=True)
        if result is None:
            return None
        result.update(mjd=row["mjd"], ra=ra, dec=dec)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return [r for r in pool.map(one, rows) if r is not None]


# ---------- 4. line the cutouts up on the object ----------
def align_on_object(pic):
    """Redraw one cutout on a fresh north-up grid whose centre is the predicted position, so the object is in the
    middle pixel. Returns the picture, its variance (or None), and the RA and Dec of every pixel."""
    target = WCS(naxis=2)
    target.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    target.wcs.crval = [pic["ra"], pic["dec"]]
    target.wcs.crpix = [CENTRE + 1, CENTRE + 1]  # FITS counts pixels from 1
    target.wcs.cdelt = [-PIXEL_ARCSEC / 3600, PIXEL_ARCSEC / 3600]
    source_wcs = WCS(pic["header"])
    image, _ = reproject_interp((pic["image"], source_wcs), target, shape_out=(FRAME, FRAME))
    variance = None
    if pic.get("variance") is not None:
        variance, _ = reproject_interp((pic["variance"], source_wcs), target, shape_out=(FRAME, FRAME))
    rows, cols = np.mgrid[:FRAME, :FRAME]
    ra, dec = target.pixel_to_world_values(cols, rows)
    return image, variance, ra, dec


# ---------- 5. measure ----------
def _distance_from(centre):
    rows, cols = np.mgrid[:FRAME, :FRAME]
    return np.hypot(rows - centre[0], cols - centre[1])


def aperture_flux(image, variance=None, centre=(CENTRE, CENTRE)):
    """Brightness in a small circle minus the local background (the median of a ring around it).

    Returns (brightness, uncertainty). The uncertainty comes from the VARIANCE layer when we have it,
    otherwise from how much the ring's pixels vary. Returns (nan, nan) if too many pixels are missing."""
    dist = _distance_from(centre)
    circle = dist <= APERTURE_R
    ring = (dist >= RING_IN) & (dist <= RING_OUT)
    inside = image[circle]
    ring_values = image[ring][np.isfinite(image[ring])]
    if np.isfinite(inside).sum() < 0.7 * circle.sum() or len(ring_values) < 15:
        return np.nan, np.nan
    background = np.median(ring_values)
    flux = float(np.nansum(inside - background))
    n_used = int(np.isfinite(inside).sum())
    if variance is not None and np.isfinite(variance[circle]).sum() >= 0.7 * circle.sum():
        error = float(np.sqrt(np.nansum(variance[circle])))
    else:
        error = float(np.sqrt(n_used) * 1.4826 * np.median(np.abs(ring_values - background)))
    return flux, error


def measure_spectrum(frames, variances, wavelengths):
    """Brightness at the centre of every picture, for the spectrum plot. Returns arrays, only for pictures we could measure."""
    out = [(w, *aperture_flux(f, v)) for f, v, w in zip(frames, variances, wavelengths)]
    out = np.array(out, dtype=float).reshape(-1, 3)
    good = np.isfinite(out[:, 1]) & np.isfinite(out[:, 2])
    return out[good, 0], out[good, 1], out[good, 2]


def colour_groups(wavelengths, tolerance=0.05):
    """Pictures of about the same colour, as lists of picture numbers (biggest group first)."""
    groups = sp.group_by_colour(wavelengths, tolerance)
    return sorted(groups, key=len, reverse=True)


def stack_median(frames):
    """Median of the cutouts after removing each one's background. The object is in the middle of every cutout, so
    it adds up; stars are in different places in each cutout, so they smear out and the median drops them."""
    cleaned = np.array([f - np.nanmedian(f) for f in frames])
    return np.nanmedian(cleaned, axis=0)


def detection_significance(stack):
    """How many 'sigmas' the middle of the stack sticks out above the noise. Returns None if we cannot tell.

    The noise is measured on the picture itself: the same small circle is placed at many other spots (away from the
    middle) and we see how much those brightness values scatter. That scatter is what 'noise' means here.
    """
    flux, _ = aperture_flux(stack)
    if not np.isfinite(flux):
        return None
    margin = int(np.ceil(RING_OUT)) + 1
    spacing = int(2 * APERTURE_R) + 3
    others = []
    for row in range(margin, FRAME - margin, spacing):
        for col in range(margin, FRAME - margin, spacing):
            if np.hypot(row - CENTRE, col - CENTRE) < RING_OUT + APERTURE_R + 1:
                continue  # too close to the object: it would count the object itself as noise
            value, _ = aperture_flux(stack, centre=(row, col))
            if np.isfinite(value):
                others.append(value)
    if len(others) < 6:
        return None
    noise = 1.4826 * float(np.median(np.abs(np.array(others) - np.median(others))))
    if noise <= 0:
        return None
    return {"sigma": flux / noise, "flux": flux, "noise": noise, "n_spots": len(others)}


def status_line(significance, n_pictures, wavelength):
    """The honest sentence. It always contains the signal-to-noise number; with no number, no claim is made."""
    where = f"in the stack of {n_pictures} pictures near {wavelength:.2f} µm"
    if significance is None:
        return False, f"Could not measure how strong the signal is {where}, so no detection is claimed."
    sigma = significance["sigma"]
    if sigma >= DETECTION_SIGMA:
        return True, f"Detected at {sigma:.1f} sigma {where}."
    return False, f"Not detected, the object is too faint for these pictures ({sigma:.1f} sigma {where})."
