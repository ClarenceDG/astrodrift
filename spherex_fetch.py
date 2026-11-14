"""Ask NASA's IRSA archive for SPHEREx pictures of one patch of sky.

Based on the IRSA tutorial "Download a collection of SPHEREx Spectral Image cutouts"
(Desai, Krick, Faisst, Raen, Sipocz; BSD 3-Clause).
"""
import concurrent.futures

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

# SPHEREx has six detector arrays. Each one sees a different range of infrared colours.
DETECTORS = [f"SPHEREx-D{i}" for i in range(1, 7)]


class NoDataError(Exception):
    """Raised when NASA has no pictures for this spot."""


class DownloadRefused(NoDataError):
    """Raised when NASA lists pictures for this spot but none of the cutouts could be downloaded
    (for example because the archive refuses a patch size that is too big)."""


def find_cutouts(ra_deg, dec_deg, size_deg, detector):
    """Step 1: ask the archive which pictures cover this spot. Returns a table of links and times."""
    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri || '?center={ra_deg},{dec_deg}d&size={size_deg}' AS uri,
        p.time_bounds_lower
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra_deg}, {dec_deg}), p.poly)
        AND p.energy_bandpassname = '{detector}'
    ORDER BY p.time_bounds_lower
    """
    return Irsa.query_tap(query).to_table()


def _download_one(uri, coord, keep_variance=False):
    """Step 2: download one small cutout and read its colour. Returns None if it fails.

    The picture is kept as float32 (half the memory of float64: a 60 arcmin patch is big).
    The variance layer (the noise per pixel) is only kept if keep_variance=True; the Follow tab needs it
    for error bars, the main pictures do not."""
    try:
        with fits.open(uri, cache=True) as hdulist:
            header = hdulist[1].header.copy()
            image = hdulist[1].data.astype(np.float32).copy()
            # The VARIANCE layer is the noise per pixel. Bad pixels have no usable noise value, so we blank them.
            variance = None
            try:
                variance = hdulist["VARIANCE"].data.astype(np.float32).copy()
                image[~np.isfinite(variance) | (variance <= 0)] = np.nan
            except (KeyError, AttributeError, ValueError):
                variance = None  # no variance layer in this file; keep the image as it is
            # work out the infrared colour (wavelength) at the target, as the IRSA tutorial does
            x, y = WCS(header).world_to_pixel(coord)
            colour_wcs = WCS(header, fobj=hdulist, key="W")
            colour_wcs.sip = None
            wavelength, _ = colour_wcs.pixel_to_world(x, y)
            return {"image": image, "header": header, "variance": variance if keep_variance else None,
                    "wavelength": float(wavelength.to(u.micrometer).value)}
    except Exception:
        return None


def cutout_rows(ra_deg, dec_deg, size_deg, detector, max_pictures=120):
    """Ask the archive which pictures cover this spot. Returns a list of (cutout link, date) pairs, oldest first."""
    table = find_cutouts(ra_deg, dec_deg, size_deg, detector)
    if len(table) == 0:
        raise NoDataError("NASA has no SPHEREx pictures of this spot for this detector yet.")
    table = table[:max_pictures]
    return [(str(uri), float(mjd)) for uri, mjd in zip(table["uri"], table["time_bounds_lower"])]


def download_batch(rows, ra_deg, dec_deg, workers=8):
    """Download several cutouts at once (several at a time is faster). Returns one entry per row:
    the picture, or None if that download failed. Each picture gets its date as `mjd`."""
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda row: _download_one(row[0], coord), rows))
    for result, (_, mjd) in zip(results, rows):
        if result is not None:
            result["mjd"] = float(mjd)  # date as a Modified Julian Date number
    return results


def fetch_pictures(ra_deg, dec_deg, size_deg, detector, max_pictures=120):
    """Find and download all the pictures of this spot for one detector (the whole job in one go)."""
    rows = cutout_rows(ra_deg, dec_deg, size_deg, detector, max_pictures)
    pictures = [r for r in download_batch(rows, ra_deg, dec_deg) if r is not None]
    if not pictures:
        raise DownloadRefused("Found pictures but could not download them. Check your internet.")
    return pictures
