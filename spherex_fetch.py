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


def _download_one(uri, coord):
    """Step 2: download one small cutout and read its colour. Returns None if it fails."""
    try:
        with fits.open(uri, cache=True) as hdulist:
            header = hdulist[1].header.copy()
            image = hdulist[1].data.astype(float).copy()
            # The VARIANCE layer is the noise per pixel. Bad pixels have no usable noise value, so we blank them.
            variance = None
            try:
                variance = hdulist["VARIANCE"].data.astype(float).copy()
                image[~np.isfinite(variance) | (variance <= 0)] = np.nan
            except (KeyError, AttributeError, ValueError):
                variance = None  # no variance layer in this file; keep the image as it is
            # work out the infrared colour (wavelength) at the target, as the IRSA tutorial does
            x, y = WCS(header).world_to_pixel(coord)
            colour_wcs = WCS(header, fobj=hdulist, key="W")
            colour_wcs.sip = None
            wavelength, _ = colour_wcs.pixel_to_world(x, y)
            return {"image": image, "header": header, "variance": variance,
                    "wavelength": float(wavelength.to(u.micrometer).value)}
    except Exception:
        return None


def fetch_pictures(ra_deg, dec_deg, size_deg, detector, max_pictures=120):
    """Find and download all the pictures of this spot for one detector."""
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    table = find_cutouts(ra_deg, dec_deg, size_deg, detector)
    if len(table) == 0:
        raise NoDataError("NASA has no SPHEREx pictures of this spot for this detector yet.")
    table = table[:max_pictures]

    # download several at once so it is faster
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda uri: _download_one(uri, coord), table["uri"]))

    pictures = []
    for result, mjd in zip(results, table["time_bounds_lower"]):
        if result is not None:
            result["mjd"] = float(mjd)  # date as a Modified Julian Date number
            pictures.append(result)
    if not pictures:
        raise NoDataError("Found pictures but could not download them. Check your internet.")
    return pictures
