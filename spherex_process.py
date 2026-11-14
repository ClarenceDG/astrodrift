"""Turn the raw pictures into something we can compare: group, line up, clean, draw."""
import numpy as np
from astropy.wcs import WCS
from PIL import Image
from reproject import reproject_interp


def assign_periods(mjds, gap_days=60):
    """Number each picture by pass. A new pass starts after a gap of more than 60 days."""
    mjds = np.asarray(mjds, dtype=float)
    order = np.argsort(mjds)
    period = np.zeros(len(mjds), dtype=int)
    current = 0
    for prev, now in zip(order[:-1], order[1:]):
        if mjds[now] - mjds[prev] > gap_days:
            current += 1
        period[now] = current
    return period


def group_by_colour(wavelengths, tolerance=0.05):
    """Group pictures whose colour is within `tolerance` micrometres. Returns lists of picture numbers."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    groups = []
    for i in np.argsort(wavelengths):
        if groups and wavelengths[i] - wavelengths[groups[-1][0]] <= tolerance:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def line_up(images, headers):
    """Re-draw every picture onto the first picture's sky grid, so a fixed star sits on the same pixel."""
    target_wcs = WCS(headers[0])
    shape = images[0].shape
    lined = []
    for img, hdr in zip(images, headers):
        out, _ = reproject_interp((img, WCS(hdr)), target_wcs, shape_out=shape)
        lined.append(out)
    return np.array(lined)


def remove_background(img):
    """Subtract the typical (median) brightness so only stars and changes are left."""
    return img - np.nanmedian(img)


def brightness_limits(frames):
    """One brightness range for the whole flipbook, so pictures are comparable."""
    lo, hi = np.nanpercentile(frames, [5, 99.5])
    return lo, hi


def to_picture(img, lo, hi, width=600, smooth=True):
    """Make a smooth black-and-white picture, enlarged for the screen."""
    scaled = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    pic = Image.fromarray((scaled * 255).astype(np.uint8))
    return pic.resize((width, int(width * img.shape[0] / img.shape[1])), Image.BICUBIC if smooth else Image.NEAREST)


def crop_centre(img, zoom):
    """Keep only the middle part of the picture. zoom=2 keeps the middle half of the width and height."""
    h, w = img.shape
    ch, cw = h // zoom, w // zoom
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]
