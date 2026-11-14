"""Turn the raw pictures into something we can compare: group, line up, clean, draw."""
import numpy as np
from PIL import ImageDraw
from scipy.ndimage import maximum_filter
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
    # one range for the whole flipbook, so pictures are comparable
    """One brightness range for the whole flipbook, so pictures are comparable."""
    lo, hi = np.nanpercentile(frames, [50, 99.8])
    return lo, hi


# Colour maps: a few (brightness, colour) stops, blended in between.
PALETTES = {
    "Amber": [(0, (0, 0, 0)), (0.45, (140, 80, 0)), (0.8, (255, 176, 0)), (1, (255, 245, 220))],
    "Viridis": [(0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
                (0.75, (94, 201, 98)), (1, (253, 231, 37))],
    "Greyscale": [(0, (0, 0, 0)), (1, (255, 255, 255))],
}


def apply_palette(scaled, palette):
    """Turn brightness numbers (0 to 1) into colours using the chosen colour map."""
    stops = PALETTES[palette]
    xs = [s[0] for s in stops]
    return np.stack([np.interp(scaled, xs, [s[1][c] for s in stops]) for c in range(3)], axis=-1).astype(np.uint8)


def to_picture(img, lo, hi, width=600, smooth=True, palette="Amber"):
    """Make a coloured picture. An asinh stretch shows faint stars without blowing out bright ones."""
    scaled = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    scaled = np.arcsinh(10 * scaled) / np.arcsinh(10)
    pic = Image.fromarray(apply_palette(scaled, palette))
    return pic.resize((width, int(width * img.shape[0] / img.shape[1])),
                      Image.LANCZOS if smooth else Image.NEAREST)


def crop_centre(img, zoom):
    """Keep only the middle part of the picture. zoom=2 keeps the middle half of the width and height."""
    h, w = img.shape
    ch, cw = h // zoom, w // zoom
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


# ---------- cleaner combining and brightness matching ----------
def combine(frames):
    """Combine the pictures of one pass. With 3 or more, the median ignores one-off glitches like cosmic-ray hits."""
    frames = np.asarray(frames)
    return np.nanmedian(frames, axis=0) if len(frames) >= 3 else np.nanmean(frames, axis=0)


def match_brightness(reference, other):
    """Scale `other` so the bright fixed stars agree with `reference`. Returns the scaled picture and the scale used.

    If a whole pass is a little brighter or dimmer (colour, calibration), every star would leave a
    leftover when subtracted. Matching the brightness first removes most of those leftovers.
    """
    bright = np.isfinite(reference) & np.isfinite(other) & (reference > np.nanpercentile(reference, 95))
    if bright.sum() < 10:
        return other, 1.0
    scale = float(np.sum(reference[bright] * other[bright]) / np.sum(other[bright] ** 2))
    if not 0.5 < scale < 2.0:
        return other, 1.0
    return other * scale, scale


# ---------- infrared colour picture ----------
def colour_composite(frames, wavelengths):
    """Split the pictures into the shortest, middle and longest colours and average each group.

    Returns three pictures (short, middle, long) and the wavelength range each one covers.
    """
    order = np.argsort(wavelengths)
    groups = np.array_split(order, 3)
    channels = [np.nanmean(frames[g], axis=0) for g in groups]
    ranges = [(float(np.min(wavelengths[g])), float(np.max(wavelengths[g]))) for g in groups]
    return channels, ranges


def render_rgb(channels, width=600, smooth=True):
    """Short = blue, middle = green, long = red. The SAME brightness scale is used for all three,
    so a star only looks red if it is really brighter at long wavelengths."""
    short, middle, long_ = channels
    stack = np.stack([long_, middle, short], axis=-1)
    lo, hi = np.nanpercentile(stack, [50, 99.8])
    scaled = np.clip((np.nan_to_num(stack) - lo) / (hi - lo + 1e-9), 0, 1)
    scaled = np.arcsinh(10 * scaled) / np.arcsinh(10)
    pic = Image.fromarray((scaled * 255).astype(np.uint8))
    return pic.resize((width, int(width * stack.shape[0] / stack.shape[1])),
                      Image.LANCZOS if smooth else Image.NEAREST)


# ---------- difference view ----------
AMBER = (255, 176, 0)


def noise_level(img):
    """Typical size of the random noise, using the median absolute deviation (ignores bright spots)."""
    return 1.4826 * np.nanmedian(np.abs(img - np.nanmedian(img)))


def render_difference(diff, sigma, width=600, smooth=True):
    """Black background. Amber = got brighter (where it is now). White = got fainter (where it was)."""
    d = np.nan_to_num(diff) / (6 * sigma + 1e-12)
    amber = np.clip(d, 0, 1)[..., None] * np.array(AMBER)
    white = np.clip(-d, 0, 1)[..., None] * np.array([255, 255, 255])
    rgb = np.clip(amber + white, 0, 255).astype(np.uint8)
    pic = Image.fromarray(rgb)
    return pic.resize((width, int(width * diff.shape[0] / diff.shape[1])),
                      Image.BICUBIC if smooth else Image.NEAREST)


def _peaks(img, threshold):
    """Pixels that are the brightest in their 3x3 neighbourhood and above the threshold."""
    keep = (img == maximum_filter(img, size=3)) & (img > threshold)
    ys, xs = np.nonzero(keep)
    return [(int(x), int(y), float(img[y, x])) for x, y in zip(xs, ys)]


def find_movers(diff, sigma, average, min_sep=1.5, max_sep=8.0, threshold=4.0, star_ratio=3.0):
    """Look for a PAIR: a fainter spot (old position) and a brighter spot (new position) a few pixels apart.

    A fixed star that is slightly misaligned makes a pair only a pixel apart, so min_sep skips those.
    A real mover leaves its old spot fainter and its new spot brighter by about the same amount.
    `average` is the average of the two pictures. If it is much brighter than the pair near the pair's
    middle, the pair is just leftover from a bright fixed star, so it is skipped.
    """
    diff = np.nan_to_num(diff)
    new_spots = _peaks(diff, threshold * sigma)
    old_spots = _peaks(-diff, threshold * sigma)
    movers = []
    for nx, ny, nval in new_spots:
        for ox, oy, oval in old_spots:
            sep = float(np.hypot(nx - ox, ny - oy))
            balance = nval / oval
            if min_sep <= sep <= max_sep and 0.4 <= balance <= 2.5:
                mx, my = (nx + ox) // 2, (ny + oy) // 2
                near = average[max(0, my - 1):my + 2, max(0, mx - 1):mx + 2]
                if np.nanmax(near) > star_ratio * min(nval, oval):
                    continue
                movers.append({"new": (nx, ny), "old": (ox, oy), "sep": sep,
                               "score": min(nval, oval) / sigma})
    return sorted(movers, key=lambda m: -m["score"])


def draw_pair(pic, mover, shape, ring=14):
    """Draw a white ring on the old position and an amber ring on the new one."""
    scale = pic.size[0] / shape[1]
    pic = pic.copy()
    draw = ImageDraw.Draw(pic)
    for (x, y), colour in ((mover["old"], (255, 255, 255)), (mover["new"], AMBER)):
        cx, cy = (x + 0.5) * scale, (y + 0.5) * scale
        draw.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=colour, width=2)
    return pic
