from typing import List, Tuple, Union
import math

ColorSpec = Union[str, Tuple[int, int, int], Tuple[float, float, float]]

def _parse_color(c: ColorSpec) -> Tuple[float, float, float]:
    """Return RGB in 0–1 floats."""
    if isinstance(c, str):
        if c.startswith("#") and len(c) == 7:
            r = int(c[1:3], 16) / 255.0
            g = int(c[3:5], 16) / 255.0
            b = int(c[5:7], 16) / 255.0
            return (r, g, b)
        # Try named colors via matplotlib if available
        try:
            import matplotlib.colors as mcolors
            return mcolors.to_rgb(c)
        except Exception:
            raise ValueError("Use hex like '#RRGGBB' or install matplotlib for named colors.")
    if isinstance(c, (tuple, list)) and len(c) == 3:
        a, b, d = c
        if all(isinstance(v, (int, float)) for v in (a, b, d)):
            if all(0 <= v <= 1 for v in (a, b, d)):
                return (float(a), float(b), float(d))
            if all(0 <= v <= 255 for v in (a, b, d)):
                return (a/255.0, b/255.0, d/255.0)
    raise ValueError("Unsupported color format.")

def _format_rgb(rgb01: Tuple[float, float, float], output: str):
    r, g, b = rgb01
    if output == "hex":
        return f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"
    if output == "rgb255":
        return (round(r*255), round(g*255), round(b*255))
    if output == "rgb01":
        return (r, g, b)
    raise ValueError("output must be 'hex', 'rgb255', or 'rgb01'.")

def make_multistop_gradient(
    x: int,
    stops: List[ColorSpec],
    output: str = "hex",
) -> List[Union[str, Tuple[int, int, int], Tuple[float, float, float]]]:
    """
    Interpolate `x` colors across `stops` (>=2 colors).
    First color == stops[0], last color == stops[-1].
    """
    if x < 1:
        raise ValueError("x must be >= 1")
    if len(stops) < 2:
        raise ValueError("Provide at least two stop colors.")

    stops01 = [_parse_color(s) for s in stops]
    if x == 1:
        return [_format_rgb(stops01[0], output)]

    out = []
    K = len(stops01) - 1  # number of segments
    for i in range(x):
        u = i / (x - 1)            # 0..1 along the whole gradient
        pos = u * K                # segment position
        s = min(int(math.floor(pos)), K - 1)
        t = pos - s                # 0..1 within the segment
        r = (1 - t) * stops01[s][0] + t * stops01[s+1][0]
        g = (1 - t) * stops01[s][1] + t * stops01[s+1][1]
        b = (1 - t) * stops01[s][2] + t * stops01[s+1][2]
        out.append(_format_rgb((r, g, b), output))
    return out

def rainbow_generate(N):
    rainbow_stops = [
        "#ff0000",  # red
        "#ff7f00",  # orange
        "#ffff00",  # yellow
        "#00ff00",  # green
        "#00ffff",  # cyan
        "#0000ff",  # blue
        "#4b0082",  # indigo
        "#8a2be2",  # purple (blueviolet)
    ]

    colors = make_multistop_gradient(N, rainbow_stops, output="hex")
    return colors