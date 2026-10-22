"""The multi-task heatmap cluster strips stay distinguishable from the other colors in their figure."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

import _bootstrap  # noqa: F401
import paper_plot

# Smallest CIELAB distance accepted between a strip block and anything it could be read as.
MIN_DISTANCE = 30.0
# The tick-label highlight is drawn at this alpha over white (_color_phase_ticklabels).
LABEL_ALPHA = 0.8


def cielab(color):
    """CIE L*a*b* (D65) of a matplotlib color."""
    rgb = np.array(mcolors.to_rgb(color))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    to_xyz = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])
    xyz = to_xyz @ linear / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def distance(first, second):
    return float(np.linalg.norm(cielab(first) - cielab(second)))


def as_drawn_label(color):
    """A phase color as it appears behind a tick label: blended over white."""
    return tuple(LABEL_ALPHA * np.array(mcolors.to_rgb(color)) + (1 - LABEL_ALPHA))


class ClusterStripColorTests(unittest.TestCase):
    def test_strip_tones_are_not_gray(self):
        for color in paper_plot._CLUSTER_COLORS:
            _, a, b = cielab(color)
            self.assertGreater(np.hypot(a, b), 20.0, color)

    def test_strip_tones_are_far_from_every_phase_label(self):
        for color in paper_plot._CLUSTER_COLORS:
            for phase, phase_color in paper_plot._PHASE_COLORS.items():
                for shown in (phase_color, as_drawn_label(phase_color)):
                    self.assertGreaterEqual(distance(color, shown), MIN_DISTANCE,
                                            f"{color} vs {phase}")

    def test_strip_tones_are_far_from_the_heatmap_scale(self):
        cmap = matplotlib.colormaps[paper_plot._MULTITASK_HEATMAP_CMAP]
        for color in paper_plot._CLUSTER_COLORS:
            nearest = min(distance(color, cmap(t)[:3]) for t in np.linspace(0, 1, 256))
            self.assertGreaterEqual(nearest, MIN_DISTANCE, color)

    def test_neighboring_blocks_are_told_apart(self):
        light, dark = paper_plot._CLUSTER_COLORS
        self.assertGreaterEqual(distance(light, dark), MIN_DISTANCE)

    def test_strips_alternate_the_two_tones(self):
        labels = np.array([1, 1, 2, 2, 2, 3, 4, 4])
        breaks = paper_plot._breaks(labels)
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        strips = (paper_plot._add_col_cluster_strip(ax, labels, breaks),
                  paper_plot._add_row_cluster_strip(ax, labels, breaks))
        expected = [mcolors.to_rgba(paper_plot._CLUSTER_COLORS[i % 2]) for i in range(4)]
        for strip in strips:
            self.assertEqual([tuple(p.get_facecolor()) for p in strip.patches], expected)


if __name__ == "__main__":
    unittest.main()
