"""Pieces of the multi-task feature tag shared by the analyses and the figures.

A multi-task run name carries its feature tag between the seed and the
hidden width (``everything_seed749_L21e4+hidden300+batch128+angle``). The
tag starts with the L2 strength (``L21e4`` is 1e-4) and continues with the
settings that change the network itself: the modulation bounds (``mb2``),
the modulation decay (``lam0.9``) and the projection or bottleneck widths.
L2 acts only during training, so controls built from a run's config without
training (an untrained network of the same architecture) are the same for
every L2 cohort of one architecture, and are keyed by ``architecture_tag``
rather than by the feature.
"""
import re

_L2_STRENGTH = re.compile(r"^L2\d+(?:\.\d+)?e\d+")


def architecture_tag(feature):
    """The feature tag without its leading L2 strength: what the tag says
    about the network rather than its training. ``'L21e4'`` and ``'L21e3'``
    both give ``''``; ``'L21e4lam0.9'`` gives ``'lam0.9'``."""
    match = _L2_STRENGTH.match(feature)
    if not match:
        raise ValueError(f"Not a multi-task feature tag (no leading L2 strength): {feature!r}")
    return feature[match.end():]


def untrained_sharing_tag(feature):
    """File tag of the untrained row-cluster sharing control that serves
    every L2 cohort of ``feature``'s architecture: ``'untrained'`` for the
    default architecture, otherwise ``'<architecture>_untrained'``."""
    architecture = architecture_tag(feature)
    return f"{architecture}_untrained" if architecture else "untrained"
