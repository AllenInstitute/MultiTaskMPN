"""Shared provenance checks for state-space results and their readers."""

M_ALIGNMENT = "post_update"
DISTANCE_ANGLE_SCHEMA_VERSION = 3


def is_aligned(data):
    return data.get("m_alignment") == M_ALIGNMENT


def require_aligned(data, source):
    if not is_aligned(data):
        raise ValueError(f"{source}: missing or incompatible m_alignment; rebuild from aligned inputs")


def distance_angle_is_current(data):
    return (data.get("schema_version", 0) >= DISTANCE_ANGLE_SCHEMA_VERSION
            and is_aligned(data))
