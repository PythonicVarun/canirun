"""Helper functions for human-readable output."""

from canirun.enum import COMPATIBILITY


def get_human_readable_status(status: COMPATIBILITY) -> str:
    """Converts a COMPATIBILITY enum value into a human-readable string.

    Args:
        status: The compatibility status.

    Returns:
        str: Human-readable status.
    """
    match status:
        case COMPATIBILITY.FULL:
            return "✅ GPU"
        case COMPATIBILITY.PARTIAL:
            return "⚠️ CPU/RAM only (Slow)"
        case _:
            return "❌ Impossible"


def get_human_readable_size(size_bytes: float) -> str:
    """Converts a size in bytes to a human-readable string format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        str: Human-readable size string.
    """
    if size_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1

    return f"{size_bytes:.2f} {units[i]}"
