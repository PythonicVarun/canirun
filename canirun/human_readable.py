from canirun.enum import COMPATIBILITY


def get_human_readable_status(status: COMPATIBILITY) -> str:
    match status:
        case COMPATIBILITY.FULL:
            return "✅ GPU"
        case COMPATIBILITY.PARTIAL:
            return "⚠️ CPU/RAM only (Slow)"
        case _:
            return "❌ Impossible"


def get_human_readable_size(size_bytes: float) -> str:
    if size_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1

    return f"{size_bytes:.2f} {units[i]}"
