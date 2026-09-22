"""Best-effort native dark caption without replacing Windows window controls."""

import ctypes
import sys


def apply_dark_caption(window_id: int) -> bool:
    """Ask DWM for a dark native caption; unsupported systems stay functional."""
    if sys.platform != "win32":
        return False
    try:
        api = ctypes.windll.dwmapi.DwmSetWindowAttribute
        api.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        api.restype = ctypes.c_long
        enabled = ctypes.c_int(1)
        for attribute in (20, 19):
            result = api(
                ctypes.c_void_p(window_id),
                attribute,
                ctypes.byref(enabled),
                ctypes.sizeof(enabled),
            )
            if result == 0:
                return True
    except (OSError, AttributeError):
        pass
    return False
