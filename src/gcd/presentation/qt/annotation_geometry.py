from __future__ import annotations


def normalize_rect(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = rect
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def move_rect(
    rect: tuple[float, float, float, float], delta: tuple[float, float]
) -> tuple[float, float, float, float]:
    left, top, right, bottom = normalize_rect(rect)
    dx, dy = delta
    return (left + dx, top + dy, right + dx, bottom + dy)


def resize_rect_with_handle(
    rect: tuple[float, float, float, float], handle_index: int, position: tuple[float, float]
) -> tuple[float, float, float, float]:
    left, top, right, bottom = normalize_rect(rect)
    opposite_corners = {
        0: (right, bottom),
        1: (left, bottom),
        2: (right, top),
        3: (left, top),
    }
    anchor_x, anchor_y = opposite_corners[handle_index]
    return normalize_rect((anchor_x, anchor_y, position[0], position[1]))


def has_meaningful_3d_box_drag(start, end, threshold: float = 1.0) -> bool:
    return any(abs(float(end[i]) - float(start[i])) >= threshold for i in range(3))
