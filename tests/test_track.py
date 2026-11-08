"""Test tracker centroid + garis keputusan (urutan centroid sintetis, tanpa gambar)."""

import pytest

from loaflens.config import TrackConfig
from loaflens.track import CentroidTracker

W, H = 400, 300
SIZE = 40  # sisi bbox roti


def box(cx: float, cy: float) -> tuple[int, int, int, int]:
    return round(cx - SIZE / 2), round(cy - SIZE / 2), SIZE, SIZE


def path(direction: str, steps: int, speed: float, cross: float = 200) -> list[tuple[float, float]]:
    """Centroid satu roti melintasi frame searah ``direction`` (koordinat silang tetap)."""
    pts = []
    for i in range(steps):
        p = 10 + speed * i  # jarak dari tepi masuk
        pts.append(
            {
                "left_to_right": (p, H / 2),
                "right_to_left": (W - p, H / 2),
                "top_to_bottom": (cross, p),
                "bottom_to_top": (cross, H - p),
            }[direction]
        )
    return pts


def tracker(direction: str, line_pos: float = 0.5, **kw: float) -> CentroidTracker:
    return CentroidTracker(TrackConfig(direction=direction, line_pos=line_pos, **kw), (W, H))


@pytest.mark.parametrize(
    "direction", ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]
)
def test_each_bun_decided_exactly_once_at_crossing(direction: str) -> None:
    tr = tracker(direction)
    line = tr.line
    decided_at = []
    for f, c in enumerate(path(direction, 60, 6)):
        crossed = tr.update([box(*c)])
        if crossed:
            assert tr.progress(c) >= line  # tidak sebelum melintas
            decided_at.append((f, crossed[0].id))
        else:
            assert not decided_at or tr.progress(c) >= line
    assert len(decided_at) == 1
    f, tid = decided_at[0]
    assert tid == 1
    assert tr.progress(path(direction, 60, 6)[f - 1]) < line


def test_lost_then_reappearing_within_max_missed_keeps_id() -> None:
    tr = tracker("top_to_bottom", max_missed=3)
    ids, crossings = set(), 0
    for f, c in enumerate(path("top_to_bottom", 50, 6)):
        dets = [] if 22 <= f <= 24 else [box(*c)]  # hilang 3 frame tepat di sekitar garis (y=150)
        crossings += len(tr.update(dets))
        ids |= {t.id for t in tr.visible()}
    assert ids == {1}
    assert crossings == 1


def test_lost_too_long_and_reborn_past_line_is_not_decided_twice() -> None:
    tr = tracker("top_to_bottom", max_missed=2)
    crossings = []
    for f, c in enumerate(path("top_to_bottom", 50, 6)):
        dets = [] if 26 <= f <= 32 else [box(*c)]  # lewat garis di f=24, hilang 7 frame
        crossings += [t.id for t in tr.update(dets)]
    assert crossings == [1]
    assert max(tr.tracks) == 2  # lahir ulang dengan id baru, tidak diputuskan


def test_born_past_line_never_decided() -> None:
    tr = tracker("left_to_right")
    assert all(not tr.update([box(x, 150)]) for x in range(250, 380, 8))


def test_two_buns_side_by_side_do_not_swap() -> None:
    tr = tracker("top_to_bottom")
    first: dict[int, float] = {}
    crossings = 0
    for y in range(30, 280, 7):
        dets = [box(150, y), box(150 + SIZE + 10, y + 3)]  # bersebelahan, hampir sejajar
        crossings += len(tr.update(dets))
        for t in tr.visible():
            first.setdefault(t.id, t.center[0])
            assert t.center[0] == pytest.approx(first[t.id], abs=1)
    assert sorted(first) == [1, 2]
    assert crossings == 2


def test_jitter_back_and_forth_over_line_decides_once() -> None:
    tr = tracker("top_to_bottom")
    ys = [100, 120, 140, 152, 148, 155, 149, 160, 180]  # garis di y=150
    assert sum(len(tr.update([box(200, y)])) for y in ys) == 1


def test_line_points_follow_direction() -> None:
    assert tracker("top_to_bottom").line_points() == ((0, 150), (W - 1, 150))
    assert tracker("right_to_left", line_pos=0.25).line_points() == ((300, 0), (300, H - 1))
