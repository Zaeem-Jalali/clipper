"""Place a clip's cut inside real silence, using the audio.

Transcript word times drift (a word's end can be off by half a second or more, and
a short word like "So" can be given no time at all), so cutting at the last word's
listed end can clip the start of the next word or keep a stray sound. This finds the
first real pause after the last word's speech instead.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_HOP_S = 0.01
_MIN_SILENCE_S = 0.14  # a pause shorter than this is a gap inside speech
_SILENCE_RATIO = 0.08  # a frame this far under the loud level counts as silence
_CUT_INTO_SILENCE_S = 0.10  # cut this far into the pause, so the last sound finishes
_LEVEL_LOOKBACK_S = 8.0
_MIN_SPEECH_FRAMES = 3


def _frames(wf: wave.Wave_read, start: float, end: float) -> np.ndarray:
    """RMS per 10ms frame of the mono mix over [start, end)."""
    sr, channels = wf.getframerate(), wf.getnchannels()
    start = max(0.0, start)
    wf.setpos(min(int(start * sr), wf.getnframes()))
    data = np.frombuffer(wf.readframes(max(0, int((end - start) * sr))), dtype=np.int16)
    if channels > 1:
        data = data[: len(data) - len(data) % channels].reshape(-1, channels).mean(axis=1)
    hop = int(_HOP_S * sr)
    n = len(data) // hop
    if n == 0:
        return np.zeros(0)
    framed = data[: n * hop].astype(np.float64).reshape(n, hop)
    return np.sqrt((framed**2).mean(axis=1))


def find_cut(
    audio: Path, word_start: float, word_end: float, *, max_after_s: float = 1.2
) -> float | None:
    """The time (source seconds) to end a clip whose last word runs from
    `word_start` to `word_end` in the transcript: just inside the first real pause
    after that word's speech. None when no clear pause is found."""
    with wave.open(str(audio)) as wf:
        loud = _frames(wf, word_start - _LEVEL_LOOKBACK_S, word_end + 0.5)
        if len(loud) == 0:
            return None
        threshold = _SILENCE_RATIO * float(np.percentile(loud, 90))
        origin = max(0.0, word_start - 0.3)
        frames = _frames(wf, origin, word_end + max_after_s)
    if len(frames) == 0 or threshold <= 0:
        return None
    quiet = frames < threshold
    first = max(0, round((word_start + 0.05 - origin) / _HOP_S))
    need = round(_MIN_SILENCE_S / _HOP_S)
    # word_start can run a little late, so its speech may begin just before it
    heard_from = max(0, round((word_start - 0.15 - origin) / _HOP_S))
    i = first
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        # the pause must come after some speech from this word, not before it
        spoke = int((~quiet[heard_from:i]).sum()) >= _MIN_SPEECH_FRAMES
        if j - i >= need and spoke:
            length_s = (j - i) * _HOP_S
            return round(origin + i * _HOP_S + min(_CUT_INTO_SILENCE_S, length_s / 2), 3)
        i = j
    return None


def nearest_pause(
    audio: Path, at: float, edge: str, *, radius_s: float = 1.0,
    min_silence_s: float = _MIN_SILENCE_S,
) -> float | None:
    """The cut point inside the pause closest to `at` (source seconds), within
    `radius_s` that lasts at least `min_silence_s`. For a clip's end the cut sits just inside the start of the pause, so
    the last sound finishes; for its start, just inside the end of the pause, so the
    first word is whole. None when there is no clear pause in reach."""
    with wave.open(str(audio)) as wf:
        loud = _frames(wf, at - _LEVEL_LOOKBACK_S, at + radius_s)
        if len(loud) == 0:
            return None
        threshold = _SILENCE_RATIO * float(np.percentile(loud, 90))
        origin = max(0.0, at - radius_s)
        frames = _frames(wf, origin, at + radius_s)
    if len(frames) == 0 or threshold <= 0:
        return None
    quiet = frames < threshold
    need = round(min_silence_s / _HOP_S)
    best: tuple[float, int, int] | None = None
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        if j - i >= need:
            lo, hi = origin + i * _HOP_S, origin + j * _HOP_S
            distance = 0.0 if lo <= at <= hi else min(abs(lo - at), abs(hi - at))
            if best is None or distance < best[0]:
                best = (distance, i, j)
        i = j
    if best is None:
        return None
    _, i, j = best
    inside = min(_CUT_INTO_SILENCE_S, (j - i) * _HOP_S / 2)
    if edge == "start":
        return round(origin + j * _HOP_S - inside, 3)
    return round(origin + i * _HOP_S + inside, 3)
