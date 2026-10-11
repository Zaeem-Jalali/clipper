"""ETA math for the processing screen (FEATURES 2.12). Pure functions only: the
database reads and writes live in store.py, the wiring in run.py and the router.

A pipeline job has three steps (transcribe, analyze, render), the same ones the
processing screen shows. Each finished job records how fast every step ran, as a
multiple of the media the step handled ("8.2x real time"), together with the
machine and the transcription backend and model. A running job's estimate is the
median of the last few of those, refined with the job's own live speed.

What "the media the step handled" means depends on the step. Transcribing and
analyzing scale with the source length. Rendering scales with how much output is
produced (the summed clip lengths of a process job, the program length of an
edit), not with the source, so its speed is per output second."""
from __future__ import annotations

import math
import platform
import statistics
from typing import Any

ETA_KINDS = ("process", "autoedit")
STEPS = ("transcribe", "analyze", "render")
# The unit a step's speed is measured in. History rows carry it, so rows recorded
# under another unit are never mixed in.
UNIT = {"transcribe": "source", "analyze": "source", "render": "output"}

_STAGE_STEP = {
    "": "transcribe", "ingest": "transcribe", "probe": "transcribe",
    "audio": "transcribe", "transcribe": "transcribe",
    "analyze": "analyze", "direct": "analyze", "assets": "analyze",
    "render": "render", "concat": "render", "mix": "render", "burn": "render",
}

HISTORY_N = 5  # past jobs an estimate is based on
KEEP_PER_GROUP = 20  # history rows kept per machine, kind, step, backend and model
ESTIMATING_S = 20.0  # with no history, "Estimating..." for this long
WIDE_SPREAD_S = 40.0  # a range wider than this shows as "a-b min left"
# k in the range width `remaining x k x (1 - progress)`, by number of past jobs
_K_BY_HISTORY = (0.55, 0.5, 0.42, 0.36, 0.3, 0.25)
_MIN_LIVE_PROGRESS = 0.05  # live speed needs some progress and time to mean anything
_MIN_LIVE_S = 3.0
# A recorded sample is noise outside these bounds. Wall-clock time also counts a
# laptop asleep mid-job, so a step slower than 1/20 of real time (MIN_SPEED) or
# longer than 6 hours (MAX_STEP_S) is dropped rather than recorded.
MIN_MEDIA_S = 1.0
MIN_STEP_S = 0.05
MAX_STEP_S = 6 * 3600.0
MIN_SPEED = 0.05
MAX_SPEED = 5000.0

_MAX_MACHINE = 64
_MAX_MODEL = 64

History = dict[str, list[tuple[float, float]]]  # step -> [(seconds per unit, units)], newest first


def machine_id() -> str:
    return platform.node()[:_MAX_MACHINE]


def step_for_stage(stage: str) -> str | None:
    """The processing-screen step a job stage belongs to; None for 'done'."""
    return _STAGE_STEP.get(stage)


def source_from_message(message: str) -> str | None:
    """Which transcription tier run.py's on_stage message announces."""
    if "YouTube" in message:
        return "youtube_captions"
    if "Groq" in message:
        return "groq"
    if "locally" in message:
        return "local"
    return None


def transcription_key(source: str, language: str, *, whisper: str, multilingual: str, gemini: str) -> tuple[str, str]:
    """(backend, model) a transcription ran on. Groq and YouTube captions have
    one fixed model each, so their model is empty."""
    if source == "local":
        model = multilingual if language in ("hi", "dari") else whisper
    elif source == "gemini":
        model = gemini
    else:
        model = ""
    return source[:32], model[:_MAX_MODEL]


def lookup_keys(announced: str, language: str, *, whisper: str, multilingual: str, gemini: str) -> list[tuple[str, str]]:
    """The (backend, model) history to try for a running job, best first. The local
    tier for Dari tries Gemini first and falls back to Whisper, and a transcript
    records the tier that produced it, so both are candidates."""
    kw = {"whisper": whisper, "multilingual": multilingual, "gemini": gemini}
    keys = [transcription_key(announced, language, **kw)]
    if announced == "local" and language == "dari":
        keys.insert(0, transcription_key("gemini", language, **kw))
    return keys


def step_speeds(marks: dict[str, Any], ended_at: float, media_s: float) -> dict[str, tuple[float, float]]:
    """step -> (speed, units of work) for each step a finished job ran. Speed is
    units of work per wall second, in the step's UNIT. A step the job skipped (a
    cached transcript), one whose work is unknown or that ran implausibly fast or
    slow is left out; a zero-length source records nothing."""
    if not media_s >= MIN_MEDIA_S:
        return {}
    starts = [(marks[f"start_{s}"], s) for s in STEPS if f"start_{s}" in marks]
    out: dict[str, tuple[float, float]] = {}
    for i, (t0, step) in enumerate(starts):
        if marks.get(f"skip_{step}"):
            continue
        t1 = starts[i + 1][0] if i + 1 < len(starts) else ended_at
        # transcription ends when the transcriber returns, before the spelling fix
        t1 = min(t1, marks.get(f"end_{step}", t1))
        seconds = t1 - t0
        if not MIN_STEP_S <= seconds <= MAX_STEP_S:
            continue
        work = marks.get("work_render") if UNIT[step] == "output" else media_s
        if not isinstance(work, (int, float)) or work < MIN_MEDIA_S:
            continue
        speed = work / seconds
        if MIN_SPEED <= speed <= MAX_SPEED:
            out[step] = (speed, float(work))
    return out


def _typical_s(rows: list[tuple[float, float]], work: float | None) -> float:
    """Typical seconds a step takes: the median past speed applied to the work at
    hand, or, when that isn't known yet, the median past duration."""
    if work is not None:
        return statistics.median(r for r, _ in rows) * work
    return statistics.median(r * w for r, w in rows)


def estimate(
    *,
    step: str,
    step_progress: float,
    progress: float,
    elapsed_s: float,
    step_elapsed_s: float,
    works: dict[str, float | None],
    history: History,
    steps: tuple[str, ...] = STEPS,
) -> dict[str, Any]:
    """The ETA payload minus its timestamps. `works` is the known work of each step
    in its unit (None when not known yet: a source-length step needs the probed
    length, render needs the clip list). `history` holds the past rows of each step,
    newest first. `progress` and `step_progress` are 0-1."""
    p = min(max(step_progress, 0.0), 1.0)
    overall = min(max(progress, 0.0), 1.0)
    needed = steps[steps.index(step):]

    def usable(s: str) -> bool:
        rows = history.get(s, [])
        if not rows:
            return False
        w = works.get(s)
        # a source-length step with no known length can't use its history
        return w is not None or UNIT[s] == "output"

    count = min((len(history[s][:HISTORY_N]) if usable(s) else 0) for s in needed)
    speed: float | None = None
    if count == 0:
        if elapsed_s < ESTIMATING_S or overall <= 0:
            return {
                "state": "estimating", "remaining_s": None, "low_s": None, "high_s": None,
                "wide": False, "confidence": 0, "history_count": 0, "speed": None,
            }
        state = "live"
        remaining = elapsed_s * (1 - overall) / overall
    else:
        state = "history"
        total = {s: _typical_s(history[s][:HISTORY_N], works.get(s)) for s in needed}
        cur_total = total[step]
        if p >= _MIN_LIVE_PROGRESS and step_elapsed_s >= _MIN_LIVE_S:
            cur_total = p * (step_elapsed_s / p) + (1 - p) * cur_total
        remaining = cur_total * (1 - p) + sum(total[s] for s in needed[1:])
        cur_work = works.get(step)
        speed = cur_work / cur_total if cur_work is not None and cur_total > 0 else None

    k = _K_BY_HISTORY[min(count, len(_K_BY_HISTORY) - 1)]
    spread = remaining * k * (1 - overall)
    score = 0.5 * min(count, HISTORY_N) / HISTORY_N + 0.5 * overall
    return {
        "state": state,
        "remaining_s": round(remaining, 1),
        "low_s": round(max(0.0, remaining - spread), 1),
        "high_s": round(remaining + spread, 1),
        "wide": spread > WIDE_SPREAD_S,
        "confidence": min(5, 1 + math.floor(score * 5)),
        "history_count": count,
        "speed": round(speed, 1) if speed is not None else None,
    }
