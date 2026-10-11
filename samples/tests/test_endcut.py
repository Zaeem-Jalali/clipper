"""endcut.find_cut: a cut placed in the real pause, whatever the transcript says."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from app.pipeline import endcut

SR = 16000


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.int16)


class _WavCase(unittest.TestCase):
    def _wav(self, *parts: np.ndarray) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(np.concatenate(parts).tobytes())
        return Path(tmp.name)


class TestFindCut(_WavCase):
    def test_cuts_inside_the_pause_even_when_the_word_end_is_late(self):
        # speech 0-3s, pause 3-3.5s, speech 3.5-5s. The transcript says the last
        # word runs 2.0-3.9, so its listed end is inside the next word.
        audio = self._wav(_tone(3.0), _silence(0.5), _tone(1.5))
        cut = endcut.find_cut(audio, 2.0, 3.9)
        self.assertIsNotNone(cut)
        self.assertGreater(cut, 3.0)
        self.assertLess(cut, 3.5)

    def test_skips_a_gap_that_comes_before_the_last_word(self):
        # a pause at 1.0-1.5 precedes the last word (starts 1.5); the cut must come
        # after that word's speech, in the pause at 2.5-3.0
        audio = self._wav(_tone(1.0), _silence(0.5), _tone(1.0), _silence(0.5), _tone(1.0))
        cut = endcut.find_cut(audio, 1.5, 2.6)
        self.assertGreater(cut, 2.5)
        self.assertLess(cut, 3.0)

    def test_no_pause_returns_none(self):
        audio = self._wav(_tone(5.0))
        self.assertIsNone(endcut.find_cut(audio, 1.0, 2.0))

    def test_a_short_dip_inside_speech_is_not_a_pause(self):
        audio = self._wav(_tone(2.0), _silence(0.05), _tone(3.0))
        self.assertIsNone(endcut.find_cut(audio, 1.0, 2.5, max_after_s=1.0))


class TestNearestPause(_WavCase):
    def test_end_edge_cuts_just_inside_the_start_of_the_nearest_pause(self):
        audio = self._wav(_tone(2.0), _silence(0.5), _tone(2.0), _silence(0.5), _tone(1.0))
        cut = endcut.nearest_pause(audio, 2.3, "end")
        self.assertGreater(cut, 2.0)
        self.assertLess(cut, 2.2)

    def test_start_edge_cuts_just_inside_the_end_of_the_pause(self):
        audio = self._wav(_tone(2.0), _silence(0.5), _tone(2.0))
        cut = endcut.nearest_pause(audio, 2.3, "start")
        self.assertGreater(cut, 2.3)
        self.assertLess(cut, 2.5)

    def test_picks_the_closest_of_two_pauses(self):
        audio = self._wav(_tone(1.0), _silence(0.4), _tone(1.0), _silence(0.4), _tone(1.0))
        cut = endcut.nearest_pause(audio, 2.3, "end", radius_s=1.5)
        self.assertGreater(cut, 2.3)  # the pause at 2.4-2.8, not the one at 1.0-1.4

    def test_no_pause_in_reach_is_none(self):
        self.assertIsNone(endcut.nearest_pause(self._wav(_tone(4.0)), 2.0, "end"))



class TestNearestPauseMinimum(_WavCase):
    def test_a_breath_is_skipped_when_a_phrase_pause_is_required(self):
        # speech, a 0.2s breath at 2.0s, speech, a 0.5s pause at 4.0s, speech
        audio = self._wav(_tone(2.0), _silence(0.2), _tone(1.8), _silence(0.5), _tone(2.0))
        breath = endcut.nearest_pause(audio, 2.1, "end", radius_s=3.0)
        phrase = endcut.nearest_pause(audio, 2.1, "end", radius_s=3.0, min_silence_s=0.3)
        self.assertLess(breath, 2.3)
        self.assertGreater(phrase, 4.0)
        self.assertLess(phrase, 4.5)

if __name__ == "__main__":
    unittest.main()
