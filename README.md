# Clipper

**Long videos in. Ready to post clips out.**

Clipper is an AI tool I built that takes a long video (a podcast, an interview, a talk), finds the moments worth posting, cuts them on clean sentence and pause boundaries, reframes them to vertical around whoever is speaking, adds animated word by word captions and publishes them to YouTube, Instagram, Facebook and TikTok on a schedule.

![Clipper hook](media/hook.gif)

[Watch the 3 minute demo](media/clipper-demo.mp4)

> This is a showcase repository. Clipper's full source code is private. A code walkthrough is available on request: zaeemjalali@gmail.com

## What it does

* **Finds the moments.** Every clip gets a score out of 100, broken down into signals (hook in the first 3 seconds, self contained, clear payoff, emotional peak), each backed by a quote from the transcript.
* **Cuts cleanly.** Clip bounds snap to sentence edges, then a second pass finds a real pause in the audio and cuts inside it, so a clip never stops mid word.
* **Reframes to vertical.** MediaPipe face tracking keeps the speaker in frame for 9:16, 1:1, 4:5 or 16:9.
* **Edits instantly.** Captions are drawn as a live layer over the video in the browser, so switching styles or timing needs no render. The final captions are burned in only on export.
* **Reviews fast.** A keyboard review queue: K to keep, X to skip, then one click deletes everything skipped.
* **Publishes on a schedule.** A drag and drop week calendar posts to four platforms through their official APIs, with safe retries and automatic rescheduling around YouTube's daily upload limit.
* **Edits raw recordings too.** A second mode where an LLM "director" plans the cuts, stock footage, on screen text and music for a talking head recording, with animated graphics rendered in Remotion.

## How it works

```mermaid
flowchart LR
    A[YouTube link or upload] --> B[Ingest and probe<br/>FFmpeg]
    B --> C[Transcribe<br/>faster-whisper]
    C --> D[Find moments<br/>Gemini, OpenRouter fallback]
    D --> E[Ending pass<br/>pause detection]
    E --> F[Reframe<br/>MediaPipe]
    F --> G[Live caption preview<br/>Next.js]
    G --> H[Export<br/>FFmpeg burn]
    H --> I[Publish and schedule<br/>YouTube, Instagram, Facebook, TikTok]
```

* **Backend:** Python and FastAPI, with a separate polling job worker and SQLite. Jobs stream their progress to the browser over Server Sent Events.
* **Frontend:** Next.js, TypeScript and Tailwind CSS.
* **AI:** faster-whisper for word level transcripts, Gemini for moment finding and social copy, free OpenRouter models as an automatic fallback when Gemini is overloaded.
* **Media:** FFmpeg for cutting, reframing, captions and loudness, MediaPipe for face tracking, Remotion for animated graphics.
* **Quality:** 1,257 automated backend tests (pytest).
* **Runs locally** on a 2 core laptop, on free AI tiers.

## Screenshots

| | |
| --- | --- |
| ![New project](media/screenshots/1-new-project.png) | ![Processing with a learned time estimate](media/screenshots/2-processing.png) |
| New project, caption styles | Processing, with a time estimate learned from past jobs |
| ![Scored clips](media/screenshots/3-scored-clips.png) | ![Review queue](media/screenshots/4-review-queue.png) |
| Scored clips and the evidence behind each score | Keyboard review queue |
| ![Caption editor](media/screenshots/5-caption-editor.png) | ![Trim](media/screenshots/6-trim.png) |
| Caption editor with live preview | Trim on the waveform, snapping to silence |
| ![Schedule](media/screenshots/7-schedule.png) | |
| Publishing calendar | |

## Code samples

Three self contained pieces from the private codebase:

* [`samples/endcut.py`](samples/endcut.py): places a clip's cut inside real silence. Transcript word times drift, so it reads the audio energy and finds the first genuine pause after the last word. Tests in [`samples/tests/test_endcut.py`](samples/tests/test_endcut.py).
* [`samples/eta.py`](samples/eta.py): the processing screen's time estimate. It records how fast each step ran on this machine, then blends the median of past jobs with the job's live speed and reports a range with a confidence level.
* [`samples/llm_fallback.py`](samples/llm_fallback.py): the LLM entry point. It rotates through API keys and models, cools down a model that is overloaded, separates outages from bad answers and falls back to free OpenRouter models.

## Built by

**Zaeem Jalali**, BS Computer Science (Artificial Intelligence) student at Preston University, Islamabad. Open to roles in AI engineering and AI automation.

[GitHub](https://github.com/Zaeem-Jalali) · zaeemjalali@gmail.com

Demo footage: Blocks Podcast w/ Neal Brennan, used under CC BY.

Copyright 2026 Zaeem Jalali. All rights reserved. See [LICENSE](LICENSE).
