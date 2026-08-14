# Spec: record-and-compare benchmark screen

**Status:** not built. Handoff spec for a follow-up agent.
**Builds on:** `research/asr-model-survey` (commit `ccada5b`).

## Goal

In the settings app: record **one** voice sample, run it through **every** installed model, and
show a table of accuracy, time, and memory so the user can pick a model on their own voice and
their own phone rather than on published WER.

## What already exists (reuse, don't rebuild)

| Thing | Where | Notes |
|---|---|---|
| Engine-agnostic model handle | `ml/SpeechModel.kt` | `run(samples, glossary, forceLanguage, decodingMode)` + `close()`. Both engines implement it. |
| Whisper impl | `ml/WhisperModel.kt` → `WhisperModelWrapper` | Takes `ModelData`. Has an `onPartialDecode` callback. |
| Parakeet impl | `ml/ParakeetModel.kt` → `ParakeetModelWrapper` | Takes `SherpaModelData` + `numThreads`. No partial callback. |
| Whisper catalog | `Util.kt` → `ENGLISH_MODELS`, `MULTILINGUAL_MODELS` | Check `context.modelNeedsDownloading(m)` before loading. |
| Parakeet catalog | `ParakeetModels.kt` → `PARAKEET_MODELS` | Check `context.parakeetModelNeedsDownloading(m)`; gate all of it on `isParakeetSupported()`. |
| Fixed-clip benchmark screen | `settings/pages/Benchmark.kt` | **This is the thing to extend.** Already loads a bundled clip, loops models one at a time, times load + decode, computes RTF, renders rows. Route `"benchmark"` is registered in `SettingsUtils.kt`. |
| Same benchmark as a test | `androidTest/ParakeetBenchmarkTest.kt` | Run with `-PtestRelease` — see "Timing correctness" below. |
| Audio capture reference | `AudioRecognizer.kt:~300-450` | `AudioRecord`, 16 kHz mono, `FloatBuffer`, Silero VAD auto-stop. |

## What to add

### 1. Record instead of (or as well as) the bundled clip

The bundled clip is `app/src/main/assets/benchmark_audio.floats.bin` — raw LE float32, 16 kHz,
5.86 s. Keep it as a "reference clip" option so results stay comparable across devices.

For recording, do **not** reuse `AudioRecognizer` wholesale — it is wired to the recognizer UI
lifecycle, VAD auto-stop, model loading and result dispatch. Lift just the `AudioRecord` setup
(16 kHz, mono, float) into a small recorder that fills a `FloatArray`, with explicit
start/stop buttons. `RECORD_AUDIO` is already granted for the app but check at runtime anyway,
since settings may be the first place the user hits it.

Let the user keep and re-run the last recording — the whole value is running *the same* audio
through every model.

### 2. Accuracy without a reference transcript

There is no ground truth for a fresh recording, so real WER is not available. Options, roughly
in order of value:

- **Let the user type what they said.** A text field; compute WER/CER against it. This is the
  only true accuracy number and it is cheap to implement. Normalise before scoring (lowercase,
  strip punctuation, expand digits) or the "mister"/"Mr." difference below will dominate.
- **Agreement matrix.** With no reference, show pairwise WER between models, plus a
  "consensus" column (WER against the majority transcript). Surprisingly useful for spotting
  the odd model out.
- **Always show the raw transcripts.** Non-negotiable. For dictation, formatting differences
  matter more than the WER digit and only the human can judge them.

Use a standard Levenshtein-on-tokens WER. Do not pull in a dependency for this.

**Known formatting difference to handle in normalisation:** Parakeet writes `mister` for `Mr.`
and does not capitalise the first word; the 600M capitalises mid-sentence nouns
(`the Apostle of the Middle Classes`). Whisper writes `Mr.` and capitalises normally. If you
score raw strings, Parakeet will look far worse than it is.

### 3. Memory

`Benchmark.kt` currently reports `Debug.getNativeHeapAllocatedSize()` deltas, which is crude —
onnxruntime and ggml both use mmap and arenas that this misses. Better:

- `Debug.MemoryInfo` via `ActivityManager.getProcessMemoryInfo()` → `totalPss`, sampled before
  load / after load / after decode. PSS is the number that predicts getting killed.
- `Runtime.getRuntime().totalMemory() - freeMemory()` for the JVM side (near-zero here; both
  engines are native).
- Peak matters more than steady state. Consider sampling PSS on a background thread every
  ~100 ms during decode and reporting the max.

Worth capturing: the 600M Parakeet takes **7.2 s to load on an S23** with ~159 MB free, versus
1.9 s on a Z Fold. Load time under memory pressure is a real signal and currently invisible.

### 4. Presentation

A sortable table beats the current card list once there are 6+ models × 4 metrics. Columns:
model, engine, load ms, decode ms, RTF, peak PSS, WER (if reference given), transcript.

Add "copy results as markdown" — this is exactly the kind of thing that ends up pasted into an
issue.

## Timing correctness — read this before trusting any number

Instrumentation tests default to the **debug** build, which compiles ggml/whisper.cpp *without*
`-DNDEBUG`, leaving asserts in the hot loops. sherpa-onnx arrives as a prebuilt release `.so`
and is unaffected. The result is a ~15× handicap applied to Whisper only:

| | debug | release |
|---|---|---|
| Whisper `tiny.en` decode | 3424 ms | 201 ms |
| Parakeet 110M decode | 147 ms | 143 ms |

`app/build.gradle` has `testBuildType = project.hasProperty('testRelease') ? 'release' : 'debug'`.
**Always benchmark with `-PtestRelease`.** The in-app screen is fine either way when the user
runs a release build, but say so in the UI if the build is debuggable
(`ApplicationInfo.FLAG_DEBUGGABLE`) — otherwise the numbers are quietly wrong.

Other traps:

- **Always warm up.** First run pays page-faults and lazy init. `ParakeetBenchmarkTest` runs the
  model once and times the second run; keep that.
- **One model at a time, close in between.** Loading 600M Parakeet next to Whisper `small.en`
  on an 8 GB S23 will get the process killed. The existing loop already does this.
- **Gradle's `connectedAndroidTest` wipes app data** on variant switches, deleting downloaded
  models. To iterate without re-staging, install manually and use
  `adb shell am instrument -w -e class ParakeetBenchmarkTest org.futo.voiceinput.dev.test/androidx.test.runner.AndroidJUnitRunner`.

## Constraints to respect

- Parakeet is **English-only** and ignores the glossary — no prompt conditioning. Don't offer it
  for multilingual comparisons.
- Parakeet is **arm64-v8a only** (the vendored AAR was stripped to one ABI). Guard with
  `isParakeetSupported()`.
- `decodingMethod` is currently `greedy_search`. Hotwords would need `modified_beam_search` plus
  a shipped BPE vocab; out of scope unless the benchmark is meant to compare those too.
- Don't regress the recognizer. Everything here lives in settings; `AudioRecognizer` should not
  need to change.

## Device notes

Two devices on wireless adb (`source /Volumes/samsungt5-512gb-ssd-apple/code/cli-tools/zsh-scripts/adb-wireless.zsh`):

- Z Fold `SM-F976W` — `adb-RFGL7292SWL-dAo6qj._adb-tls-connect._tcp`
- S23 `SM-S911W` (SM8550, 7.2 GB) — `adb-RFCW11GA1DB-ct9OzA._adb-tls-connect._tcp`, the minimum target

Both have all four English models staged in `filesDir` and the release build installed as
`org.futo.voiceinput.dev`. Release is not debuggable, so `run-as` needs the debug build
installed first; `adb install -r` between debug and release preserves data (same debug signing key).

**The user tests dictation themselves — do not automate speech testing on their phones.** Give
them a checklist.
