# Newer ASR models for FUTO Voice Input — research report

**Date:** 2026-08-13
**Branch:** `research/asr-model-survey` (off `local-dev`)
**Scope:** Is there an open-source ASR model newer/better than the Whisper tiny/base/small
models this app ships, that runs on-device on a Galaxy S23 (minimum target) and is either
*more accurate* or *clearly faster*? English-only candidates are in scope.

---

> **Update 2026-08-13 — implemented and measured on real hardware.** The Parakeet engine now
> exists behind a setting (see §10). Release builds, same 5.86 s English clip, decode time is
> the steady-state second run:
>
> | Model | Z Fold (SM-F976W) | | S23 (SM-S911W, SM8550) | |
> |---|---|---|---|---|
> | | **decode** | RTF | **decode** | RTF |
> | Whisper English-39 (`tiny.en`) | 201 ms | 0.034 | 500 ms | 0.085 |
> | Whisper English-74 (`base.en`) | — | — | 826 ms | 0.141 |
> | Whisper English-244 (`small.en`) | 1355 ms | 0.231 | 2385 ms | 0.407 |
> | **Parakeet 110M** | **143 ms** | **0.024** | **213 ms** | **0.036** |
> | Parakeet 600M | 348 ms | 0.059 | 2211 ms | 0.378 |
>
> **Parakeet 110M is the clear winner on both devices**: 9.5× faster than `small.en` on the
> Fold and **11.2× faster on the S23**, while also beating `tiny.en` outright. On the minimum
> target it turns the best English model from a 2.4-second wait into 0.2 seconds.
>
> **Parakeet 600M does not survive contact with the S23.** Decode is 2211 ms — no better than
> `small.en` — and model load takes **7.2 seconds** (vs 1.9 s on the Fold), on a device that had
> 159 MB free at the time. It did not crash, but it is not a sensible daily driver on 8 GB.
> Keep it opt-in; 110M is the recommendation on both devices.
>
> The §7.1 "measure first" step is done. The extrapolated S23 estimates below were roughly right
> for the 110M and much too optimistic for the 600M.

## 1. Bottom line

**Yes — and the gap is not marginal.**

The recommendation is **NVIDIA Parakeet** for English, run through **sherpa-onnx**, added
*alongside* the existing whisper.cpp engine rather than replacing it.

Two concrete model swaps, both English-only:

| Slot | Today | Proposed | Accuracy (8-dataset avg WER, lower better) | Params | On-disk |
|---|---|---|---|---|---|
| New default | English-74 (`base.en`) — 10.32% | **parakeet-tdt_ctc-110m** | **7.50%** | 110M | 178 MB (GGUF q8_0) / ~250 MB (ONNX int8) |
| "Most accurate" | English-244 (`small.en`) — ~8.6% | **parakeet-tdt-0.6b-v2** | **6.05%** | 600M | 638 MB (GGUF q8_0) / ~671 MB (ONNX int8) |

The 110M model is **more accurate than the app's current *largest* English model while
being less than half its parameter count**, and its architecture is dramatically cheaper
per second of audio. That is the headline: this is not an accuracy-vs-speed trade, it wins
on both axes at once.

Supporting datapoint that matters most for the hard constraint — an independent
multi-platform on-device benchmark run on a **Galaxy S10 (Exynos 9820, 2019, 8 GB)**, i.e.
hardware substantially *slower* than the S23 floor:

| Model | Engine | RTF | Inference |
|---|---|---|---|
| Parakeet TDT 0.6B v3 | sherpa-onnx | **0.09** | 2,841 ms |
| Whisper Small | sherpa-onnx | 0.41 | 12,329 ms |
| Whisper Base | sherpa-onnx | 0.13 | 3,917 ms |
| Moonshine Tiny | sherpa-onnx | 0.05 | 1,363 ms |

Parakeet 0.6B is **~4.5× faster than Whisper Small on a seven-year-old phone** while being
~2.5 WER points more accurate. On an S23 (Cortex-X3, ~2.5–3× the single-core of the 9820)
expect roughly RTF 0.03–0.04 for the 0.6B and well under 0.01 for the 110M — i.e. a 5-second
dictation transcribed in ~150–200 ms and ~50 ms respectively. Those are estimates by
scaling, not measurements; see §7 for the benchmark you should actually run.

There is one real feature regression to decide on (the personal dictionary / glossary) —
see §6.4. It has a good mitigation.

---

## 2. What the app actually is today

Established by reading the code, not assumed.

### Engine

- **whisper.cpp + ggml, vendored and forked**, at `app/src/main/cpp/ggml/`
  (`whisper.cpp` 238 KB, `ggml.c` 630 KB, plus `ggml-alloc.c`, `ggml-backend.c`,
  `ggml-quants.c`).
- The vintage is **pre-GGUF, roughly whisper.cpp v1.5.x (late 2023 / early 2024)**: the
  file layout is the old flat `ggml.c` + `ggml-quants.c` set, and the GPU macro is
  `GGML_USE_CUBLAS` (renamed to `GGML_USE_CUDA` upstream shortly after). Current upstream
  is the **v1.9.x** series.
- **CPU only.** `app/src/main/cpp/CMakeLists.txt:48` links exactly `android` and `log` —
  no Vulkan, OpenCL, CoreML, or OpenVINO backend is compiled in. Every one of those
  `#ifdef GGML_USE_METAL` / `WHISPER_USE_COREML` blocks in `whisper.cpp` is dead code here.
- Built at `-O3` (`CMakeLists.txt:15` and `app/build.gradle:30`), NEON forced on armv7
  (`CMakeLists.txt:40-42`). `minSdk 24`, `compileSdk 35`, `ndkVersion 28.2.13676358`.
- **No `abiFilters`** in `app/build.gradle` — the native lib is built and packaged for all
  four ABIs (arm64-v8a, armeabi-v7a, x86, x86_64). Relevant to APK size if a second
  runtime is added.
- v1.3.7 added **16 KB page size support** (CHANGELOG). Any new native `.so` must preserve
  that alignment.

### Fork deltas vs upstream whisper.cpp

These are the customisations that any engine upgrade must carry forward
(`app/src/main/cpp/ggml/whisper.h`):

- `allowed_langs` / `allowed_langs_size` (whisper.h:336-337, 479-480) — constrained
  language auto-detect across a user-selected subset.
- `whisper_partial_text_callback` + `partial_text_callback_user_data` (whisper.h:406,
  511-512) — live partial decode text, which drives the in-progress text in the recognizer
  UI.
- `whisper_full_n_segments_from_state` (whisper.h:573) — used by the partial callback to
  stitch multi-segment partials.
- `WHISPER_LOG_ERROR` redirected to the app's `AKLOGE` (`whisper.cpp:128`).

### Model format and loading

- Format: **legacy ggml `.bin`**, not GGUF. Loaded by
  `whisper_init_from_buffer` over an mmap'd `MappedByteBuffer`
  (`voiceinput.cpp:74-89`, `ml/WhisperModel.kt:33-74`).
- Quantization: **q8_0** for every shipped model.
- All shipped models are **FUTO "ACFT" fine-tunes** of Whisper, existing specifically so
  the `audio_ctx` truncation trick works. Per `futo-org/voice-input-models`, models *not*
  ACFT-tuned "will work with long dictations (30s) but shorter dictations (under 15s) will
  exhibit infinite repetition or a long delay at the end." Only whisper tiny/base/small are
  supported as bases; large is explicitly excluded as too big for phones.

### Catalog (`Util.kt:250-404`)

| UI name | File | Bundled? |
|---|---|---|
| English-39 (default) | `tiny_en_acft_q8_0.bin.not.tflite` | yes, 43.5 MB asset |
| English-74 | `base_en_acft_q8_0.bin` | download |
| English-244 | `small_en_acft_q8_0.bin` | download |
| Multilingual-39 / -74 (default) / -244 | `tiny_acft_q8_0.bin`, `base_acft_q8_0.bin`, `small_acft_q8_0.bin` | download |

Downloads come from `https://voiceinput.futo.org/VoiceInput/<file>`
(`downloader/DownloadActivity.kt:342`) and are SHA-256 verified against
`ModelDataGGML.digest`. `startModelDownloadActivity` (`Util.kt:119-135`) passes a **flat
list of one filename per model** — a multi-file model does not fit that shape as written.

### Decode configuration (`voiceinput.cpp:121-155`)

- Greedy, or beam search with `beam_size = 5` (`DecodingMode` in `ggml/WhisperGGML.kt`).
- `max_tokens = 256`, `temperature_inc = 0.0f` (fallback temperature sampling disabled).
- `audio_ctx = min(1500, ceil(n_samples / 320) + 32)` — the ACFT speed trick: shrink the
  encoder context to the actual utterance length instead of Whisper's mandatory 30 s.
- `n_threads = nproc`, clamped to `[2,16]`, default 6.
- `initial_prompt` carries the personal dictionary as `"(Glossary: a, b, c)"`
  (`ml/WhisperModel.kt:150-151`).

### Audio path (`AudioRecognizer.kt`)

`AudioRecord` at **16 kHz mono** → `FloatBuffer` (30 s initial, grows by 30 s chunks) →
`FloatArray` straight into JNI. Silero VAD via the prebuilt `vad-release.aar`
(konovalov), 480-sample frames at 16 kHz, used for auto-stop.

**This is exactly the input contract every candidate below wants** (16 kHz mono float32),
so the audio front-end needs no changes for any option in this report.

### Language fallback

Multilingual model runs first; an `abort_callback` bails the moment the detected language
is `en`, and the English model is re-run from scratch (`ml/WhisperModel.kt:155-184`,
`voiceinput.cpp:210-221`). Relevant because a Parakeet English model slots into the
"fallback English model" position cleanly.

---

## 3. Evaluation criteria

From the constraints given:

1. **Accuracy ≥ current**, or clearly faster at equal accuracy. Baseline to beat is
   English-244 (`small.en`).
2. **Runs on a Galaxy S23** (Snapdragon 8 Gen 2 for Galaxy, 8 GB RAM) — that is the floor,
   not the target. Daily driver is an S26.
3. **Speed matters a lot.**
4. Considers RAM, model size, NPU/GPU availability, realtime factor.
5. Usable license, and weights that actually exist in a mobile-loadable format.
6. English-only is acceptable.

A note on the WER numbers used throughout: all "average WER" figures are the **8-dataset
Open ASR / ESB average** (AMI, Earnings-22, GigaSpeech, LibriSpeech clean, LibriSpeech
other, SPGISpeech, TED-LIUM, VoxPopuli) so they are comparable to each other. Whisper
`small.en` at ~8.6% is taken from the Nemotron paper's ESB batch evaluation; one secondary
blog source claims 6.8% for the same model, which is inconsistent with the
tiny.en → base.en → large-v3 ladder (12.81 → 10.32 → 7.44) and I treat it as wrong.
Nothing in the recommendation flips if small.en is really 6.8%: parakeet-tdt-0.6b-v2 at
6.05% still wins, and 110m at 7.50% would then be "close on accuracy, several times
faster" rather than "better on both."

---

## 4. Candidates

### 4.1 Summary table

Sorted by accuracy. **Bold** = viable on the S23.

| Model | Params | Avg WER | License | Format available | Runtime | Verdict |
|---|---|---|---|---|---|---|
| Granite Speech 4.1 2B | 2B | 5.33% | Apache-2.0 | HF/torch | — | ✗ far too big |
| Cohere Transcribe | 2B | 5.42% | Apache-2.0 | HF/torch | — | ✗ too big |
| Canary-Qwen-2.5B | 2.5B | 5.63% | CC-BY-4.0 | HF/torch | — | ✗ too big |
| Qwen3-ASR-1.7B | 1.7B | 5.76% | Apache-2.0 | ONNX | ORT | ✗ too big (1.9 GB int8) |
| **parakeet-tdt-0.6b-v2** | **600M** | **6.05%** | **CC-BY-4.0** | **GGUF, ONNX int8** | **parakeet.cpp / whisper.cpp / sherpa-onnx** | **✓ top accuracy pick** |
| **parakeet-tdt-0.6b-v3** | **600M** | **6.32%** | **CC-BY-4.0** | **GGUF, ONNX int8** | **same** | **✓ if 25-lang coverage wanted** |
| Kyutai STT 1B | 1B | 6.40% | CC-BY-4.0 | torch/candle/MLX | — | ✗ no mobile runtime, too big |
| **Moonshine v2 Medium** | **245M** | **6.65%** | **MIT (English)** | **ONNX / .ort** | **ORT, sherpa-onnx** | **✓ strong runner-up** |
| distil-whisper large-v3.5 | ~750M | 7.21% | MIT | **ggml .bin** | whisper.cpp | ~ drop-in but big & slower |
| whisper large-v3 | 1.55B | 7.44% | MIT | ggml .bin | whisper.cpp | ✗ excluded by FUTO as too big |
| **parakeet-tdt_ctc-110m** | **110M** | **7.50%** | **CC-BY-4.0** | **GGUF, ONNX** | **parakeet.cpp / sherpa-onnx** | **✓✓ best size:accuracy:speed** |
| whisper large-v3-turbo | 809M | 7.83% | MIT | ggml .bin | whisper.cpp | ~ 1.0 GB, RTF 0.60 on S10 |
| **Moonshine v2 Small** | **123M** | **7.84%** | **MIT (English)** | **ONNX / .ort** | **ORT, sherpa-onnx** | **✓ viable** |
| Nemotron-3.5-ASR-streaming | 600M | 8.20% (streaming int4) | OpenMDW-1.1 | ONNX, GGUF | ORT / parakeet.cpp | ~ streaming-first |
| **whisper small.en — CURRENT BEST** | 244M | ~8.6% | MIT | ggml .bin | whisper.cpp | baseline |
| **whisper base.en — CURRENT DEFAULT-ISH** | 74M | 10.32% | MIT | ggml .bin | whisper.cpp | baseline |
| Moonshine v1 Base | 61M | 10.07% | MIT | ONNX | ORT, sherpa-onnx | ~ superseded by v2 |
| **whisper tiny.en — CURRENT BUNDLED** | 39M | 12.81% | MIT | ggml .bin | whisper.cpp | baseline |
| **Moonshine v2 Tiny** | **34M** | **12.01%** | **MIT** | **ONNX / .ort** | **ORT, sherpa-onnx** | ✓ bundled-asset replacement |
| SenseVoice Small | 234M | not on ESB | — | ONNX | sherpa-onnx | ~ fast, EN/ZH/JA/KO/YUE |
| Streaming Zipformer (icefall) | ~70M | weak off-domain | Apache-2.0 | ONNX | sherpa-onnx | ✗ LibriSpeech-domain; 41% WER off-domain |

### 4.2 NVIDIA Parakeet — the one you asked about specifically

**Licensing: clean.** `parakeet-tdt-0.6b-v2`, `-v3` and `tdt_ctc-110m` are all
**CC-BY-4.0** per their NVIDIA model cards — commercial use permitted, attribution
required. (One secondary source claimed Apache-2.0 for v3; the model card says CC-BY-4.0
and that governs.) Attribution slots into the existing `Credits.kt` /
`assets/license-list.html` machinery. The streaming `parakeet_realtime_eou_120m-v1` is
under the *NVIDIA Open Model License* instead — different terms, worth reading separately
if you go for that one. `nemotron-3.5-asr-streaming-0.6b` is OpenMDW-1.1.

**Mobile-loadable weights: yes, in three independent formats.** This was the key
uncertainty and it is fully resolved:

1. **GGUF for ggml** — `mudler/parakeet-cpp-gguf` publishes every variant at f32/f16/q8_0/
   q6_k/q5_k/q4_k. Sizes: `tdt_ctc-110m` **f16 267.5 MB, q8_0 177.8 MB, q4_k 131.4 MB**;
   `tdt-0.6b-v2` q8_0 **638 MB**; `tdt-0.6b-v3` q8_0 **675 MB**;
   `realtime_eou_120m-v1` q8_0 176 MB. Self-contained — **no separate tokenizer/vocab
   file**.
2. **In-tree in whisper.cpp** — upstream **v1.9.0 added Parakeet support** (PR #3735, by
   danbev), with streaming following in #3900. It uses the ggml `.bin` format, ships a
   Python conversion script, and exposes a *separate* API surface —
   `parakeet_init_from_file()`, `parakeet_full()`, `parakeet_reset_state()` — rather than
   extending `whisper_full`. Reported 1.96% WER on LibriSpeech. English-only, no
   timestamps in the initial PR.
3. **ONNX for sherpa-onnx** — `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` and
   friends, with **prebuilt Android arm64-v8a APKs** demonstrating it
   (`sherpa-onnx-1.12.40-arm64-v8a-simulated_streaming_asr-multi-parakeet_tdt_0.6b_v3.apk`).

**Runtime: three options, all of which are ports of things this project already uses or
could use.** See §5.

**Why it suits a dictation keyboard specifically, beyond WER:**

- **No 30-second padding.** FastConformer processes the actual audio length. The entire
  `audio_ctx` hack and the ACFT fine-tune requirement exist only to work around Whisper's
  fixed 30 s context — with Parakeet the problem does not exist, and *any* stock checkpoint
  works on short clips without infinite repetition.
- **No hallucination on silence.** A transducer emits tokens frame-by-frame against audio
  evidence; no audio, no tokens. Whisper's autoregressive decoder is known to invent fluent
  sentences during silence and pauses. For dictation — which starts and ends with silence
  and is full of mid-thought pauses — this shows up more often than the leaderboard gap does.
- **Punctuation and capitalisation** are native to `tdt_ctc-110m` (the "PnC" model) and the
  0.6B models.
- **Memory is lower than NeMo's**: parakeet.cpp measured 563 MB peak RSS for the 110M model
  at f32 vs NeMo's 1,650 MB, roughly halving across the family, "lower still once
  quantized."

**Per-dataset WER for `parakeet-tdt_ctc-110m`** (NVIDIA model card), which is where the
7.50% average comes from:

| LS clean | LS other | AMI | Earnings-22 | GigaSpeech | TED-LIUM | SPGI | VoxPopuli | **avg** |
|---|---|---|---|---|---|---|---|---|
| 2.4 | 5.2 | 15.88 | 12.42 | 10.52 | 4.16 | 2.54 | 6.91 | **7.50** |

**The RAM caveat, stated plainly.** sherpa-onnx issue #2626 reports
`parakeet-tdt-0.6b-v3-int8` consuming **1.23 GB RAM** at load on iOS despite a 1.02 GB file
— runtime buffers on top of weights. On an 8 GB S23 that is survivable for a foreground
recognizer activity but is genuinely risky for a process the system may want to reclaim,
and it is the single strongest argument for making **110m the default** and 0.6B the
opt-in "most accurate" tier. The 110M model's footprint is ~4× smaller.

### 4.3 Moonshine v2 — the strongest runner-up

Purpose-built for exactly this use case (low-latency, short utterances, edge devices), and
the numbers are good:

| Variant | Params | Avg WER | Latency (M3 MacBook) | vs Whisper |
|---|---|---|---|---|
| Tiny | 33.6M | 12.01% | 50 ms | 5.8× faster than Whisper Tiny (289 ms) |
| Small | 123.4M | 7.84% | 148 ms | 13.1× faster than Whisper Small |
| Medium | 244.9M | 6.65% | 258 ms | 43.7× faster than Whisper Large v3 (11,286 ms) |

- **License: MIT for the code and for the English models.** Non-English Moonshine models
  are under a non-commercial license — irrelevant here since English-only is acceptable,
  but worth knowing if scope ever widens.
- Format: ONNX, converted to the memory-mappable OnnxRuntime `.ort` flatbuffer. Also
  supported by **sherpa-onnx** (including a QNN/NPU export as of 1.13.5) and by a
  third-party `moonshine.cpp` (ONNXRuntime-based, not ggml).
- Moonshine Tiny was the *fastest* model in the S10 Android benchmark at RTF 0.05.
- Downside vs Parakeet: **no timestamps**, and Moonshine Medium at 6.65% is still behind
  parakeet-tdt-0.6b-v2 at 6.05% while parakeet's 110M beats Moonshine Small at half again
  the accuracy-per-param. One HN report also found Moonshine "noticeably slower... and
  didn't seem as accurate" than Parakeet V3 on older x86.

Moonshine v2 Medium is a completely defensible pick and would be my choice if the 0.6B
Parakeet's RAM proves unworkable and you also want better-than-current accuracy in a
245M-param envelope. **Moonshine v2 Tiny (34M, 12.01%, RTF 0.05) is also the obvious
replacement for the bundled 43.5 MB `tiny.en` asset** — same accuracy class as tiny.en's
12.81%, far faster, and smaller on disk.

### 4.4 Newer Whisper distillations — the low-effort option, and why it loses

`distil-large-v3.5` is **published directly in ggml format**
(`distil-whisper/distil-large-v3.5-ggml`, plus an all-quants mirror), MIT-licensed. It is
7.21% avg WER, ~1.5× faster than large-v3-turbo. In principle this is the smallest possible
change — a new `ModelDataGGML` entry and a URL.

It loses anyway:

- **It is not ACFT-tuned**, so per FUTO's own model docs it will exhibit infinite
  repetition or long end-delays on the sub-15-second clips that are ~all of dictation. You
  would need to fine-tune it yourself to ship it.
- Size: distil-large-v3.5 is ~750M params; large-v3-turbo is 809M / ~1.0 GB and measured
  **RTF 0.60 on the S10** — six times slower than Parakeet 0.6B for *worse* accuracy
  (7.83% vs 6.32%).
- 7.21% is worse than parakeet-tdt-0.6b-v2's 6.05% at more than twice the parameters.

Not worth it. The distillation direction has been overtaken by the transducer models.

### 4.5 Everything else, briefly

- **Kyutai STT** (1B / 2.6B, CC-BY-4.0, 6.40%) — genuinely good streaming model, but there
  is no mobile runtime (torch/candle/MLX only) and 1B+ is over budget. Rule out.
- **Zipformer / icefall streaming** (sherpa-onnx's traditional strength) — small and fast,
  but the published English streaming models are LibriSpeech-domain and fall apart
  off-domain (41% WER on accented English before fine-tuning). Accuracy would regress
  badly vs your current models. Rule out.
- **SenseVoice Small** (234M, sherpa-onnx, RTF 0.06 on the S10) — very fast, multilingual
  EN/ZH/JA/KO/YUE, non-autoregressive. Not evaluated on the ESB suite so I can't place it
  on the accuracy ladder, and it has no punctuation/casing story as strong as Parakeet's.
  Interesting only if you later want a fast CJK path.
- **Nemotron-3.5-ASR-streaming-0.6b** (June 2026, OpenMDW-1.1, 40 locales) — the streaming
  sibling of Parakeet. 8.20% WER streaming int4, 0.67 GB at int4, ONNX Runtime. Supported
  by parakeet.cpp. Worth revisiting *if* you decide to build true streaming dictation, but
  as a batch model it is beaten by parakeet-tdt-0.6b-v2 on both accuracy and size.
- **`parakeet_realtime_eou_120m-v1`** (120M, streaming RNNT with **end-of-utterance
  detection**, 80–160 ms latency) — architecturally the most interesting thing here for a
  voice keyboard: it would give live text *and* replace the Silero VAD auto-stop with a
  model-native "user finished speaking" signal. But: NVIDIA Open Model License (not
  CC-BY-4.0), **no punctuation or capitalisation**, and HF discussions report high WER and
  EOU false positives unless the streaming buffer implementation is exactly right. File
  under "future, once the batch path is shipped."
- **2B+ class** (Granite Speech, Cohere Transcribe, Canary-Qwen, Qwen3-ASR-1.7B, Voxtral,
  ARK-ASR) — all top the leaderboard, all out of budget for a phone keyboard. Qwen3-ASR
  0.6B's int8 ONNX is 1.9 GB and measured RTF 0.53 on the S10; its pure-C/NEON build was
  RTF 11.28. Rule out.

---

## 5. Runtime options

Three ways to actually run Parakeet on the S23. They are not equally sized pieces of work.

### Route A — bump the vendored whisper.cpp to v1.9.x and use its in-tree Parakeet API

**Appeal:** one runtime, one build, ggml stays the only ML dependency, models stay `.bin`,
and you'd also pick up VAD improvements, better Android inference perf, and the option of a
Vulkan backend.

**Cost:** this is the expensive route. The fork is at ~v1.5.x. Between then and v1.9.x,
ggml was restructured substantially (CPU backend split out, GGUF, backend registry), and
whisper.cpp moved to `src/`. The three fork patches (`allowed_langs`,
`partial_text_callback`, `whisper_full_n_segments_from_state`) all touch decode internals
that moved. And Parakeet's API is *separate* (`parakeet_full()`), so you get no reuse of the
whisper call path anyway — you write a new JNI surface regardless.

Realistically a multi-week job whose main deliverable is "we're on a modern whisper.cpp
again." Valuable, but it is a separate project from "ship a better English model."

### Route B — add sherpa-onnx alongside whisper.cpp ★ recommended

**Appeal:**

- **Prebuilt Android AAR on Maven Central**: `com.k2fsa.sherpa.onnx:sherpa-onnx-android`
  (Apache-2.0), currently ~1.13.x. One Gradle line; no new CMake, no new NDK build, no
  vendored source.
- Every candidate in this report is already exported and hosted for it: Parakeet TDT
  (`tdt_ctc-110m`, `0.6b-v2`, `0.6b-v3`), Moonshine v1 and v2, Whisper, SenseVoice, Zipformer.
- **All the credible on-device Android numbers in this report come from sherpa-onnx**, so
  the performance is measured rather than extrapolated from x86.
- **"Simulated streaming"** demos exist for Parakeet — re-running a non-streaming model
  over a growing buffer — which is how you'd restore the live partial-text UX.
- **Hotwords / contextual biasing** for transducers (`--hotwords-file`, `--hotwords-score`,
  `--modeling-unit=bpe`, `--bpe-vocab`, requires `modified_beam_search`), including
  explicit NeMo/TDT support via `OfflineTransducerModifiedBeamSearchNeMoDecoder`. This is
  the replacement for the personal dictionary — see §6.4.
- **A path to the NPU.** sherpa-onnx 1.13.4/1.13.5 added *"Export Parakeet TDT models to
  QNN"* and *"C++ runtime for Parakeet TDT models with QNN"*, plus the same for Moonshine,
  and publishes `sherpa-onnx-qnn-parakeet-tdt_ctc-110m-5s-android-aarch64` artifacts. QNN
  is Qualcomm's Hexagon NPU stack — which both the S23 (8 Gen 2) and S26 have.

**Cost:** a second inference runtime in the app. `libonnxruntime.so` is ~15 MB plus ~4.6 MB
for the JNI lib, **per ABI** — and this project currently has no `abiFilters`, so it would
ship four copies unless that's fixed. ONNX model files are also larger than the equivalent
GGUF (~671 MB int8 vs 638 MB q8_0 for 0.6B; ~250 MB vs 178 MB for 110m).

**Caveat on one widely-quoted number.** The S10 benchmark reports Whisper Tiny under
whisper.cpp at **RTF 3.52 / 105.6 s**, and concludes "sherpa-onnx is 51× faster than
whisper.cpp." That is not believable — whisper tiny under whisper.cpp on a 2019 flagship is
normally RTF 0.3–0.6, and this app demonstrably does better than RTF 3.5 today. Their
whisper.cpp build was almost certainly misconfigured (single-thread, no NEON, or a debug
build). **Do not use that comparison to justify the switch.** The Parakeet-vs-Whisper
comparison *within* sherpa-onnx (RTF 0.09 vs 0.41) is internally consistent and is the one
that matters.

### Route C — vendor parakeet.cpp (MIT, ggml) and build it with the NDK

`mudler/parakeet.cpp` from the LocalAI team: a C++17 ggml port covering **every** Parakeet
variant (TDT/CTC/RNNT/hybrid + cache-aware streaming), validated at **WER 0 against NeMo** —
byte-identical transcripts on 7 of 10 models.

**Appeal:**

- **MIT**, ggml-based — philosophically identical to what the app already does.
- **Smallest models on disk** (q8_0 GGUF: 178 MB for 110m, 638 MB for 0.6B-v2), all
  self-contained.
- The C API is a near-perfect match for this codebase's existing JNI shape:
  ```c
  parakeet_ctx *parakeet_capi_load(const char *model_path);
  char *parakeet_capi_transcribe_pcm(parakeet_ctx *ctx, const float *samples,
                                     int n_samples, int sample_rate, int decoder);
  parakeet_stream *parakeet_capi_stream_begin(parakeet_ctx *ctx);
  char *parakeet_capi_stream_feed(parakeet_stream *s, const float *pcm, int n, int *eou);
  ```
  `transcribe_pcm(float*, n, 16000)` is exactly what `AudioRecognizer` already produces.
- Measured 1.4–1.9× faster than NeMo on CPU at ~2× lower peak RAM.

**Cost / risk:** **no official Android or NDK support.** Linux/macOS/Windows only; the one
mobile datapoint is a third-party iOS demo. It vendors its own copy of ggml at
`third_party/ggml`, so you'd have two ggml versions in one APK unless you unify them. All
published benchmarks are 20-core x86 or GB10/M4 — nothing on ARM phones. It would probably
build (it's CMake + C++17 + ggml, and ggml targets Android well), but "probably builds" is
a day of yak-shaving followed by unmeasured performance.

### Route D — the hybrid ★ what I'd actually ship

Keep whisper.cpp **completely untouched** for the multilingual models. Add sherpa-onnx +
Parakeet **only for the English slots**. Existing multilingual behaviour, the ACFT models,
the language-bail logic, and the whole `audio_ctx` path carry on working exactly as they do
today; the new engine is additive and independently revertible.

This is Route B, scoped so that a bad outcome costs you one settings entry rather than the
app's core.

---

## 6. Recommendation

**Add Parakeet English models via sherpa-onnx, alongside the existing whisper.cpp engine.**

**Model choices:**

1. **`parakeet-tdt_ctc-110m`** as the new recommended English default. 7.50% avg WER beats
   the current *best* English model (small.en, ~8.6%) at 110M vs 244M params, with
   punctuation and capitalisation, 178 MB GGUF / ~250 MB ONNX int8. On the S10 datapoint
   scaled down from the 0.6B, this should be comfortably sub-100 ms for a typical
   dictation on an S23.
2. **`parakeet-tdt-0.6b-v2`** as the "most accurate" English tier, replacing English-244.
   6.05% avg WER — better than Whisper large-v3 (7.44%) at 600M params. Gate it behind the
   RAM check in §7; ~1.2 GB runtime RSS is real.
3. Leave English-39 (bundled `tiny.en`) alone for now — it's the zero-download fallback and
   the 43.5 MB asset is doing a job. If you later want to shrink or speed up the bundled
   model, **Moonshine v2 Tiny** (34M, 12.01%, MIT, RTF 0.05) is the swap.
4. Leave all three multilingual Whisper models alone. Parakeet v3's 25 European languages
   do not cover Whisper's set, and this is not the problem being solved.

**Why Parakeet over Moonshine v2 Medium**, since both beat the current best:

- Better accuracy at the top end (6.05% vs 6.65%).
- Better accuracy *per parameter* at the small end — 110M at 7.50% vs Moonshine Small's
  123M at 7.84%.
- Native timestamps (Moonshine has none), native punctuation/casing.
- Three independent runtime paths (ggml in-tree, parakeet.cpp, sherpa-onnx) vs Moonshine's
  ONNX-only story — that's real insurance against any one of them going stale.
- Both have a QNN/NPU export in sherpa-onnx, so that's a wash.

Moonshine v2 Medium remains the fallback if Parakeet's memory profile disappoints on real
hardware.

### 6.4 The one real regression: the personal dictionary

The glossary feature works by stuffing `"(Glossary: foo, bar)"` into Whisper's
`initial_prompt` (`ml/WhisperModel.kt:150-151`, `voiceinput.cpp:157-159`). **Parakeet has no
prompt conditioning** — a transducer's prediction network is conditioned only on tokens it
has already emitted. Ported naively, the personal dictionary silently stops working for
Parakeet models.

**Mitigation: sherpa-onnx hotwords.** Transducer contextual biasing is supported
(`hotwordsFile` + `hotwordsScore`, `modelingUnit = "bpe"` + `bpeVocab`, with
`decodingMethod = "modified_beam_search"`), and NeMo/TDT models are explicitly covered.
This is arguably a *better* mechanism than prompt-stuffing: it biases the beam directly
rather than hoping the decoder takes the hint, and it doesn't consume decoder context.

Two consequences to plan for: you must ship the BPE vocab alongside the model, and you must
use `modified_beam_search` rather than greedy — which costs some speed. Given the headroom
(RTF 0.09 vs Whisper Small's 0.41 on a 2019 phone) that is affordable.

Route C (parakeet.cpp) has **no** hotwords equivalent in its public C API. That is a
further point in Route B's favour and is worth weighting heavily if you use the glossary.

---

## 7. Implementation sketch (Route D)

### 7.1 Measure before building

There's already an instrumentation-test fixture to build on:
`app/src/androidTest/assets/audio.floats.bin` plus `features.floats.bin`.

Write a `ModelBenchmarkTest` that runs the same float array through (a) the current
`small_en_acft_q8_0` via `WhisperGGML`, and (b) `parakeet-tdt_ctc-110m` and
`parakeet-tdt-0.6b-v2` via sherpa-onnx, on the actual S23, recording wall-clock,
`Debug.getNativeHeapAllocatedSize()` / PSS, and the transcript. **Do not skip this** — the
entire speed case rests on one third-party benchmark on a different phone, and the 0.6B RAM
figure (1.23 GB) is close enough to uncomfortable on an 8 GB device that it needs a real
number before it ships as a user-selectable option.

Per the standing instruction: this is a checklist item for you to run, not something to
automate against the phone unattended.

### 7.2 Files to touch

| File | Change | Size |
|---|---|---|
| `app/build.gradle` | Add `implementation 'com.k2fsa.sherpa.onnx:sherpa-onnx-android:<ver>'`. Add `ndk { abiFilters 'arm64-v8a' }` (plus `'armeabi-v7a'` if 32-bit devices still matter) so onnxruntime isn't packaged 4×. Verify the AAR's `.so`s are 16 KB-page-aligned — v1.3.7 explicitly added that support. | S |
| `Util.kt:59-70` | `ModelData` currently hard-codes `ggml: ModelDataGGML` **and** a non-null `legacy: ModelDataLegacy`. Introduce a backend discriminator — cleanest is a `sealed interface ModelBackend { Ggml, SherpaTransducer }`, minimal is a nullable `onnx: ModelDataOnnx?` plus branching. `legacy` must become nullable or get a dummy for Parakeet entries. | M |
| `Util.kt:92-135` | `modelNeedsDownloading` and `startModelDownloadActivity` assume **one file per model**. Parakeet ONNX is multi-file (`encoder.onnx`, `decoder.onnx`, `joiner.onnx`, `tokens.txt`, + `bpe.vocab` for hotwords). `startModelDownloadActivity` flattens to `arrayListOf(model.ggml.ggml_file)` — needs to emit N entries with N digests. | M |
| `downloader/DownloadActivity.kt:342` | Base URL is `https://voiceinput.futo.org/VoiceInput/${it}` — you can't publish there from the fork. Either host the model files on a `js-commit/voice-input` GitHub release, or point at the HF repos (`csukuangfj/sherpa-onnx-nemo-parakeet-*`). Either way compute per-file SHA-256 for the existing digest check. | S |
| **new** `ml/ParakeetModel.kt` | Wrap sherpa-onnx `OfflineRecognizer` + `OfflineModelConfig`/`OfflineTransducerModelConfig`. Set `numThreads` from the same `nproc` logic as `voiceinput.cpp:118-119`, `decodingMethod = "modified_beam_search"` when a glossary is set, `hotwordsFile`/`hotwordsScore` from the personal dictionary. Expose `suspend fun run(samples: FloatArray, glossary: String): String` on the existing `inferenceContext` single-thread dispatcher. | M |
| `ml/WhisperModel.kt:94-194` | `WhisperModelWrapper` is concrete and Whisper-specific. Extract the `run(...)` / `close()` contract into an interface and give it a Parakeet implementation. `AudioRecognizer` picks by backend. The `BailLanguageException` fallback path stays whisper-only. | M |
| `settings/pages/Models.kt`, `settings/Settings.kt` | Add the new English entries. **Careful:** `ENGLISH_MODEL_INDEX` is a persisted *list index* (`Util.kt:407-419`) — inserting into the middle of `ENGLISH_MODELS` silently changes which model existing users are on. Append only, or write a migration. | S |
| `RecognizerView.kt` / `RunState` | `ExtractingFeatures` / `ProcessingEncoder` / `StartedDecoding` map loosely onto a one-shot ONNX call; `SwitchingModel` is unused for English-only. Either collapse to a single "processing" state for Parakeet or fake the transitions. | S |
| `assets/license-list.html`, `settings/pages/Credits.kt` | CC-BY-4.0 attribution for the NVIDIA weights, Apache-2.0 for sherpa-onnx and onnxruntime. | S |

### 7.3 Partial results

`ml/WhisperModel.kt` and `WhisperGGML.kt` plumb a `partialResultCallback` that drives live
text in the recognizer overlay. sherpa-onnx's **offline** recognizer has no equivalent.
Three options:

1. **Drop partials for Parakeet.** If a 5 s utterance decodes in ~50–200 ms, partials are
   arguably pointless — the final result lands faster than the partials would have
   rendered. Simplest, and probably correct.
2. **Simulated streaming.** Re-run the offline recognizer on the growing buffer every
   ~500 ms. This is exactly what sherpa-onnx's own `simulated_streaming_asr` Android demos
   do with Parakeet, so it's a supported pattern. Costs CPU proportional to how often you
   re-run.
3. **True streaming** via `parakeet_realtime_eou_120m-v1` (online recognizer + EOU). Best
   UX by far — live text *and* a model-native replacement for the Silero VAD auto-stop —
   but a different license, no punctuation/casing, and reported EOU tuning difficulties.
   Not for v1.

Start with (1), measure, and only add (2) if it actually feels worse.

### 7.4 Effort

- **Spike — one English Parakeet model working behind a hidden setting:** ~1–2 days.
  Gradle dependency, a `ParakeetModel.kt`, sideload the model files to `filesDir` manually,
  hardcode the path, run the benchmark test. This is the decision point; everything after
  is productionisation.
- **Shippable in your fork:** ~1 week on top. Multi-file download + digests, model hosting,
  the `ModelData` refactor, settings entries with index migration, hotwords wiring, APK
  size work (`abiFilters`), credits.
- **Upstreamable to FUTO:** ~2–3 weeks. Same plus: keeping the legacy TFLite migration path
  coherent, the multilingual fallback interaction, hosting on FUTO's own model CDN, and
  a decision from them about carrying a second inference runtime — which is a real
  architectural commitment, not just a diff.

Route C (parakeet.cpp under the NDK) is not obviously less work despite being ggml-native:
you'd trade the multi-file-download and APK-size problems for an unproven Android build,
a duplicated ggml, and no hotwords.

---

## 8. Risks and open questions

1. **0.6B RAM on 8 GB.** 1.23 GB observed on iOS. Must be measured on the S23 before it
   ships as an option. Mitigation: default to 110m; gate 0.6B behind an
   `ActivityManager.MemoryInfo` / `isLowRamDevice` check, or just behind an "advanced"
   warning.
2. **APK size.** onnxruntime is ~15 MB of `.so` per ABI and the project currently ships
   four ABIs. Fixing `abiFilters` is required, not optional, and dropping x86/x86_64 will
   break the emulator for anyone who develops on one.
3. **Two runtimes to maintain.** whisper.cpp for multilingual, onnxruntime for English.
   Real ongoing cost. The alternative (Route A: modernise whisper.cpp and use its in-tree
   Parakeet) removes it but is a much larger job — worth doing eventually and independently.
4. **The glossary.** §6.4. Needs the hotwords path to actually work with the shipped BPE
   vocab; verify in the spike, because if it doesn't, that's a user-visible feature loss.
5. **The 51×-faster claim is not trustworthy.** Flagged in §5 Route B. The case for this
   change should rest on the within-sherpa-onnx comparison and on your own S23 numbers.
6. **Whisper `small.en`'s true ESB average** — I use ~8.6%; one source says 6.8%. Doesn't
   change the recommendation (§3) but means the 110m-vs-current margin is the softest
   number in this report. Your own A/B on real dictation audio settles it better than any
   leaderboard.
7. **NPU is a real but unexercised option.** sherpa-onnx has Parakeet-on-QNN artifacts for
   Android aarch64, and both target phones have Hexagon NPUs. But QNN needs the Qualcomm
   SDK at build time (`QNN_SDK_ROOT`), the docs don't state which HTP versions are
   supported, and there's no prebuilt QNN-enabled AAR — you'd build sherpa-onnx yourself
   with `SHERPA_ONNX_ENABLE_QNN=ON`. Treat as a phase 2 speed lever, not part of v1.
   Given CPU already gets you RTF <0.1, it may never be necessary.
8. **Upstream is still Whisper-only.** Verified: `futo-org/voice-input` is at v1.3.6/1.3.7
   with GGML Whisper + deprecated TFLite, `voice-input-models` documents only
   whisper-tiny/base/small with ACFT, and there are no GitHub releases. A secondary source
   claimed FUTO ships "Parakeet and Quill models" with Verbatim/Tidy/Formatted modes — that
   does not match the code, the model repo, or the official site, and I believe it's a
   conflation with a different app. So this work is not duplicating something FUTO has
   already done, but it also means no upstream precedent to follow.

---

## 9. Sources

Inference stack / upstream state
- [whisper.cpp releases](https://github.com/ggml-org/whisper.cpp/releases) · [v1.9.0](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.0) · [Parakeet support PR #3735](https://github.com/ggml-org/whisper.cpp/pull/3735)
- [futo-org/voice-input](https://github.com/futo-org/voice-input) · [voice-input-models README](https://github.com/futo-org/voice-input-models/blob/main/README.md) · [DeepWiki: model types](https://deepwiki.com/futo-org/voice-input/4.1-model-types-and-selection) · [voiceinput.futo.tech](https://voiceinput.futo.tech/)

Parakeet
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) · [nvidia/parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) · [nvidia/parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)
- [mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp) · [README](https://github.com/mudler/parakeet.cpp/blob/master/README.md) · [LocalAI benchmark writeup](https://localai.io/blog/parakeet-cpp-asr-on-cpu/) · [mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf)
- [transcribe.cpp parakeet docs](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/parakeet.md)
- [HN: Parakeet V3 vs Moonshine Medium](https://news.ycombinator.com/item?id=47146033)

Moonshine
- [moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine) · [Moonshine v2 paper (arXiv 2602.12241)](https://arxiv.org/html/2602.12241v1) · [Moonshine v1 paper (arXiv 2410.15608)](https://arxiv.org/html/2410.15608v1) · [Flavors of Moonshine (arXiv 2509.02523)](https://arxiv.org/html/2509.02523v1)

Benchmarks / leaderboards
- [VoicePing offline transcription benchmark (Android/iOS/macOS/Windows)](https://voiceping.net/en/blog/research-offline-speech-transcription-benchmark/)
- [Open ASR Leaderboard paper (arXiv 2510.06961)](https://arxiv.org/html/2510.06961v4) · [Open ASR Leaderboard space](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [MarkTechPost: best open ASR models 2026](https://www.marktechpost.com/2026/07/23/best-open-speech-recognition-asr-models-in-2026-wer-languages-latency-and-license-compared/)
- [On-device streaming ASR / Nemotron-0.6B (arXiv 2604.14493)](https://arxiv.org/html/2604.14493v2)

sherpa-onnx
- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) · [CHANGELOG](https://github.com/k2-fsa/sherpa-onnx/blob/master/CHANGELOG.md) · [prebuilt Android APKs](https://k2-fsa.github.io/sherpa/onnx/android/prebuilt-apk.html) · [QNN build docs](https://k2-fsa.github.io/sherpa/onnx/qnn/build.html) · [hotwords docs](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html) · [issue #2626 — Parakeet 0.6B int8 RAM](https://github.com/k2-fsa/sherpa-onnx/issues/2626) · [csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8)

Whisper distillations
- [distil-whisper/distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) · [distil-large-v3.5-ggml](https://huggingface.co/distil-whisper/distil-large-v3.5-ggml)

Hallucination / dictation behaviour
- [Parakeet vs Whisper vs Nemotron (OpenWhispr)](https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron)

---

## 10. What was actually built (2026-08-13)

Implemented on this branch, engine selectable at **Settings → Model options → English engine**.

### Shape

- `ml/SpeechModel.kt` — interface both engines implement. `AudioRecognizer.model` is now a
  `SpeechModel?` instead of a `WhisperModelWrapper?`.
- `ml/ParakeetModel.kt` — sherpa-onnx `OfflineRecognizer`, fed the same 16 kHz `FloatArray` the
  Whisper path already produces. No audio front-end changes were needed.
- `ParakeetModels.kt` — model catalog, per-file URLs and SHA-256, ABI guard.
- `settings/pages/Benchmark.kt` — in-app benchmark over a bundled 5.86 s clip.
- `androidTest/ParakeetBenchmarkTest.kt` — the same benchmark as an instrumentation test.
- `app/libs/sherpa-onnx-1.13.5-arm64.aar` — upstream v1.13.5 release AAR repacked to arm64-v8a
  and stripped of the unused c-api/cxx-api libs: 49 MB → 9.9 MB.

The Whisper path is untouched. Parakeet only engages when the recognition is unambiguously
English (multilingual off, or the caller pinned `en`); multilingual and the
detect-then-switch-to-English fallback stay on whisper.cpp. If the Parakeet files are missing or
the ABI is wrong, it falls back rather than failing.

### Corrections to this report found during implementation

- **sherpa-onnx is not on Maven Central.** §5 Route B claimed `com.k2fsa.sherpa.onnx:
  sherpa-onnx-android`; that coordinate 404s. Distribution is a prebuilt AAR attached to GitHub
  releases, so it is vendored as a file dependency instead.
- **APK cost was overstated.** §8 predicted ~15 MB per ABI. Restricting the AAR to arm64-v8a and
  dropping the unused libs gives one `libonnxruntime.so` (21.7 MB) plus
  `libsherpa-onnx-jni.so` (4.8 MB), 9.9 MB compressed in the AAR. No `abiFilters` change was
  needed: whisper.cpp still builds for all four ABIs and `isParakeetSupported()` gates the
  Parakeet engine at runtime.
- **NeMo transducers need `modelType = "nemo_transducer"`.** Using `"transducer"` (the
  icefall/Zipformer path) fails with `'vocab_size' does not exist in the metadata` and takes the
  process down. Cost an hour; caught only because the 600M path was tested rather than assumed.
- **The 110M int8 weights are not on Hugging Face** as individual files — that repo holds only
  `.gitattributes`. They exist solely inside a `.tar.bz2` release archive, which the in-app
  downloader cannot unpack, so they need re-hosting as plain files. The 600M weights *are* on
  Hugging Face and download end-to-end today.
- **`app/src/androidTest/java/FeatureExtractorTestAndroid.kt` was already broken** on
  `local-dev`, referencing a `WhisperModel` class that no longer exists, so `connectedAndroidTest`
  could not compile. Deleted, matching what upstream did.

### Output formatting differs, and it is noticeable

On the same clip:

```
Whisper  : Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
110M     : mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel
600M     : mister Quilter is the Apostle of the Middle Classes, and we are glad to welcome his gospel.
```

Both Parakeet models spell out "mister" instead of "Mr.", and neither capitalises the first
word. The 600M also capitalises mid-sentence nouns. Commas and terminal periods do appear, so
this is not missing punctuation so much as a different normalisation convention from Whisper's.
For dictation this is a real, visible difference and is worth a period of daily use before
switching the default — it is not captured by WER, which is computed on normalised text.

### Not done

- **Hosting for the 110M weights.** `PARAKEET_MODEL_BASE_URL` points at a
  `js-commit/voice-input` release that does not exist yet; until it does, the 110M option only
  works if the files are staged manually. The 600M option downloads from Hugging Face today.
- **Partial results.** The Parakeet path shows no in-progress text. At 143 ms per utterance this
  may not matter; §7.3 lists the options if it does.
- **Hotwords / personal dictionary.** Not wired up. The user does not use the feature, so it was
  left out rather than half-built; `decodingMethod` is `greedy_search`, and enabling hotwords
  would mean switching to `modified_beam_search` and shipping the BPE vocab.
- **QNN / NPU.** Still CPU-only. At RTF 0.024 there is no pressing reason to chase it.
