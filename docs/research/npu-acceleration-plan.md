# Plan: run Parakeet on the Hexagon NPU instead of the CPU

**Date:** 2026-08-13
**Branch:** `research/asr-model-survey`
**Goal:** cut the battery cost of dictation by moving inference off the CPU onto the Qualcomm
Hexagon NPU, **without losing accuracy**.
**Status:** researched, not built. Staged plan with kill criteria below.

---

## Bottom line

**It is buildable, the accuracy risk is lower than expected, and the case for doing it is
currently unproven.** Do stages 0 and 1 before committing to stage 2 — together they are about
a day and a half and they answer both open questions with measurements instead of guesses.

The single biggest risk is not accuracy and not licensing. It is that **dictation is a
duty-cycled workload** — load model, run one 200 ms inference, go idle — and NPU acceleration
is weakest in exactly that pattern. See §4.

---

## 1. What is confirmed to exist

All verified directly against the sherpa-onnx repo and release assets, not from summaries.

**The model we already ship has QNN binaries.** `parakeet-tdt_ctc-110m`, under the
`asr-models-qnn-binary-2` release tag on `k2-fsa/sherpa-onnx`:

| | |
|---|---|
| SoCs | SM8350, SM8450, SM8475, **SM8550**, SM8650, SM8750, SM8850, QCS9100, SA8255, SA8295 |
| Durations | 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30 seconds |
| Branches | CTC and transducer, separately |
| Download | 78–82 MB `.tar.bz2` per combination |

**The S23 is `SM8550`** — confirmed by `adb shell getprop ro.soc.model` (board platform
`kalama`). It is on the list. *(The Z Fold's SoC still needs confirming — it kept dropping off
wireless adb. Both SM8750 and SM8850 have binaries, so it is almost certainly covered either
way, but check before relying on it.)*

**The package contents are trivially simple** — I downloaded and extracted the SM8550 10s build:

```
model.bin     113 MB     (single pre-compiled Hexagon context binary)
tokens.txt     12 KB
info.txt        num_seconds=10s
test_wavs/    0.wav 1.wav 2.wav
```

One model file plus tokens. That is the **same shape as the existing `SherpaModelKind.NemoCtc`
catalog entry**, so `ParakeetModels.kt` needs no structural change — just a new kind and a
per-SoC URL.

**sherpa-onnx has real support, not just exported weights:** C++ runtimes for Parakeet CTC
(#3688) and TDT (#3720), Kotlin API surface (`provider = "qnn"`, `QnnConfig(backendLib,
systemLib, …)`, `prependAdspLibraryPath`), and QNN paths in the Android demo apps.

---

## 2. The accuracy question — answered, and it is better news than expected

This is the hard constraint, so it goes first.

**The QNN export is W8A16, statically calibrated.** From
`.github/workflows/export-parakeet-tdt-ctc-qnn.yaml`:

```
qnn-onnx-converter \
  --input_network ./model.onnx \
  --input_list model.txt \        # <- real audio calibration data (.raw files)
  --use_native_input_files \
  --act_bitwidth 16 \             # <- 16-bit activations
  --bias_bitwidth 32              # <- 32-bit bias
```

Weights are 8-bit (converter default), activations **16-bit**, bias 32-bit. That is Qualcomm's
recommended accuracy-preserving configuration for HTP, and it is *higher* activation precision
than a typical int8-everything export. It is also **statically calibrated on real audio** rather
than dynamically quantized.

What we run today is `parakeet_tdt_ctc_110m_int8.onnx` — dynamic int8.

So the two are different quantization schemes, and **W8A16-with-calibration is the more
conservative of the two on paper**. The literature backs this direction: int8 causes measurable
WER degradation on transformer-based ASR but transducer/RNN-T style models quantize well, and
16-bit activations are specifically the mitigation for the activation-outlier problem that hurts
conformer encoders.

**But nobody has published a WER for this specific export.** That is the gap, and it is not
worth speculating about because it is *cheaply measurable*:

- The package ships `test_wavs/` for exactly this purpose
- We already have `ParakeetBenchmarkTest` and the in-app benchmark screen
- The record-and-compare spec (`record-and-compare-benchmark-spec.md`) is designed for
  precisely this comparison

**Kill criterion:** if QNN transcripts differ from CPU transcripts on real dictation in any way
that is not trivially formatting, stop. Accuracy is not being traded here.

---

## 3. The obstacles, and what is actually true about each

### 3.1 No prebuilt QNN AAR — real, and it is the main build cost

v1.13.5 ships `sherpa-onnx-1.13.5.aar` (CPU, what we vendor) and `-rknn.aar` (Rockchip). There
is **no `-qnn.aar`**. QNN requires building sherpa-onnx from source:

```
export QNN_SDK_ROOT=.../qairt/2.40.0.251030
export ANDROID_NDK=...
SHERPA_ONNX_ENABLE_QNN=ON SHERPA_ONNX_ENABLE_BINARY=ON ./build-android-arm64-v8a.sh
```

Produces `libonnxruntime.so` (~15 MB, larger than the CPU build) and `libsherpa-onnx-jni.so`
(~4.6 MB), which then get repacked into an AAR the same way I repacked the CPU one.

This has to be redone on every sherpa-onnx upgrade. That is the ongoing maintenance tax.

**SDK version is pinned and mirrored:** the export workflow uses QNN SDK **2.40.0.251030**,
mirrored at `huggingface.co/csukuangfj/qnn-toolkit/resolve/main/v2.40.0.251030.zip` — so
obtaining it does not require a Qualcomm account in practice, though see the next point.

### 3.2 Shipping Qualcomm's runtime blobs — the genuine blocker

From sherpa-onnx's own Kotlin API, verbatim:

```kotlin
// Please copy libQnnHtp.so and libQnnSystem.so to jniLibs/arm64-v8a by yourself
```

Plus per-Hexagon-version DSP skeleton libraries — that is what `prependAdspLibraryPath` exists
for. These come from the Qualcomm AI Engine Direct SDK.

**I could not find explicit redistribution permission.** Two pieces of circumstantial evidence,
both pointing the same way:

- Qualcomm's *own* `quic/ai-engine-direct-helper` repo does **not** redistribute
  `libQnnHtp.so` — zero hits in a code search. Users are told to fetch the SDK themselves.
- Its license is `NOASSERTION` on GitHub, i.e. a custom Qualcomm variant, not a standard OSS
  licence.

This does not mean redistribution is forbidden — it means **the EULA has to be read before
shipping**, and that is a genuine gate, not a formality.

**Where this bites hardest:** `app/build.gradle` has an `fDroid` product flavour. F-Droid
requires free, buildable-from-source packages. Proprietary Qualcomm blobs would very likely
disqualify that variant, so QNN would have to be a **flavour-specific feature** — present in
`standalone`/`playStore`/`dev`, absent in `fDroid`. That is achievable (the flavour machinery
already does this for billing) but it means the codebase carries both paths permanently.

For a personal fork this is a non-issue. For anything upstream it is the deciding constraint.

### 3.3 Fixed input shape — real, but manageable

These are pre-compiled Hexagon context binaries. **The input length is baked in.** A 30s binary
computes 30 seconds of work no matter what was said — which is precisely the mandatory-30-second
encoder cost that makes Whisper slow, reintroduced into the model chosen for avoiding it.

The obvious-but-wrong fix is shipping every duration bucket: at ~80 MB each and per-SoC, that is
a hosting and selection matrix nobody wants to maintain.

**Better design — pick one duration and fall back:**

- Ship the **15s** binary only (one per SoC).
- Utterances ≤ 15s → NPU. This is the overwhelming majority of dictation.
- Utterances > 15s → the existing CPU path, which already works well and is already there.

One model per SoC, no matrix, and the fallback is code that exists and is tested. The wasted
compute on a 3-second utterance is real but happens on the NPU where it is cheap, which is the
whole point.

### 3.4 Hosting — a problem we have already solved once

The QNN assets are `.tar.bz2` archives and **`DownloadActivity` cannot unpack archives** — the
exact issue that forced re-hosting the 110M CPU weights on our own release. Same fix: extract
`model.bin` + `tokens.txt`, upload as plain assets to the `js-commit/voice-input` release, pin
SHA-256 in `ParakeetModels.kt`. The digest-pinning machinery is already in place.

---

## 4. The thing most likely to kill this

Everything above is tractable engineering. This is the part that could make it pointless.

**NPUs win on sustained throughput. Dictation is duty-cycled.** Load the model, run one ~200 ms
inference, go idle for minutes. Benchmarks of NPU acceleration consistently show the advantage
collapsing when fixed overhead dominates a small workload — one published comparison found
Hexagon at 2–4× the CPU on large vision workloads narrowing to **~1.3× on a small classifier
where fixed preprocessing dominated the tiny workload**. And NPU init overhead is specifically
called out as problematic for "duty-cycled applications, where models must be frequently
loaded/unloaded".

Concretely, the concern is that a 113 MB context binary has to be loaded onto the DSP and the
HTP powered up. If that costs more energy than it saves on a 213 ms inference, the whole thing
is net-negative for battery — the exact opposite of the goal.

Mitigations exist (keep the recognizer's model resident between dictations, which the app
already does within a session) but this must be **measured, not assumed**.

---

## 5. Staged plan, with kill criteria

### Stage 0 — Is dictation even costing battery? (~1–2 hours)

Nothing gets built until this is answered. It may end the project.

1. Run `ParakeetBenchmarkTest` in a loop (N=500) via `am instrument` — decode only, no speech
   needed, so no manual dictation.
2. Bracket with `dumpsys batterystats --reset` / `dumpsys batterystats`, and read
   `adb shell dumpsys batterystats | grep -A5 "org.futo.voiceinput"` for the CPU/power figures.
3. Convert to mAh per dictation, then to "% of battery for 100 dictations/day".

**Kill criterion:** if realistic daily dictation is < ~1% of battery, stop here and write it up.
No NPU work can beat "already negligible", and every downstream cost — the licence question, the
flavour split, the maintenance tax — is then unjustifiable.

### Stage 1 — Does it work, is it faster, is it accurate? (~1 day)

Throwaway harness, **zero app changes**. Prove all three before touching the codebase.

1. Fetch QNN SDK 2.40.0.251030 (mirror URL in §3.1). **Read the EULA while it downloads** — this
   is the gate.
2. Build sherpa-onnx with `SHERPA_ONNX_ENABLE_QNN=ON SHERPA_ONNX_ENABLE_BINARY=ON` for
   arm64-v8a.
3. `adb push` to `/data/local/tmp` on the S23: the sherpa CLI binary, `libQnnHtp.so`,
   `libQnnSystem.so`, the SM8550 Hexagon skel lib, and the extracted
   `sherpa-onnx-qnn-SM8550-binary-parakeet-tdt_ctc-110m-15s/`.
4. Decode the bundled `test_wavs/` **and** `app/src/main/assets/benchmark_audio.floats.bin`
   (as wav) on both the QNN and CPU paths.
5. Record: **transcript diff**, decode ms, model load ms, and battery delta over N runs.

**Kill criteria — any one of these stops it:**
- Transcripts differ beyond trivial formatting → accuracy is being traded. Stop. *(§2)*
- Energy per inference is not meaningfully below CPU → the duty-cycle problem is real. Stop.
  *(§4)*
- The EULA forbids redistribution → personal-fork-only at best; decide then.

### Stage 2 — Productionise (~2–3 days, only if stage 1 passes)

| Step | Where |
|---|---|
| Repack QNN AAR (arm64 only, as with the CPU one) | `app/libs/` |
| Bundle `libQnnHtp.so`, `libQnnSystem.so`, skel libs — **non-fDroid flavours only** | `app/build.gradle` sourceSets |
| Re-host extracted `model.bin` + `tokens.txt`, pin SHA-256 | `js-commit/voice-input` release |
| Add `SherpaModelKind.QnnCtc`; per-SoC URL map keyed on `ro.soc.model` | `ParakeetModels.kt` |
| `isQnnSupported()`: arm64 **and** SoC in the supported set **and** flavour has the blobs | `ParakeetModels.kt` |
| QNN branch: `provider = "qnn"`, `QnnConfig(...)`, `prependAdspLibraryPath` | `ml/ParakeetModel.kt` |
| Route utterances > 15s to the CPU path | `AudioRecognizer.kt` |
| Engine option + honest tip about what it does | `settings/pages/Models.kt` |

The `SpeechModel` interface means `AudioRecognizer` needs almost nothing beyond the length
routing — this is the payoff from having built the CPU path behind an interface.

---

## 6. Alternative considered: LiteRT + Google Play for On-device AI

Google's **LiteRT Qualcomm AI Engine Direct Accelerator** (replacing the old TFLite QNN
delegate) solves §3.2 and §3.3 outright: Play Services delivers the NPU runtime libraries, and
"AI Packs" deliver the right per-SoC compiled model automatically. No bundled blobs, no SoC
matrix to host.

**Rejected anyway**, for two reasons:

1. It requires **LiteRT/TFLite model format**. Parakeet would have to be converted and
   re-validated — throwing away the working sherpa-onnx integration entirely.
2. It makes NPU acceleration **depend on Google Play Services**. For a privacy-focused,
   F-Droid-distributed keyboard this is a worse trade than the blob problem it solves.

Worth revisiting only if Qualcomm's licence turns out to actually forbid redistribution *and*
NPU acceleration proves valuable enough to justify a format migration.

---

## 7. Open questions

- **The Qualcomm EULA.** Unresolved and gating. Read it at the start of stage 1.
- **The Z Fold's SoC.** `adb` kept dropping; needs `getprop ro.soc.model`. Almost certainly
  SM8750 or SM8850, both covered.
- **Whether keeping the QNN context resident across dictations is viable**, and what it costs in
  RAM while idle. This is the main lever against §4.
- **CTC vs transducer branch.** Both are published. We run the CTC branch on CPU; no reason to
  change, but the transducer binaries are there if accuracy turns out to differ.

---

## 8. Honest summary

The user's instinct is sound: NPUs *are* more power-efficient than CPUs for neural inference,
and this is the right question to ask. The research says the path is real — the exact model we
ship has binaries for the exact chip in the minimum-target phone, the quantization is the
conservative kind, and the integration is small because the CPU path was built behind an
interface.

What the research cannot say is whether it is *worth* it, because two numbers do not exist
anywhere: what dictation currently costs in battery, and what a duty-cycled NPU inference costs
against a 213 ms CPU one. Stages 0 and 1 produce both for about a day and a half of work, and
either could reasonably end the project.

That is the right way to spend the next unit of effort — not on building it, and not on
speculating further.
