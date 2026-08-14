package org.futo.voiceinput

import android.content.Context
import android.os.Build
import java.io.File

/**
 * NVIDIA Parakeet models, run through sherpa-onnx / onnxruntime instead of whisper.cpp.
 *
 * These are FastConformer models rather than Whisper encoder-decoders, which matters in three
 * ways for dictation:
 *  - they consume audio at its actual length instead of padding to 30 seconds, so short
 *    utterances are cheap without needing an ACFT fine-tune to make audio_ctx truncation safe,
 *  - a CTC/transducer head emits tokens against audio evidence, so silence produces silence
 *    rather than the invented text Whisper's decoder can produce,
 *  - they have no prompt conditioning, so the personal dictionary does not apply to them.
 *
 * The native libraries are arm64-v8a only (see app/libs), so callers must check
 * [isParakeetSupported] before offering these models.
 */
enum class SherpaModelKind {
    /** A single model file holding the CTC head. */
    NemoCtc,

    /** encoder / decoder / joiner triple holding the TDT transducer head. */
    Transducer
}

data class SherpaFile(
    /**
     * Name the file gets under filesDir. Flat, because the downloader renames straight into
     * filesDir and does not create directories.
     */
    val fileName: String,

    val url: String,

    /** SHA-256 of the file. Empty means "size check only". */
    val sha256: String,

    val sizeBytes: Long
)

data class SherpaModelData(
    val name: String,
    val kind: SherpaModelKind,
    val files: List<SherpaFile>,

    /**
     * Rough peak resident memory while decoding, used to warn before selecting a model that
     * may not survive on a smaller device.
     */
    val approxRuntimeMb: Int
) {
    val totalBytes: Long
        get() = files.sumOf { it.sizeBytes }

    /** The single model file, for [SherpaModelKind.NemoCtc]. */
    fun modelFile(): SherpaFile = files.first { it.fileName.endsWith(".onnx") }

    fun fileEndingWith(suffix: String): SherpaFile =
        files.first { it.fileName.endsWith(suffix) }

    fun tokensFile(): SherpaFile = files.first { it.fileName.endsWith("tokens.txt") }
}

/**
 * Where the model files are fetched from. The sherpa-onnx project distributes the 110M model
 * only inside a .tar.bz2 release archive, which the in-app downloader cannot unpack, so those
 * two files are re-hosted as plain assets on this fork's release. The 0.6B model is published
 * on Hugging Face as individual files and is fetched from there directly.
 */
const val PARAKEET_MODEL_BASE_URL =
    "https://github.com/js-commit/voice-input/releases/download/asr-models-v1/"

private const val HF_PARAKEET_06B =
    "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/"

val PARAKEET_MODELS = listOf(
    SherpaModelData(
        name = "Parakeet 110M (recommended)",
        kind = SherpaModelKind.NemoCtc,
        approxRuntimeMb = 400,
        files = listOf(
            SherpaFile(
                fileName = "parakeet_tdt_ctc_110m_int8.onnx",
                url = PARAKEET_MODEL_BASE_URL + "parakeet_tdt_ctc_110m_int8.onnx",
                sha256 = "9177a9146cf32ee0cc8152276ef95116f312018d316be37ccf57f7efea81fc1a",
                sizeBytes = 131652171L
            ),
            SherpaFile(
                fileName = "parakeet_tdt_ctc_110m_tokens.txt",
                url = PARAKEET_MODEL_BASE_URL + "parakeet_tdt_ctc_110m_tokens.txt",
                sha256 = "450e56bd2f036fe5b6aa821865838cc5aa9d8b0106134ce9a9ba0664abe6cd10",
                sizeBytes = 9953L
            )
        )
    ),

    SherpaModelData(
        name = "Parakeet 600M (slightly more accurate, much slower)",
        kind = SherpaModelKind.Transducer,
        approxRuntimeMb = 1300,
        files = listOf(
            SherpaFile(
                fileName = "parakeet_tdt_0.6b_v2_encoder.int8.onnx",
                url = HF_PARAKEET_06B + "encoder.int8.onnx",
                sha256 = "a32b12d17bbbc309d0686fbbcc2987b5e9b8333a7da83fa6b089f0a2acd651ab",
                sizeBytes = 652184296L
            ),
            SherpaFile(
                fileName = "parakeet_tdt_0.6b_v2_decoder.int8.onnx",
                url = HF_PARAKEET_06B + "decoder.int8.onnx",
                sha256 = "b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e",
                sizeBytes = 7257753L
            ),
            SherpaFile(
                fileName = "parakeet_tdt_0.6b_v2_joiner.int8.onnx",
                url = HF_PARAKEET_06B + "joiner.int8.onnx",
                sha256 = "7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2",
                sizeBytes = 1739080L
            ),
            SherpaFile(
                fileName = "parakeet_tdt_0.6b_v2_tokens.txt",
                url = HF_PARAKEET_06B + "tokens.txt",
                sha256 = "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
                sizeBytes = 9384L
            )
        )
    )
)

/**
 * The sherpa-onnx AAR bundled in app/libs carries arm64-v8a binaries only, to keep the APK from
 * growing by an onnxruntime copy per ABI. On anything else the Parakeet engine is unavailable
 * and the app stays on whisper.cpp.
 */
fun isParakeetSupported(): Boolean =
    Build.SUPPORTED_64_BIT_ABIS.any { it == "arm64-v8a" }

fun Context.parakeetModelNeedsDownloading(model: SherpaModelData): Boolean =
    model.files.any { fileNeedsDownloading(it.fileName) }

fun Context.parakeetModelPath(file: SherpaFile): String =
    File(filesDir, file.fileName).absolutePath

/**
 * Launches the shared downloader for any missing files of [model]. Full URLs and digests are
 * passed alongside the names, because unlike the Whisper models these do not all live under the
 * FUTO model host.
 */
fun Context.startParakeetDownloadActivity(model: SherpaModelData) {
    val missing = model.files.filter { fileNeedsDownloading(it.fileName) }
    if (missing.isEmpty()) return

    startModelDownloadActivity(
        names = missing.map { it.fileName },
        urls = missing.map { it.url },
        digests = missing.map { it.sha256 }
    )
}
