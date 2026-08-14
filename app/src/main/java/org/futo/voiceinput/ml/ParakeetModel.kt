package org.futo.voiceinput.ml

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import com.k2fsa.sherpa.onnx.QnnConfig
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import org.futo.voiceinput.ggml.DecodingMode
import org.futo.voiceinput.QnnVariant
import org.futo.voiceinput.SherpaModelData
import org.futo.voiceinput.SherpaModelKind
import org.futo.voiceinput.parakeetModelPath
import org.futo.voiceinput.qnnTokensFile
import java.io.File
import java.io.IOException

/**
 * Serialises access to the native recognizer, mirroring what WhisperModel.kt does for ggml.
 * onnxruntime itself is thread-safe, but the recognizer + stream pair is not, and this also
 * keeps a decode from racing a close().
 */
@OptIn(DelicateCoroutinesApi::class)
private val parakeetInferenceContext = newSingleThreadContext("parakeet-inference")

/**
 * Runs an NVIDIA Parakeet model through sherpa-onnx.
 *
 * Unlike the Whisper path there is no partial-decode callback: sherpa's offline recognizer is a
 * single blocking call with no token stream to hook. That is tolerable here because the models
 * are fast enough that the final result usually lands sooner than partials would have rendered;
 * see docs/research/asr-alternatives-2026-08.md.
 */
class ParakeetModelWrapper(
    context: Context,
    private val model: SherpaModelData,
    private val numThreads: Int,
    /**
     * When set, run on the Hexagon NPU using this pre-compiled context binary instead of on the
     * CPU. The caller is responsible for only selecting a variant whose [QnnVariant.maxSeconds]
     * covers the utterance - the runtime truncates silently past that.
     */
    private val qnnVariant: QnnVariant? = null
) : SpeechModel {
    private var recognizer: OfflineRecognizer? = null

    init {
        val required = if (qnnVariant != null) {
            qnnVariant.files + qnnTokensFile()
        } else {
            model.files
        }
        val missing = required.filter { !File(context.filesDir, it.fileName).exists() }
        if (missing.isNotEmpty()) {
            // IOException so AudioRecognizer's existing catch kicks off the downloader, the same
            // way a missing ggml file does.
            throw IOException("Parakeet model files missing: ${missing.map { it.fileName }}")
        }

        if (qnnVariant != null) {
            // The Hexagon DSP loads its skeleton library (libQnnHtpV73Skel.so / V81) itself, and
            // only looks along ADSP_LIBRARY_PATH - it cannot see the app's nativeLibraryDir
            // unless we put it there first. Without this the backend falls back or fails to
            // initialise.
            OfflineRecognizer.prependAdspLibraryPath(context.applicationInfo.nativeLibraryDir)
        }

        val modelConfig = if (qnnVariant != null) {
            // The transducer branch, not the CTC one: sherpa-onnx 1.13.5 exposes qnnConfig on
            // OfflineTransducerModelConfig but not on OfflineNemoEncDecCtcModelConfig, so the CTC
            // context binaries are unreachable from Kotlin even though the C++ supports them.
            // Both branches are published for this model and produce the same transcripts.
            OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    // Ignored when a context binary is supplied, but must be present.
                    encoder = "", decoder = "", joiner = "",
                    qnnConfig = QnnConfig(
                        // Resolved by the dynamic linker out of the APK's native lib dir.
                        backendLib = "libQnnHtp.so",
                        contextBinary = qnnVariant.contextBinaryArg { context.parakeetModelPath(it) },
                        systemLib = "libQnnSystem.so"
                    )
                ),
                tokens = context.parakeetModelPath(qnnTokensFile()),
                numThreads = numThreads,
                debug = false,
                provider = "qnn",
                modelType = "nemo_transducer"
            )
        } else when (model.kind) {
            SherpaModelKind.NemoCtc -> OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = context.parakeetModelPath(model.modelFile())
                ),
                tokens = context.parakeetModelPath(model.tokensFile()),
                numThreads = numThreads,
                debug = false,
                provider = "cpu",
                modelType = "nemo_ctc"
            )

            SherpaModelKind.Transducer -> OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = context.parakeetModelPath(model.fileEndingWith("encoder.int8.onnx")),
                    decoder = context.parakeetModelPath(model.fileEndingWith("decoder.int8.onnx")),
                    joiner = context.parakeetModelPath(model.fileEndingWith("joiner.int8.onnx"))
                ),
                tokens = context.parakeetModelPath(model.tokensFile()),
                numThreads = numThreads,
                debug = false,
                provider = "cpu",
                // Not "transducer": that path expects an icefall/Zipformer export and dies with
                // "'vocab_size' does not exist in the metadata" on a NeMo checkpoint.
                modelType = "nemo_transducer"
            )
        }

        recognizer = OfflineRecognizer(
            assetManager = null,
            config = OfflineRecognizerConfig(
                featConfig = FeatureConfig(sampleRate = 16000, featureDim = 80),
                modelConfig = modelConfig,
                decodingMethod = "greedy_search"
            )
        )
    }

    /** [glossary], [forceLanguage] and [decodingMode] do not apply: these models are
     * English-only, greedy, and have no prompt conditioning. */
    override suspend fun run(
        samples: FloatArray,
        glossary: String,
        forceLanguage: String?,
        decodingMode: DecodingMode
    ): String = withContext(parakeetInferenceContext) {
        val r = recognizer ?: throw IllegalStateException("Parakeet model already closed")

        yield()

        val stream = r.createStream()
        try {
            stream.acceptWaveform(samples, 16000)
            r.decode(stream)
            r.getResult(stream).text.trim()
        } finally {
            stream.release()
        }
    }

    override suspend fun close() = withContext(parakeetInferenceContext) {
        recognizer?.release()
        recognizer = null
    }

    companion object {
        private const val TAG = "ParakeetModel"
    }
}
