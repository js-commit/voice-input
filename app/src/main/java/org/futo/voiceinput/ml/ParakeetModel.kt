package org.futo.voiceinput.ml

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineNemoEncDecCtcModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import org.futo.voiceinput.ggml.DecodingMode
import org.futo.voiceinput.SherpaModelData
import org.futo.voiceinput.SherpaModelKind
import org.futo.voiceinput.parakeetModelPath
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
    private val numThreads: Int
) : SpeechModel {
    private var recognizer: OfflineRecognizer? = null

    init {
        val missing = model.files.filter { !File(context.filesDir, it.fileName).exists() }
        if (missing.isNotEmpty()) {
            // IOException so AudioRecognizer's existing catch kicks off the downloader, the same
            // way a missing ggml file does.
            throw IOException("Parakeet model files missing: ${missing.map { it.fileName }}")
        }

        val modelConfig = when (model.kind) {
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
