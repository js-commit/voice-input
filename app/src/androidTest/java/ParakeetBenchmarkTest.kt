import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.futo.voiceinput.ENGLISH_MODELS
import org.futo.voiceinput.PARAKEET_MODELS
import org.futo.voiceinput.ggml.DecodingMode
import org.futo.voiceinput.isParakeetSupported
import org.futo.voiceinput.ml.ParakeetModelWrapper
import org.futo.voiceinput.ml.SpeechModel
import org.futo.voiceinput.ml.WhisperModelWrapper
import org.futo.voiceinput.modelNeedsDownloading
import org.futo.voiceinput.parakeetModelNeedsDownloading
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "ParakeetBench"

/**
 * Times whichever English models are present on the device against the shared 5.9 s fixture.
 * Skips any model that has not been downloaded, so it is safe to run on a fresh install.
 */
@RunWith(AndroidJUnit4::class)
class ParakeetBenchmarkTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun audio(): FloatArray {
        InstrumentationRegistry.getInstrumentation().context.assets.open("audio.floats.bin")
            .use { input ->
                val bytes = DataInputStream(input).readBytes()
                val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                val out = FloatArray(fb.remaining())
                fb.get(out)
                return out
            }
    }

    private fun bench(label: String, samples: FloatArray, open: () -> SpeechModel): String =
        runBlocking {
            val loadStart = System.currentTimeMillis()
            val model = open()
            val loadMs = System.currentTimeMillis() - loadStart
            try {
                // Warm the caches so the timing below is steady-state, not first-touch.
                model.run(samples, "", "en", DecodingMode.Greedy)

                val start = System.currentTimeMillis()
                val text = model.run(samples, "", "en", DecodingMode.Greedy)
                val decodeMs = System.currentTimeMillis() - start
                val rtf = decodeMs / 1000.0f / (samples.size / 16000.0f)

                Log.i(TAG, "RESULT | $label | load ${loadMs}ms | decode ${decodeMs}ms | RTF %.4f".format(rtf))
                Log.i(TAG, "TEXT   | $label | $text")
                text
            } finally {
                model.close()
            }
        }

    @Test
    fun benchmarkEnglishModels() {
        val samples = audio()
        Log.i(TAG, "clip ${samples.size} samples = %.2f s".format(samples.size / 16000.0f))

        var ran = 0

        ENGLISH_MODELS.forEach { m ->
            if (context.modelNeedsDownloading(m)) {
                Log.i(TAG, "SKIP   | Whisper ${m.name} (not downloaded)")
                return@forEach
            }
            bench("Whisper ${m.name}", samples) {
                WhisperModelWrapper(
                    context, m, null,
                    suppressNonSpeech = false,
                    languages = setOf("en"),
                    onStatusUpdate = {},
                    onPartialDecode = {}
                )
            }
            ran++
        }

        if (isParakeetSupported()) {
            PARAKEET_MODELS.forEach { m ->
                if (context.parakeetModelNeedsDownloading(m)) {
                    Log.i(TAG, "SKIP   | Parakeet ${m.name} (not downloaded)")
                    return@forEach
                }
                val text = bench("Parakeet ${m.name}", samples) {
                    ParakeetModelWrapper(
                        context, m,
                        numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
                    )
                }
                assertTrue("Parakeet produced no text for ${m.name}", text.isNotBlank())
                ran++
            }
        } else {
            Log.i(TAG, "SKIP   | Parakeet unsupported on this ABI")
        }

        assertTrue("No models were available to benchmark", ran > 0)
    }
}
