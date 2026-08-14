import android.content.Context
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.futo.voiceinput.PARAKEET_MODELS
import org.futo.voiceinput.isQnnSupported
import org.futo.voiceinput.ml.ParakeetModelWrapper
import org.futo.voiceinput.qnnNeedsDownloading
import org.futo.voiceinput.qnnVariantsForDevice
import org.futo.voiceinput.ggml.DecodingMode
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.util.Log

/**
 * Exercises the Hexagon NPU path exactly as AudioRecognizer does, which is the bit the CLI
 * harness cannot cover: resolving libQnnHtp.so out of the APK's native lib dir by soname, and
 * getting the DSP skeleton onto ADSP_LIBRARY_PATH via prependAdspLibraryPath.
 *
 * Requires the QNN context binaries in filesDir. Run with -PtestRelease; see
 * docs/research/record-and-compare-benchmark-spec.md for why debug timings are worthless.
 */
class QnnParakeetTest {
    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun loadAudio(): FloatArray {
        val stream = DataInputStream(context.assets.open("benchmark_audio.floats.bin"))
        val bytes = stream.readBytes()
        stream.close()
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    @Test
    fun npuMatchesCpu() = runBlocking {
        assertTrue("device SoC has no QNN build", isQnnSupported())
        assertFalse("QNN models not staged in filesDir", context.qnnNeedsDownloading())

        val audio = loadAudio()
        val variant = qnnVariantsForDevice().last()
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)

        fun timeIt(tag: String, make: () -> ParakeetModelWrapper): Pair<String, Long> {
            val loadStart = System.currentTimeMillis()
            val m = make()
            val loadMs = System.currentTimeMillis() - loadStart
            return runBlocking {
                m.run(audio, "", null, DecodingMode.Greedy)      // warm up
                val t0 = System.currentTimeMillis()
                val text = m.run(audio, "", null, DecodingMode.Greedy)
                val decodeMs = System.currentTimeMillis() - t0
                m.close()
                Log.i(TAG, "RESULT | $tag | load ${loadMs}ms | decode ${decodeMs}ms")
                Log.i(TAG, "TEXT   | $tag | $text")
                text to decodeMs
            }
        }

        val (npuText, npuMs) = timeIt("NPU-${variant.maxSeconds}s") {
            ParakeetModelWrapper(context, PARAKEET_MODELS[0], threads, qnnVariant = variant)
        }
        val (cpuText, cpuMs) = timeIt("CPU") {
            ParakeetModelWrapper(context, PARAKEET_MODELS[0], threads, qnnVariant = null)
        }

        Log.i(TAG, "RESULT | speedup ${"%.2f".format(cpuMs.toDouble() / npuMs)}x")
        Log.i(TAG, "RESULT | match  ${npuText == cpuText}")

        assertTrue("NPU produced no text", npuText.isNotBlank())
    }

    companion object {
        private const val TAG = "QnnParakeet"
    }
}
