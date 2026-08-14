package org.futo.voiceinput.settings.pages

import android.content.Context
import android.os.Debug
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.rememberNavController
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.futo.voiceinput.ENGLISH_MODELS
import org.futo.voiceinput.ModelData
import org.futo.voiceinput.PARAKEET_MODELS
import org.futo.voiceinput.SherpaModelData
import org.futo.voiceinput.isParakeetSupported
import org.futo.voiceinput.modelNeedsDownloading
import org.futo.voiceinput.parakeetModelNeedsDownloading
import org.futo.voiceinput.ggml.DecodingMode
import org.futo.voiceinput.ml.ParakeetModelWrapper
import org.futo.voiceinput.ml.SpeechModel
import org.futo.voiceinput.ml.WhisperModelWrapper
import org.futo.voiceinput.settings.ScreenTitle
import org.futo.voiceinput.settings.ScrollableList
import org.futo.voiceinput.settings.Tip
import org.futo.voiceinput.theme.Typography
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val BENCHMARK_ASSET = "benchmark_audio.floats.bin"
private const val SAMPLE_RATE = 16000

data class BenchmarkRow(
    val label: String,
    val loadMs: Long,
    val decodeMs: Long,
    val audioSeconds: Float,
    val deltaMemMb: Long,
    val text: String
) {
    /** Fraction of realtime: decode wall-clock divided by clip length. Lower is faster. */
    val rtf: Float get() = decodeMs / 1000.0f / audioSeconds
}

/**
 * The clip that ships in assets: raw little-endian float32 PCM at 16 kHz, the same fixture the
 * instrumentation tests use.
 */
private fun loadBenchmarkAudio(context: Context): FloatArray {
    context.assets.open(BENCHMARK_ASSET).use { input ->
        val bytes = DataInputStream(input).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(buffer.remaining())
        buffer.get(out)
        return out
    }
}

private fun usedMemoryMb(): Long = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)

private suspend fun benchmarkOne(
    label: String,
    samples: FloatArray,
    open: () -> SpeechModel
): BenchmarkRow = withContext(Dispatchers.Default) {
    // Give the previous model's native allocations a chance to be released so the delta below
    // reflects this model rather than the last one.
    System.gc()
    val memBefore = usedMemoryMb()

    val loadStart = System.currentTimeMillis()
    val model = open()
    val loadMs = System.currentTimeMillis() - loadStart

    try {
        val decodeStart = System.currentTimeMillis()
        val text = model.run(samples, "", "en", DecodingMode.Greedy)
        val decodeMs = System.currentTimeMillis() - decodeStart

        BenchmarkRow(
            label = label,
            loadMs = loadMs,
            decodeMs = decodeMs,
            audioSeconds = samples.size.toFloat() / SAMPLE_RATE,
            deltaMemMb = usedMemoryMb() - memBefore,
            text = text
        )
    } finally {
        model.close()
    }
}

@Composable
fun BenchmarkRowItem(row: BenchmarkRow) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp, 4.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(8.dp)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(row.label, style = Typography.bodyLarge)
            Text(
                "decode %d ms  ·  RTF %.3f  ·  load %d ms  ·  ~%d MB".format(
                    row.decodeMs, row.rtf, row.loadMs, row.deltaMemMb
                ),
                style = Typography.bodySmall,
                color = MaterialTheme.colorScheme.outline
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(row.text, style = Typography.bodySmall)
        }
    }
}

/**
 * Times every downloaded English model against the same bundled clip, so Whisper and Parakeet
 * can be compared on this specific device rather than from published benchmarks. Models are run
 * one at a time and closed in between - loading the 600M Parakeet alongside anything else would
 * be a good way to get killed by the low-memory killer.
 */
@Composable
fun BenchmarkScreen(
    navController: NavHostController = rememberNavController()
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var running by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("") }
    var rows by remember { mutableStateOf(listOf<BenchmarkRow>()) }

    ScrollableList {
        ScreenTitle("Benchmark", showBack = true, navController = navController)

        Tip(
            "Runs a 5.9 second English clip through each model that is already downloaded. " +
                "RTF is decode time divided by clip length, so lower is faster."
        )

        Button(
            onClick = {
                if (running) return@Button
                running = true
                rows = listOf()

                scope.launch {
                    try {
                        val samples = withContext(Dispatchers.IO) { loadBenchmarkAudio(context) }
                        val results = mutableListOf<BenchmarkRow>()

                        ENGLISH_MODELS.forEach { m: ModelData ->
                            if (context.modelNeedsDownloading(m)) return@forEach
                            status = "Whisper: ${m.name}"
                            results += benchmarkOne("Whisper — ${m.name}", samples) {
                                WhisperModelWrapper(
                                    context, m, null,
                                    suppressNonSpeech = false,
                                    languages = setOf("en"),
                                    onStatusUpdate = {},
                                    onPartialDecode = {}
                                )
                            }
                            rows = results.toList()
                        }

                        if (isParakeetSupported()) {
                            PARAKEET_MODELS.forEach { m: SherpaModelData ->
                                if (context.parakeetModelNeedsDownloading(m)) return@forEach
                                status = "Parakeet: ${m.name}"
                                results += benchmarkOne("Parakeet — ${m.name}", samples) {
                                    ParakeetModelWrapper(
                                        context, m,
                                        numThreads = Runtime.getRuntime()
                                            .availableProcessors().coerceIn(2, 8)
                                    )
                                }
                                rows = results.toList()
                            }
                        }

                        status = if (results.isEmpty()) {
                            "No models are downloaded yet."
                        } else {
                            "Done."
                        }
                    } catch (e: Exception) {
                        status = "Failed: ${e.message}"
                    } finally {
                        running = false
                    }
                }
            },
            modifier = Modifier.padding(16.dp, 8.dp),
            enabled = !running
        ) {
            Text(if (running) "Running…" else "Run benchmark")
        }

        if (status.isNotEmpty()) {
            Text(
                status,
                style = Typography.bodySmall,
                modifier = Modifier.padding(16.dp, 0.dp),
                color = MaterialTheme.colorScheme.outline
            )
        }

        Spacer(modifier = Modifier.height(8.dp))
        rows.forEach { BenchmarkRowItem(it) }
    }
}
