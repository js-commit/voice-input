package org.futo.voiceinput.ml

import org.futo.voiceinput.ggml.DecodingMode

/**
 * A loaded recognizer, whichever engine backs it. [WhisperModelWrapper] runs whisper.cpp;
 * [ParakeetModelWrapper] runs sherpa-onnx. AudioRecognizer talks only to this.
 */
interface SpeechModel {
    /**
     * @param samples 16 kHz mono float PCM.
     * @param glossary the personal dictionary. Whisper feeds it in as an initial prompt;
     *   Parakeet has no prompt conditioning and ignores it.
     * @param forceLanguage a language to pin, or null to use the configured set.
     */
    suspend fun run(
        samples: FloatArray,
        glossary: String,
        forceLanguage: String?,
        decodingMode: DecodingMode
    ): String

    suspend fun close()
}
