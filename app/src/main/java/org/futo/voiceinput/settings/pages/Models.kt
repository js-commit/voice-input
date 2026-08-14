package org.futo.voiceinput.settings.pages

import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.rememberNavController
import org.futo.voiceinput.ENGLISH_MODELS
import org.futo.voiceinput.MULTILINGUAL_MODELS
import org.futo.voiceinput.ModelData
import org.futo.voiceinput.PARAKEET_MODELS
import org.futo.voiceinput.R
import org.futo.voiceinput.isParakeetSupported
import org.futo.voiceinput.modelNeedsDownloading
import org.futo.voiceinput.parakeetModelNeedsDownloading
import org.futo.voiceinput.startParakeetDownloadActivity
import org.futo.voiceinput.settings.ENGLISH_ENGINE
import org.futo.voiceinput.settings.ENGLISH_ENGINE_PARAKEET
import org.futo.voiceinput.settings.ENGLISH_ENGINE_WHISPER
import org.futo.voiceinput.settings.NavigationItem
import org.futo.voiceinput.settings.NavigationItemStyle
import org.futo.voiceinput.settings.PARAKEET_MODEL_INDEX
import org.futo.voiceinput.migration.ConditionalModelUpdate
import org.futo.voiceinput.migration.NeedsMigration
import org.futo.voiceinput.settings.DISMISS_MIGRATION_TIP
import org.futo.voiceinput.settings.ENABLE_MULTILINGUAL
import org.futo.voiceinput.settings.ENGLISH_MODEL_INDEX
import org.futo.voiceinput.settings.LANGUAGE_TOGGLES
import org.futo.voiceinput.settings.MANUALLY_SELECT_LANGUAGE
import org.futo.voiceinput.settings.MODELS_MIGRATED
import org.futo.voiceinput.settings.MULTILINGUAL_MODEL_INDEX
import org.futo.voiceinput.settings.PERSONAL_DICTIONARY
import org.futo.voiceinput.settings.ScreenTitle
import org.futo.voiceinput.settings.ScrollableList
import org.futo.voiceinput.settings.SettingRadio
import org.futo.voiceinput.settings.SettingToggleDataStore
import org.futo.voiceinput.settings.SettingsViewModel
import org.futo.voiceinput.settings.Tip
import org.futo.voiceinput.settings.USE_LANGUAGE_SPECIFIC_MODELS
import org.futo.voiceinput.settings.getSettingBlocking
import org.futo.voiceinput.settings.useDataStore
import org.futo.voiceinput.startModelDownloadActivity

@Composable
fun modelsSubtitle(): String? {
    val (languages, _) = useDataStore(LANGUAGE_TOGGLES)
    val (useLanguageSpecificModels, _) = useDataStore(USE_LANGUAGE_SPECIFIC_MODELS)

    val (multilingual, _) = useDataStore(ENABLE_MULTILINGUAL)

    val (englishIdxActual, _) = useDataStore(ENGLISH_MODEL_INDEX)
    val (multilingualIdxActual, _) = useDataStore(MULTILINGUAL_MODEL_INDEX)

    // It doesn't matter what the multilingual model is set to if multilingual is disabled, the model
    // isn't used anyway. So suppress any text about its value by pretending it's default
    val multilingualIdx =
        if (multilingual) multilingualIdxActual else MULTILINGUAL_MODEL_INDEX.default

    val englishIdx = if((!multilingual) || (languages.contains("en") && useLanguageSpecificModels)) {
        englishIdxActual
    } else {
        ENGLISH_MODEL_INDEX.default
    }

    val totalDiff =
        (englishIdx - ENGLISH_MODEL_INDEX.default) + (multilingualIdx - MULTILINGUAL_MODEL_INDEX.default)
    val usePlural =
        ((englishIdx != ENGLISH_MODEL_INDEX.default) && (multilingualIdx != MULTILINGUAL_MODEL_INDEX.default))
    return if (totalDiff < 0) {
        if (usePlural) {
            stringResource(R.string.using_smaller_models_accuracy_may_be_worse)
        } else {
            stringResource(R.string.using_smaller_model_accuracy_may_be_worse)
        }
    } else if (totalDiff > 0) {
        if (usePlural) {
            stringResource(R.string.using_larger_models_speed_may_be_slower)
        } else {
            stringResource(R.string.using_larger_model_speed_may_be_slower)
        }
    } else if ((englishIdx != ENGLISH_MODEL_INDEX.default) || (multilingualIdx != MULTILINGUAL_MODEL_INDEX.default)) {
        if (usePlural) {
            stringResource(R.string.using_non_default_models)
        } else {
            stringResource(R.string.using_non_default_model)
        }
    } else {
        null
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PersonalDictionaryEditor(disabled: Boolean) {
    val context = LocalContext.current

    val personalDict = useDataStore(PERSONAL_DICTIONARY)
    val textFieldValue = remember { mutableStateOf(context.getSettingBlocking(
        PERSONAL_DICTIONARY.key, PERSONAL_DICTIONARY.default)) }

    LaunchedEffect(textFieldValue.value) {
        personalDict.setValue(textFieldValue.value)
    }
    
    ScreenTitle(title = stringResource(R.string.personal_dictionary))

    TextField(
        value = textFieldValue.value,
        onValueChange = {
            textFieldValue.value = it
        },
        placeholder = { Text(stringResource(R.string.personal_dictionary_placeholder)) },
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp, 4.dp),
        enabled = !disabled
    )

}

@Composable
@Preview
fun ModelsScreen(
    settingsViewModel: SettingsViewModel = viewModel(),
    navController: NavHostController = rememberNavController()
) {
    val (useMultilingual, _) = useDataStore(ENABLE_MULTILINGUAL)

    val englishModelIndex = useDataStore(ENGLISH_MODEL_INDEX)
    val multilingualModelIndex = useDataStore(MULTILINGUAL_MODEL_INDEX)

    val (languages, _) = useDataStore(LANGUAGE_TOGGLES)
    val (useLanguageSpecificModels, _) = useDataStore(USE_LANGUAGE_SPECIFIC_MODELS)

    val englishEngine = useDataStore(ENGLISH_ENGINE)
    val parakeetModelIndex = useDataStore(PARAKEET_MODEL_INDEX)

    val context = LocalContext.current
    val needsUpdate = NeedsMigration()

    val wasMigrated = useDataStore(setting = MODELS_MIGRATED)
    val dismissMigrationTip = useDataStore(setting = DISMISS_MIGRATION_TIP)

    val usingParakeet = isParakeetSupported() && englishEngine.value == ENGLISH_ENGINE_PARAKEET

    val launchDownloaderIfNecessary = {
        val whisperModels = mutableListOf<ModelData>()

        if (useMultilingual) {
            whisperModels.add(MULTILINGUAL_MODELS[multilingualModelIndex.value])
        }

        // Only fetch the Whisper English model if something will actually use it. Since the
        // English default is now the largest Whisper model, pulling it unconditionally would
        // mean a 264 MB download just for opening this screen with Parakeet selected.
        val whisperStillBacksEnglish = !usingParakeet ||
            (useMultilingual && useLanguageSpecificModels && languages.contains("en"))
        if (whisperStillBacksEnglish) {
            whisperModels.add(ENGLISH_MODELS[englishModelIndex.value])
        }

        // One downloader at a time - the activity handles a single batch, and launching both
        // would stack two of them. Whichever is left over is picked up on the next visit.
        val whisperNeeded = whisperModels.filter { context.modelNeedsDownloading(it) }
        if (whisperNeeded.isNotEmpty()) {
            context.startModelDownloadActivity(whisperNeeded)
        } else if (usingParakeet) {
            context.startParakeetDownloadActivity(
                PARAKEET_MODELS[parakeetModelIndex.value.coerceIn(PARAKEET_MODELS.indices)]
            )
        }
    }

    LaunchedEffect(
        listOf(
            useMultilingual,
            englishModelIndex.value,
            multilingualModelIndex.value,
            englishEngine.value,
            parakeetModelIndex.value
        )
    ) {
        launchDownloaderIfNecessary()
    }


    ScrollableList {
        ScreenTitle(stringResource(R.string.model_options), showBack = true, navController = navController)

        ConditionalModelUpdate()

        if(wasMigrated.value && !dismissMigrationTip.value) {
            Tip(stringResource(R.string.new_model_features_tip), onDismiss = { dismissMigrationTip.setValue(true) })
        }

        if(languages.size > 1) {
            SettingToggleDataStore(
                stringResource(R.string.manually_select_language),
                MANUALLY_SELECT_LANGUAGE,
                subtitle = stringResource(R.string.manual_language_selection_toggle_subtitle)
            )
        }

        if(!needsUpdate) {
            PersonalDictionaryEditor(disabled = false)

            Spacer(modifier = Modifier.height(32.dp))
        }

        if (useMultilingual) {
            SettingRadio(
                stringResource(R.string.multilingual_model),
                MULTILINGUAL_MODELS.indices.toList(),
                MULTILINGUAL_MODELS.map { it.name },
                MULTILINGUAL_MODEL_INDEX
            )
        }

        if((!useMultilingual) || (languages.contains("en") && useLanguageSpecificModels)) {
            if (isParakeetSupported()) {
                SettingRadio(
                    "English engine",
                    listOf(ENGLISH_ENGINE_WHISPER, ENGLISH_ENGINE_PARAKEET),
                    listOf("Whisper (whisper.cpp)", "Parakeet (sherpa-onnx)"),
                    ENGLISH_ENGINE
                )
            }

            if (usingParakeet) {
                SettingRadio(
                    "Parakeet model",
                    PARAKEET_MODELS.indices.toList(),
                    PARAKEET_MODELS.map { it.name },
                    PARAKEET_MODEL_INDEX
                )

                val selected = PARAKEET_MODELS[
                    parakeetModelIndex.value.coerceIn(PARAKEET_MODELS.indices)
                ]
                if (context.parakeetModelNeedsDownloading(selected)) {
                    Tip(
                        "%.0f MB still to download for \"%s\". It will be fetched the next time you dictate."
                            .format(selected.totalBytes / 1_000_000.0, selected.name)
                    )
                }
                if (parakeetModelIndex.value == 1) {
                    Tip(
                        "600M is only about 1.5% more accurate than 110M, but on a Galaxy S23 it " +
                            "decodes 10x slower and takes 7 seconds just to load. Measured on a " +
                            "5.9s clip: 110M 213 ms, 600M 2211 ms. Prefer 110M unless you are on " +
                            "a fast phone and want the last bit of accuracy."
                    )
                }
                Tip(
                    "Parakeet is English-only and ignores the personal dictionary - these models " +
                        "have no prompt conditioning. Multilingual dictation always uses Whisper."
                )
            } else {
                SettingRadio(
                    stringResource(R.string.english_model),
                    ENGLISH_MODELS.indices.toList(),
                    ENGLISH_MODELS.map { it.name },
                    ENGLISH_MODEL_INDEX
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        NavigationItem(
            title = "Benchmark models",
            subtitle = "Time Whisper against Parakeet on this device",
            style = NavigationItemStyle.MiscNoArrow,
            navigate = { navController.navigate("benchmark") }
        )

        Tip(stringResource(R.string.parameter_count_tip))
    }
}
