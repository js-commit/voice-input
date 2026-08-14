package org.futo.voiceinput.settings

import android.Manifest
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch
import org.futo.voiceinput.R
import org.futo.voiceinput.payments.BillingManager
import org.futo.voiceinput.theme.UixThemeAuto
import org.futo.voiceinput.updates.scheduleUpdateCheckingJob

/**
 * Optional string extra naming a NavHost route to open instead of the home screen, so the
 * recognizer popup can drop the user straight into the screen they wanted. See [openAppSettings].
 */
const val EXTRA_NAVIGATE_TO = "org.futo.voiceinput.NAVIGATE_TO"

/**
 * Opens the settings app, optionally at a specific screen. Usable from the input method service,
 * which has no activity of its own to launch from.
 */
fun Context.openAppSettings(route: String? = null) {
    try {
        val intent = Intent(this, SettingsActivity::class.java)
        route?.let { intent.putExtra(EXTRA_NAVIGATE_TO, it) }
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        startActivity(intent)
    } catch (e: Exception) {
        // A dead settings shortcut must never take down a dictation in progress.
        e.printStackTrace()
    }
}

class SettingsActivity : ComponentActivity() {
    internal lateinit var billing: BillingManager
    private fun updateContent() {
        setContent {
            UixThemeAuto {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    SetupOrMain(billing = billing)
                }
            }
        }
    }

    private val permission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {
            viewModel.onResume()
        }


    private val voiceIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
    }
    private val runVoiceIntent =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            viewModel.onIntentResult(
                when (it.resultCode) {
                    RESULT_OK -> {
                        val result =
                            it.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                        if (result.isNullOrEmpty()) {
                            getString(R.string.intent_result_is_null_or_empty)
                        } else {
                            result[0]
                        }
                    }

                    RESULT_CANCELED -> getString(R.string.intent_was_cancelled)
                    else -> getString(R.string.unknown_intent_result)
                }
            )
        }

    internal fun requestPermission() {
        permission.launch(Manifest.permission.RECORD_AUDIO)
    }

    internal fun launchVoiceIntent() {
        runVoiceIntent.launch(voiceIntent)
    }

    private lateinit var viewModel: SettingsViewModel
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        billing = BillingManager(this, lifecycleScope)

        viewModel = viewModels<SettingsViewModel>().value

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect {
                    updateContent()
                }
            }
        }

        scheduleUpdateCheckingJob(applicationContext)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)

        // This activity is singleTask, so a deep link arriving while it is already running lands
        // here rather than in onCreate. Without setIntent, getIntent() would keep returning the
        // launcher intent and EXTRA_NAVIGATE_TO would be silently dropped.
        setIntent(intent)
    }

    override fun onStart() {
        super.onStart()

        billing.startConnection {
            it.checkAlreadyOwnsProduct()
        }
    }

    override fun onResume() {
        super.onResume()

        billing.onResume()
        viewModel.onResume()
    }

    override fun onRestart() {
        super.onRestart()

        billing.onResume()
        viewModel.onResume()
    }
}
