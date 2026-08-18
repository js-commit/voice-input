# Draft email to FUTO Voice Input support — NOT SENT

**To:** (support address listed on https://voiceinput.futo.org/ — or file on https://github.com/futo-org/voice-input/issues)
**Subject:** Dictated text intermittently never reaches the input field (SwiftKey + FUTO Voice Input)

---

Hi,

I'm hitting an intermittent bug where dictated text is recognized but never
arrives in the text field.

**Setup**

- Device: `<DEVICE MODEL>`
- Android version: `<ANDROID VERSION / ROM>`
- FUTO Voice Input: `<VERSION, from Settings → about>`
- Keyboard: Microsoft SwiftKey, using FUTO Voice Input as the voice input
  method (SwiftKey's mic key, which goes through the
  `android.speech.action.RECOGNIZE_SPEECH` intent)
- Apps where I see it: `<e.g. WhatsApp, Signal, Chrome address bar…>`

**What happens**

I tap the mic, speak a sentence, and stop (or let VAD stop it). The FUTO
window shows the normal processing spinner, finishes without any error, and
closes. Nothing is inserted into the input box — the field is exactly as it
was, and the recognized text is gone for good. There is no error message and
no visible failure; it looks like a successful recognition that simply never
gets pasted.

**Expected vs. actual**

- Expected: when the window closes after a successful recognition, the text is
  inserted at the cursor.
- Actual: the window closes and no text is inserted, some of the time.

**Intermittency**

It is not every time. Rough pattern from my use:

- Short utterances succeed more often than long ones.
- It seems more likely on the second and later dictations in the same app
  without leaving the app in between.
- It is app-dependent — some apps are reliable, others fail often.

Sometimes tapping the input field before dictating makes it work, which is
what makes me think this is the same underlying problem as
https://github.com/futo-org/voice-input/issues/77 ("In some apps you have to
tap the inputfield that text is written"), and probably also #156 ("Inputs
sometimes fail if used more than once in the same app") and #169 ("Paste
recognized text into application fails most of the time"). All three are still
open and marked "Awaiting Info", so I'd like to help narrow it down rather
than just add another "me too".

**What I can provide**

Tell me which of these is useful and I'll capture it:

- `adb logcat` around a failing dictation — is there a specific tag or filter
  you want? I can grab `InputMethodManager`, `RemoteInputConnectionImpl`,
  `ActivityTaskManager` and your own log output, or just a full unfiltered
  capture over a reproduction.
- `adb shell dumpsys input_method` immediately before and after a failure.
- `adb shell dumpsys activity activities` at the moment the recognizer window
  is up, to show which task the recognizer activity landed in.
- A screen recording of a success and a failure back to back.
- A debug build with extra logging, if you have one — I'm happy to run it.

**One observation, in case it's useful**

I had a look at the source. `RecognizeActivity` is declared
`launchMode="singleInstance"` with `clearTaskOnLaunch="true"`, but every
`ACTION_RECOGNIZE_SPEECH` caller starts it with `startActivityForResult`.
`singleInstance` forces the activity into its own task, and Android does not
reliably deliver results across tasks — and even when the result does arrive,
the caller's task has to be re-resumed asynchronously after `finish()`. That
would make the keyboard's `onActivityResult` race the target editor regaining
focus, which fits both the intermittency and the "tap the field first and it
works" workaround in #77. It was `standard` until commit fc8fb12 (Jun 2023),
then `singleTask`, then `singleInstance` in bc53e16 (Oct 2023).

I may well be wrong about that — happy to test a build either way.

Thanks for the app, and for keeping it working with third-party keyboards.

`<NAME>`
