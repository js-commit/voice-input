package org.futo.voiceinput

import android.accessibilityservice.AccessibilityService
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.SystemClock
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

/**
 * Direct text insertion for keyboards whose own insert cannot be trusted. The
 * ACTION_RECOGNIZE_SPEECH contract hands the transcription to the calling keyboard and hopes; with
 * SwiftKey on Android 17 / One UI 9 that hope is dead: SwiftKey parks the returned text and only
 * commits it from its next onStartInputView, which One UI 9's IME-visibility rules never deliver
 * after a dictation (decompiled and logcat-verified 2026-08-14, see RecognizeActivity.sendResult).
 *
 * When the user enables this service in Accessibility settings, RecognizeActivity returns
 * RESULT_CANCELED to such keyboards - their cancel path stores nothing, so there is no parked
 * text to ghost-flush later (upstream #77) - and this service inserts the transcription into the
 * focused text field itself. The insertion goes through the accessibility API, which does not
 * care about InputConnection lifetimes, IME visibility, or window focus races.
 *
 * For keyboards that deliver correctly, RecognizeActivity keeps the normal RESULT_OK contract and
 * this service is never involved.
 */
class VoiceRepairAccessibilityService : AccessibilityService() {
    companion object {
        // The recognizer window is still closing when the clock starts; the editor's
        // accessibility tree needs a beat to settle before the focused field is findable.
        // There is no keyboard insert to wait out - the keyboard was told CANCELED.
        private const val FIRST_CHECK_DELAY_MS = 300L
        private const val RETRY_INTERVAL_MS = 300L

        // One retry only: a legitimately targetable field is found on the first or second look.
        // Anything later risks acting on a different screen than the one that was dictated into.
        private const val MAX_ATTEMPTS = 2

        /** How long after a dictation an insertion is still considered relevant. */
        private const val PENDING_EXPIRY_MS = 2500L

        /**
         * Window-state changes within this time of the dictation are the recognizer/keyboard
         * windows tearing down, not the user going somewhere else.
         */
        private const val NAVIGATION_GRACE_MS = 1200L

        private var instance: VoiceRepairAccessibilityService? = null

        /**
         * Takes ownership of delivering [text] into the focused field. Returns false when the
         * user has not enabled the service (or the text is blank), in which case the caller must
         * fall back to the normal result contract.
         */
        fun takeInsertion(text: String): Boolean {
            if (text.isBlank()) return false
            val service = instance ?: return false
            service.schedule(text)
            return true
        }
    }

    private lateinit var handler: Handler
    private var pendingText: String? = null
    private var pendingSince: Long = 0L
    private var attempts: Int = 0
    private var pendingPackage: CharSequence? = null

    override fun onServiceConnected() {
        super.onServiceConnected()
        handler = Handler(mainLooper)
        instance = this
    }

    override fun onDestroy() {
        if (instance === this) {
            instance = null
        }
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // The user navigated somewhere (new screen, new conversation, new app) while an insertion
        // was pending: abort it. The transcription belongs to the field that was being dictated
        // into, and any field found from here on is the wrong one - inserting into it is how a
        // transcription ends up pasted into a second chat thread. Early events are the dictation
        // UI itself tearing down and must not count as navigation.
        if (event?.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED &&
            pendingText != null &&
            SystemClock.uptimeMillis() - pendingSince > NAVIGATION_GRACE_MS
        ) {
            cancelPending()
        }
    }

    override fun onInterrupt() {}

    private fun schedule(text: String) {
        handler.removeCallbacksAndMessages(null)
        pendingText = text
        pendingSince = SystemClock.uptimeMillis()
        attempts = 0
        pendingPackage = null
        handler.postDelayed({ attemptInsertion() }, FIRST_CHECK_DELAY_MS)
    }

    private fun cancelPending() {
        pendingText = null
        pendingPackage = null
        handler.removeCallbacksAndMessages(null)
    }

    private fun attemptInsertion() {
        val text = pendingText ?: return
        if (SystemClock.uptimeMillis() - pendingSince > PENDING_EXPIRY_MS) {
            cancelPending()
            return
        }
        attempts += 1

        // If the foreground app changed between checks, the dictation target is gone.
        val currentPackage = rootInActiveWindow?.packageName
        if (pendingPackage == null) {
            pendingPackage = currentPackage
        } else if (currentPackage != null && currentPackage != pendingPackage) {
            cancelPending()
            return
        }

        val target = findTargetField()
        if (target == null) {
            // Field not found (e.g. the app cleared its focus and has several candidate fields) -
            // try again briefly in case focus is still being restored, then give up quietly. The
            // clipboard copy and transcription history still have the text.
            if (attempts < MAX_ATTEMPTS) {
                handler.postDelayed({ attemptInsertion() }, RETRY_INTERVAL_MS)
            } else {
                cancelPending()
            }
            return
        }

        val existing = existingText(target)
        if (normalized(existing).contains(normalized(text))) {
            // Already present (double-schedule, or the app restored it) - nothing to insert.
            cancelPending()
            return
        }

        val inserted = if (existing.isBlank()) {
            text
        } else {
            existing.trimEnd() + " " + text
        }

        val args = Bundle()
        args.putCharSequence(
            AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
            inserted
        )
        target.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
        cancelPending()
    }

    /**
     * The input-focused editable field in any on-screen window; otherwise, if the active window
     * contains exactly one editable field (the common chat-app layout), that field.
     */
    private fun findTargetField(): AccessibilityNodeInfo? {
        // Input focus can live outside the app's window: a field inside the keyboard itself
        // (e.g. SwiftKey's GIF search box) belongs to the IME window, which rootInActiveWindow
        // never covers. Windows are ordered topmost-first, so a focused IME field wins over the
        // app's compose box behind it - it is the field the user was actually dictating into.
        for (window in windows) {
            val focused = window.root?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: continue
            if (!focused.isEditable) continue
            // Never write into (or reason about) password fields.
            return if (focused.isPassword) null else focused
        }

        val root = rootInActiveWindow ?: return null

        // The window list is empty until flagRetrieveInteractiveWindows is in effect (a service
        // enabled before this flag shipped keeps its old config until toggled); the active
        // window can still answer for the focused field directly.
        val focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
        if (focused != null && focused.isEditable) {
            return if (focused.isPassword) null else focused
        }

        val editables = ArrayList<AccessibilityNodeInfo>()
        collectEditable(root, editables)
        return if (editables.size == 1) editables[0] else null
    }

    private fun collectEditable(node: AccessibilityNodeInfo, out: MutableList<AccessibilityNodeInfo>) {
        if (node.isEditable && node.isVisibleToUser && !node.isPassword) {
            out.add(node)
        }
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            collectEditable(child, out)
        }
    }

    private fun existingText(node: AccessibilityNodeInfo): String {
        // An empty field reports its hint as text; treat that as empty or the hint would be
        // prepended to the inserted transcription.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && node.isShowingHintText) {
            return ""
        }
        return node.text?.toString() ?: ""
    }

    private fun normalized(s: String): String {
        return s.lowercase().replace(Regex("\\s+"), " ").trim()
    }
}
