package dev.ashik.sift.telephony

import android.app.role.RoleManager
import android.content.Context
import android.content.Intent
import android.database.ContentObserver
import android.os.Handler
import android.os.Looper
import android.provider.Telephony
import android.telephony.SmsMessage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/** The default-SMS-app role. Checked on every resume and before writes; never cached for long. */
class DefaultSmsRole(context: Context) {
    private val roleManager = context.getSystemService(RoleManager::class.java)
    private val _held = MutableStateFlow(isHeldNow())
    val held: StateFlow<Boolean> = _held

    fun isHeldNow(): Boolean = roleManager?.isRoleHeld(RoleManager.ROLE_SMS) == true

    fun refresh(): Boolean = isHeldNow().also { _held.value = it }

    fun requestIntent(): Intent = roleManager.createRequestRoleIntent(RoleManager.ROLE_SMS)
}

/** GSM-7 / UCS-2 segment math for the composer. */
object SegmentCounter {
    data class Count(val segments: Int, val remainingInSegment: Int, val unicode: Boolean)

    fun count(text: String): Count {
        val r = SmsMessage.calculateLength(text, false)
        return Count(segments = r[0], remainingInSegment = r[2], unicode = r[3] == SmsMessage.ENCODING_16BIT)
    }
}

/**
 * Watches `content://sms` while the role is held and runs [onChange] debounced (300 ms, latest wins).
 * Our own writes also trigger it; the incremental sync it runs is cheap and idempotent.
 */
class SmsProviderObserver(
    private val context: Context,
    scope: CoroutineScope,
    private val onChange: suspend () -> Unit,
) {
    private val trigger = MutableSharedFlow<Unit>(extraBufferCapacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    private var registered = false
    private val observer = object : ContentObserver(Handler(Looper.getMainLooper())) {
        override fun onChange(selfChange: Boolean) {
            trigger.tryEmit(Unit)
        }
    }

    init {
        scope.launch {
            trigger.collectLatest {
                delay(300)
                onChange()
            }
        }
    }

    @Synchronized
    fun register() {
        if (registered) return
        context.contentResolver.registerContentObserver(Telephony.Sms.CONTENT_URI, true, observer)
        registered = true
    }

    @Synchronized
    fun unregister() {
        if (!registered) return
        context.contentResolver.unregisterContentObserver(observer)
        registered = false
    }
}
