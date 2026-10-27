package dev.ashik.sift.telephony.service

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.telephony.TelephonyManager
import android.util.Log
import dev.ashik.sift.SiftApp
import kotlinx.coroutines.launch

/**
 * "Respond via message" (quick replies to incoming calls). Required to qualify for the SMS role. The system's
 * own Telecom quick replies are sent and persisted by the system; this handles other apps that ask us to send.
 */
class RespondViaMessageService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == TelephonyManager.ACTION_RESPOND_VIA_MESSAGE) {
            val text = intent.getStringExtra(Intent.EXTRA_TEXT)?.trim()
            val address = intent.data?.schemeSpecificPart?.substringBefore('?')?.split(',', ';')?.firstOrNull()?.trim()
            if (!text.isNullOrEmpty() && !address.isNullOrEmpty()) {
                val graph = (application as SiftApp).graph
                graph.appScope.launch {
                    runCatching { graph.sender.send(address, text) }.onFailure { Log.e("SiftRespond", "send failed", it) }
                }
            }
        }
        stopSelf(startId)
        return START_NOT_STICKY
    }
}
