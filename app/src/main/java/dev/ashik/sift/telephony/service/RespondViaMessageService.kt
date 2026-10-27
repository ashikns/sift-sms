package dev.ashik.sift.telephony.service

import android.app.Service
import android.content.Intent
import android.os.IBinder

/**
 * "Respond via message": declared only because the SMS role requires it. Sift doesn't send messages, so requests are
 * ignored. The dialer's own quick replies to calls don't come through here: the system (Telecom) sends those itself.
 */
class RespondViaMessageService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        stopSelf(startId)
        return START_NOT_STICKY
    }
}
