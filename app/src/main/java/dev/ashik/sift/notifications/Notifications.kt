package dev.ashik.sift.notifications

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ShortcutInfo
import android.net.Uri
import android.telephony.PhoneNumberUtils
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.Person
import androidx.core.app.RemoteInput
import androidx.core.content.LocusIdCompat
import androidx.core.content.pm.ShortcutInfoCompat
import androidx.core.content.pm.ShortcutManagerCompat
import androidx.core.graphics.drawable.IconCompat
import dev.ashik.sift.R
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.telephony.receiver.NotificationActionReceiver
import dev.ashik.sift.ui.MainActivity
import kotlinx.coroutines.flow.first
import java.util.Locale

/** The thread the user is looking at right now (set by the thread screen while resumed). */
class VisibleThreadTracker {
    @Volatile
    var threadId: Long? = null

    fun isVisible(threadId: Long) = this.threadId == threadId
}

object Channels {
    const val MESSAGES = "messages"
    const val SPAM = "spam"
    const val SEND_FAILURES = "send_failures"
    const val MMS = "mms_unsupported"
}

/** Long-lived conversation shortcuts so message notifications appear in the system's Conversations section. */
class ConversationShortcuts(private val context: Context) {
    fun shortcutId(senderKey: String) = "s_$senderKey"

    fun ensure(threadId: Long, senderKey: String, person: Person, name: String): String {
        val id = shortcutId(senderKey)
        val info = ShortcutInfoCompat.Builder(context, id)
            .setShortLabel(name)
            .setLongLabel(name)
            .setLongLived(true)
            .setPerson(person)
            .setLocusId(LocusIdCompat(id))
            .setIcon(IconCompat.createWithResource(context, R.drawable.ic_shortcut_person))
            .setCategories(setOf(ShortcutInfo.SHORTCUT_CATEGORY_CONVERSATION))
            .setIntent(MainActivity.threadIntent(context, threadId))
            .build()
        runCatching { ShortcutManagerCompat.pushDynamicShortcut(context, info) }
        return id
    }
}

class MessageNotifier(
    private val context: Context,
    private val db: AppDatabase,
    private val shortcuts: ConversationShortcuts,
) {
    private val manager = NotificationManagerCompat.from(context)

    fun ensureChannels() {
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannels(
            listOf(
                NotificationChannel(Channels.MESSAGES, context.getString(R.string.channel_messages), NotificationManager.IMPORTANCE_HIGH)
                    .apply { description = context.getString(R.string.channel_messages_desc) },
                NotificationChannel(Channels.SPAM, context.getString(R.string.channel_spam), NotificationManager.IMPORTANCE_LOW)
                    .apply {
                        description = context.getString(R.string.channel_spam_desc)
                        setSound(null, null)
                        enableVibration(false)
                    },
                NotificationChannel(Channels.SEND_FAILURES, context.getString(R.string.channel_failures), NotificationManager.IMPORTANCE_DEFAULT),
                NotificationChannel(Channels.MMS, context.getString(R.string.channel_mms), NotificationManager.IMPORTANCE_DEFAULT)
                    .apply { description = context.getString(R.string.channel_mms_desc) },
            ),
        )
    }

    /** Posts (or updates) the thread's notification with its unread, non-spam messages. */
    suspend fun notifyThread(threadId: Long, alert: Boolean) {
        val row = db.conversations().observe(threadId).first() ?: return cancelThread(threadId)
        val conv = row.conversation
        val unread = db.messages().unreadNotSpam(threadId, 6).reversed()
        if (unread.isEmpty()) return cancelThread(threadId)
        val name = row.displayName ?: displayAddress(conv.address)
        val sender = Person.Builder().setName(name).setKey(conv.senderKey).build()
        val me = Person.Builder().setName(context.getString(R.string.notification_me)).build()
        val style = NotificationCompat.MessagingStyle(me)
        unread.forEach { style.addMessage(it.body, it.date, sender) }
        val shortcutId = shortcuts.ensure(threadId, conv.senderKey, sender, name)
        val public = NotificationCompat.Builder(context, Channels.MESSAGES)
            .setSmallIcon(R.drawable.ic_stat_message)
            .setContentTitle(context.getString(R.string.app_name))
            .setContentText(context.resources.getQuantityString(R.plurals.notification_new_messages, unread.size, unread.size))
            .build()
        val builder = NotificationCompat.Builder(context, Channels.MESSAGES)
            .setSmallIcon(R.drawable.ic_stat_message)
            .setStyle(style)
            .setShortcutId(shortcutId)
            .setLocusId(LocusIdCompat(shortcutId))
            .setCategory(NotificationCompat.CATEGORY_MESSAGE)
            .setContentIntent(openThread(threadId))
            .setAutoCancel(true)
            .setOnlyAlertOnce(!alert)
            .setWhen(unread.last().date)
            .setShowWhen(true)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
            .setPublicVersion(public)
        if (conv.canReply) builder.addAction(replyAction(threadId))
        builder.addAction(action(threadId, NotificationActionReceiver.ACTION_MARK_READ, R.string.action_mark_read))
        builder.addAction(action(threadId, NotificationActionReceiver.ACTION_MARK_SPAM, R.string.action_mark_spam))
        post(TAG_THREAD, threadId, builder)
    }

    /** Silent notice for a message filtered as spam (only when the user opted in). */
    suspend fun notifySpam(threadId: Long) {
        val row = db.conversations().observe(threadId).first() ?: return
        val name = row.displayName ?: displayAddress(row.conversation.address)
        val builder = NotificationCompat.Builder(context, Channels.SPAM)
            .setSmallIcon(R.drawable.ic_stat_message)
            .setContentTitle(context.getString(R.string.notification_spam_title))
            .setContentText(context.getString(R.string.notification_spam_text, name))
            .setContentIntent(openThread(threadId))
            .setAutoCancel(true)
            .setSilent(true)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
            .addAction(action(threadId, NotificationActionReceiver.ACTION_NOT_SPAM, R.string.action_not_spam))
        post(TAG_SPAM, threadId, builder)
    }

    fun notifySendFailed(threadId: Long, address: String) {
        val builder = NotificationCompat.Builder(context, Channels.SEND_FAILURES)
            .setSmallIcon(R.drawable.ic_stat_message)
            .setContentTitle(context.getString(R.string.notification_send_failed_title))
            .setContentText(context.getString(R.string.notification_send_failed_text, displayAddress(address)))
            .setContentIntent(openThread(threadId))
            .setAutoCancel(true)
        post(TAG_FAILED, threadId, builder)
    }

    fun notifyMmsUnsupported() {
        val builder = NotificationCompat.Builder(context, Channels.MMS)
            .setSmallIcon(R.drawable.ic_stat_message)
            .setContentTitle(context.getString(R.string.notification_mms_title))
            .setContentText(context.getString(R.string.notification_mms_text))
            .setStyle(NotificationCompat.BigTextStyle().bigText(context.getString(R.string.notification_mms_text)))
            .setAutoCancel(true)
        post(TAG_MMS, 0, builder)
    }

    fun cancelThread(threadId: Long) {
        manager.cancel(TAG_THREAD, threadId.toInt())
        manager.cancel(TAG_SPAM, threadId.toInt())
    }

    private fun post(tag: String, id: Long, builder: NotificationCompat.Builder) {
        if (!manager.areNotificationsEnabled()) return
        try {
            manager.notify(tag, id.toInt(), builder.build())
        } catch (_: SecurityException) {
            // POST_NOTIFICATIONS revoked: nothing to do.
        }
    }

    private fun openThread(threadId: Long): PendingIntent = PendingIntent.getActivity(
        context, threadId.toInt(), MainActivity.threadIntent(context, threadId).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    private fun actionIntent(threadId: Long, action: String) =
        Intent(action, Uri.parse("sift://notification/$threadId/$action"), context, NotificationActionReceiver::class.java)
            .putExtra(NotificationActionReceiver.EXTRA_THREAD_ID, threadId)

    private fun action(threadId: Long, action: String, label: Int): NotificationCompat.Action {
        val pi = PendingIntent.getBroadcast(
            context, threadId.toInt(), actionIntent(threadId, action), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Action.Builder(0, context.getString(label), pi).build()
    }

    private fun replyAction(threadId: Long): NotificationCompat.Action {
        val input = RemoteInput.Builder(NotificationActionReceiver.KEY_REPLY)
            .setLabel(context.getString(R.string.notification_reply_hint))
            .build()
        // Mutable so the system can attach the typed reply; explicit, so that's allowed on Android 14+.
        val pi = PendingIntent.getBroadcast(
            context, threadId.toInt(), actionIntent(threadId, NotificationActionReceiver.ACTION_REPLY),
            PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Action.Builder(R.drawable.ic_stat_message, context.getString(R.string.action_reply), pi)
            .addRemoteInput(input)
            .setAllowGeneratedReplies(true)
            .setSemanticAction(NotificationCompat.Action.SEMANTIC_ACTION_REPLY)
            .setShowsUserInterface(false)
            .build()
    }

    fun displayAddress(address: String): String =
        PhoneNumberUtils.formatNumber(address, Locale.getDefault().country.ifEmpty { "US" }) ?: address

    private companion object {
        const val TAG_THREAD = "thread"
        const val TAG_SPAM = "spam"
        const val TAG_FAILED = "failed"
        const val TAG_MMS = "mms"
    }
}
