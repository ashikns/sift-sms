package dev.ashik.sift.notifications

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ShortcutInfo
import android.telephony.PhoneNumberUtils
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.Person
import androidx.core.content.LocusIdCompat
import androidx.core.content.pm.ShortcutInfoCompat
import androidx.core.content.pm.ShortcutManagerCompat
import androidx.core.graphics.drawable.IconCompat
import androidx.work.WorkManager
import dev.ashik.sift.R
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.spam.FolderMoves
import dev.ashik.sift.ui.MainActivity
import kotlinx.coroutines.flow.first
import java.util.Locale
import java.util.UUID

/** A thread as one tab shows it: only its messages in [folder]. */
data class ThreadView(val threadId: Long, val folder: Int)

/** The thread view the user is looking at right now (set by the thread screen while resumed). */
class VisibleThreadTracker {
    @Volatile
    var view: ThreadView? = null

    fun isVisible(threadId: Long, folder: Int) = view == ThreadView(threadId, folder)
}

object Channels {
    const val MESSAGES = "messages"
    const val SPAM = "spam"
    const val MMS = "mms_unsupported"
    const val SCAN = "scan"
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
            .setIntent(MainActivity.threadIntent(context, threadId, Folder.INBOX))
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
                NotificationChannel(Channels.MMS, context.getString(R.string.channel_mms), NotificationManager.IMPORTANCE_DEFAULT)
                    .apply { description = context.getString(R.string.channel_mms_desc) },
                NotificationChannel(Channels.SCAN, context.getString(R.string.channel_scan), NotificationManager.IMPORTANCE_LOW)
                    .apply { description = context.getString(R.string.channel_scan_desc) },
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
            .setContentIntent(openThread(threadId, Folder.INBOX))
            .setAutoCancel(true)
            .setOnlyAlertOnce(!alert)
            .setWhen(unread.last().date)
            .setShowWhen(true)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
            .setPublicVersion(public)
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
            .setContentIntent(openThread(threadId, Folder.SPAM))
            .setAutoCancel(true)
            .setSilent(true)
            .setVisibility(NotificationCompat.VISIBILITY_PRIVATE)
        post(TAG_SPAM, threadId, builder)
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

    /** Cancels the notification about the thread's messages in [folder]: its messages one, or its spam notice. */
    fun cancel(threadId: Long, folder: Int) = manager.cancel(if (folder == Folder.SPAM) TAG_SPAM else TAG_THREAD, threadId.toInt())

    /**
     * Id of a rescan's or re-check's progress notice: untagged, like a foreground-service notification, and one per work
     * request so a replaced one can't clear its successor's.
     */
    fun scanNotificationId(workId: UUID): Int = workId.hashCode().takeIf { it != 0 } ?: 1

    /** The ongoing progress notice titled [title], also the work's foreground-service notification ([total] 0 while counting). */
    fun scanProgress(workId: UUID, title: Int, done: Int, total: Int): Notification = scanProgressBuilder(workId, title, done, total).build()

    fun updateScanProgress(workId: UUID, title: Int, done: Int, total: Int) =
        post(null, scanNotificationId(workId).toLong(), scanProgressBuilder(workId, title, done, total))

    /** Removes the notice when the work ran as a plain job; a foreground service's notice leaves with the service. */
    fun cancelScanProgress(workId: UUID) = manager.cancel(scanNotificationId(workId))

    fun cancelScanDone() = manager.cancel(TAG_SCAN_DONE, 0)

    /** The rescan's summary; [moves] is null when the rescan was interrupted and resumed, so it can't tell. */
    fun notifyScanDone(checked: Int, moves: FolderMoves?) {
        val r = context.resources
        val parts = mutableListOf(r.getQuantityString(R.plurals.scan_done_checked, checked, checked))
        when {
            moves == null -> Unit
            moves.toSpam == 0 && moves.toInbox == 0 -> parts += context.getString(R.string.scan_done_unchanged)
            else -> {
                if (moves.toSpam > 0) parts += r.getQuantityString(R.plurals.scan_done_to_spam, moves.toSpam, moves.toSpam)
                if (moves.toInbox > 0) parts += r.getQuantityString(R.plurals.scan_done_to_inbox, moves.toInbox, moves.toInbox)
            }
        }
        val text = parts.joinToString(" ")
        val builder = NotificationCompat.Builder(context, Channels.SCAN)
            .setSmallIcon(R.drawable.ic_stat_message)
            .setContentTitle(context.getString(R.string.scan_done_title))
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(openApp())
            .setAutoCancel(true)
            .setSilent(true)
        post(TAG_SCAN_DONE, 0, builder)
    }

    private fun scanProgressBuilder(workId: UUID, title: Int, done: Int, total: Int) = NotificationCompat.Builder(context, Channels.SCAN)
        .setSmallIcon(R.drawable.ic_stat_message)
        .setContentTitle(context.getString(title))
        .setContentText(if (total > 0) context.getString(R.string.scan_progress_text, done, total) else null)
        .setProgress(total, done, total == 0)
        .setOngoing(true)
        .setOnlyAlertOnce(true)
        .setSilent(true)
        .setShowWhen(false)
        .setCategory(NotificationCompat.CATEGORY_PROGRESS)
        .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
        .setContentIntent(openApp())
        .addAction(0, context.getString(R.string.action_stop), WorkManager.getInstance(context).createCancelPendingIntent(workId))

    private fun post(tag: String?, id: Long, builder: NotificationCompat.Builder) {
        if (!manager.areNotificationsEnabled()) return
        try {
            manager.notify(tag, id.toInt(), builder.build())
        } catch (_: SecurityException) {
            // POST_NOTIFICATIONS revoked: nothing to do.
        }
    }

    private fun openApp(): PendingIntent = PendingIntent.getActivity(
        context, 0, Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    private fun openThread(threadId: Long, folder: Int): PendingIntent = PendingIntent.getActivity(
        context, threadId.toInt(), MainActivity.threadIntent(context, threadId, folder).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    fun displayAddress(address: String): String =
        PhoneNumberUtils.formatNumber(address, Locale.getDefault().country.ifEmpty { "US" }) ?: address

    private companion object {
        const val TAG_THREAD = "thread"
        const val TAG_SPAM = "spam"
        const val TAG_MMS = "mms"
        const val TAG_SCAN_DONE = "scan_done"
    }
}
