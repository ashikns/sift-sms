package dev.ashik.sift.telephony

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.Telephony.Sms
import dev.ashik.sift.data.db.MsgType

/** Thrown when a provider write is silently rejected: the app is no longer the default SMS app. */
class RoleLostException : IllegalStateException("Sift is no longer the default SMS app")

/** One row of the system SMS provider. [body] is null when the query skipped it. */
data class ProviderSms(
    val id: Long,
    val threadId: Long,
    val address: String,
    val body: String?,
    val date: Long,
    val type: Int,
    val read: Boolean,
    val status: Int,
)

/** Everything the app writes to or reads from `content://sms` goes through here. */
class SmsProviderGateway(context: Context) {
    private val resolver = context.contentResolver

    fun insertIncoming(
        address: String, body: String, date: Long, dateSent: Long, subId: Int, protocol: Int,
        replyPathPresent: Boolean, serviceCenter: String?,
    ): Long {
        val values = ContentValues().apply {
            put(Sms.ADDRESS, address)
            put(Sms.BODY, body)
            put(Sms.DATE, date)
            put(Sms.DATE_SENT, dateSent)
            put(Sms.READ, 0)
            put(Sms.SEEN, 0)
            put(Sms.TYPE, MsgType.INBOX)
            put(Sms.SUBSCRIPTION_ID, subId)
            put(Sms.PROTOCOL, protocol)
            put(Sms.REPLY_PATH_PRESENT, if (replyPathPresent) 1 else 0)
            if (serviceCenter != null) put(Sms.SERVICE_CENTER, serviceCenter)
        }
        return insert(Sms.Inbox.CONTENT_URI, values)
    }

    private fun insert(uri: Uri, values: ContentValues): Long {
        val result = resolver.insert(uri, values) ?: throw RoleLostException()
        val id = runCatching { ContentUris.parseId(result) }.getOrDefault(0L)
        if (id <= 0L) throw RoleLostException()
        return id
    }

    /** A message re-delivered after a reboot must not be inserted twice. */
    fun findDuplicateIncoming(address: String, dateSent: Long, body: String): Long? =
        resolver.query(
            Sms.CONTENT_URI, arrayOf(Sms._ID),
            "${Sms.TYPE} = ${MsgType.INBOX} AND ${Sms.ADDRESS} = ? AND ${Sms.DATE_SENT} = ? AND ${Sms.BODY} = ?",
            arrayOf(address, dateSent.toString(), body), null,
        )?.use { c -> if (c.moveToFirst()) c.getLong(0) else null }

    fun markThreadRead(threadId: Long): Int {
        val values = ContentValues().apply {
            put(Sms.READ, 1)
            put(Sms.SEEN, 1)
        }
        return resolver.update(
            Sms.CONTENT_URI, values, "${Sms.THREAD_ID} = ? AND (${Sms.READ} = 0 OR ${Sms.SEEN} = 0)",
            arrayOf(threadId.toString()),
        )
    }

    fun deleteMessage(id: Long): Int = resolver.delete(ContentUris.withAppendedId(Sms.CONTENT_URI, id), null, null)

    /** Deletes only the thread's SMS rows; MMS rows the app can't show are left alone. */
    fun deleteThreadSms(threadId: Long): Int =
        resolver.delete(Sms.CONTENT_URI, "${Sms.THREAD_ID} = ?", arrayOf(threadId.toString()))

    fun query(id: Long): ProviderSms? =
        resolver.query(ContentUris.withAppendedId(Sms.CONTENT_URI, id), FULL, null, null, null)
            ?.use { c -> if (c.moveToFirst()) read(c, withBody = true) else null }

    /** The newest [limit] rows, newest first (the cursor is read lazily and closed early). */
    fun queryNewest(limit: Int): List<ProviderSms> =
        resolver.query(Sms.CONTENT_URI, FULL, NOT_DRAFT, null, "${Sms._ID} DESC")?.use { c ->
            val out = ArrayList<ProviderSms>(minOf(limit, c.count))
            while (out.size < limit && c.moveToNext()) out += read(c, withBody = true)
            out
        }.orEmpty()

    /** Every row without bodies, ascending `_id`, for reconciliation. */
    fun allKeys(): List<ProviderSms> =
        resolver.query(Sms.CONTENT_URI, KEYS, NOT_DRAFT, null, "${Sms._ID} ASC")?.use { c ->
            val out = ArrayList<ProviderSms>(c.count)
            while (c.moveToNext()) out += read(c, withBody = false)
            out
        }.orEmpty()

    fun keysAfter(afterId: Long): List<ProviderSms> =
        resolver.query(Sms.CONTENT_URI, KEYS, "$NOT_DRAFT AND ${Sms._ID} > ?", arrayOf(afterId.toString()), "${Sms._ID} ASC")
            ?.use { c ->
                val out = ArrayList<ProviderSms>(c.count)
                while (c.moveToNext()) out += read(c, withBody = false)
                out
            }.orEmpty()

    fun keysSince(sinceDate: Long): List<ProviderSms> =
        resolver.query(Sms.CONTENT_URI, KEYS, "$NOT_DRAFT AND ${Sms.DATE} >= ?", arrayOf(sinceDate.toString()), "${Sms._ID} ASC")
            ?.use { c ->
                val out = ArrayList<ProviderSms>(c.count)
                while (c.moveToNext()) out += read(c, withBody = false)
                out
            }.orEmpty()

    /** Full rows (with bodies) for [ids], fetched in chunks. */
    fun rows(ids: List<Long>): List<ProviderSms> = ids.chunked(500).flatMap { chunk ->
        val placeholders = chunk.joinToString(",") { "?" }
        resolver.query(
            Sms.CONTENT_URI, FULL, "${Sms._ID} IN ($placeholders)", chunk.map { it.toString() }.toTypedArray(), null,
        )?.use { c ->
            val out = ArrayList<ProviderSms>(c.count)
            while (c.moveToNext()) out += read(c, withBody = true)
            out
        }.orEmpty()
    }

    fun count(): Int = resolver.query(Sms.CONTENT_URI, arrayOf(Sms._ID), NOT_DRAFT, null, null)?.use { it.count } ?: 0

    private fun read(c: Cursor, withBody: Boolean): ProviderSms = ProviderSms(
        id = c.getLong(c.getColumnIndexOrThrow(Sms._ID)),
        threadId = c.getLong(c.getColumnIndexOrThrow(Sms.THREAD_ID)),
        address = c.getString(c.getColumnIndexOrThrow(Sms.ADDRESS)).orEmpty(),
        body = if (withBody) c.getString(c.getColumnIndexOrThrow(Sms.BODY)).orEmpty() else null,
        date = c.getLong(c.getColumnIndexOrThrow(Sms.DATE)),
        type = c.getInt(c.getColumnIndexOrThrow(Sms.TYPE)),
        read = c.getInt(c.getColumnIndexOrThrow(Sms.READ)) != 0,
        status = c.getInt(c.getColumnIndexOrThrow(Sms.STATUS)),
    )

    private companion object {
        val KEYS = arrayOf(Sms._ID, Sms.THREAD_ID, Sms.ADDRESS, Sms.DATE, Sms.TYPE, Sms.READ, Sms.STATUS)
        val FULL = KEYS + Sms.BODY
        const val NOT_DRAFT = "${Sms.TYPE} != ${MsgType.DRAFT}"
    }
}
