package dev.ashik.sift.data.contacts

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.ContactsContract
import android.provider.ContactsContract.CommonDataKinds.Phone
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ContactCacheEntity
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.telephony.AddressNormalizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class ContactPhone(val name: String, val number: String, val label: String?, val photoUri: String?)

/** Contact names/photos for senders, cached in Room by sender key. Degrades to "no contacts" without permission. */
class ContactsRepository(
    private val context: Context,
    private val db: AppDatabase,
    private val normalizer: AddressNormalizer,
) {
    fun hasPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED

    /** Cached lookup; refreshes entries older than a day. */
    suspend fun lookup(senderKey: String, address: String, kind: SenderKind): ContactCacheEntity? {
        if (kind != SenderKind.PHONE && kind != SenderKind.SHORT_CODE) return null
        val cached = db.contacts().get(senderKey)
        val now = System.currentTimeMillis()
        if (cached != null && now - cached.refreshedAt < DAY_MS) return cached
        if (!hasPermission()) return cached
        val found = withContext(Dispatchers.IO) { phoneLookup(address) }
        val row = ContactCacheEntity(senderKey, found?.name, found?.photoUri, isContact = found != null, refreshedAt = now)
        db.contacts().upsertAll(listOf(row))
        return row
    }

    /** One pass over all phone numbers, matched to every phone sender in the mirror. */
    suspend fun refreshAll() {
        if (!hasPermission()) return
        val byKey = withContext(Dispatchers.IO) {
            val map = HashMap<String, Pair<String, String?>>()
            context.contentResolver.query(
                Phone.CONTENT_URI,
                arrayOf(Phone.NORMALIZED_NUMBER, Phone.NUMBER, Phone.DISPLAY_NAME_PRIMARY, Phone.PHOTO_THUMBNAIL_URI),
                null, null, null,
            )?.use { c ->
                while (c.moveToNext()) {
                    val name = c.getString(2) ?: continue
                    val number = c.getString(0) ?: c.getString(1) ?: continue
                    map.putIfAbsent(normalizer.normalize(number).key, name to c.getString(3))
                }
            }
            map
        }
        val now = System.currentTimeMillis()
        val rows = db.messages().phoneSenderKeys().map { key ->
            val hit = byKey[key]
            ContactCacheEntity(key, hit?.first, hit?.second, isContact = hit != null, refreshedAt = now)
        }
        rows.chunked(500).forEach { db.contacts().upsertAll(it) }
    }

    private fun phoneLookup(address: String): ContactPhone? {
        if (address.isBlank()) return null
        val uri = Uri.withAppendedPath(ContactsContract.PhoneLookup.CONTENT_FILTER_URI, Uri.encode(address))
        return runCatching {
            context.contentResolver.query(
                uri,
                arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME, ContactsContract.PhoneLookup.PHOTO_THUMBNAIL_URI),
                null, null, null,
            )?.use { c -> if (c.moveToFirst()) ContactPhone(c.getString(0) ?: address, address, null, c.getString(1)) else null }
        }.getOrNull()
    }

    private companion object {
        const val DAY_MS = 24L * 60 * 60 * 1000
    }
}
