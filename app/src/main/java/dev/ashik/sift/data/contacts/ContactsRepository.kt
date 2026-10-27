package dev.ashik.sift.data.contacts

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.provider.ContactsContract.CommonDataKinds.Phone
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ContactCacheEntity
import dev.ashik.sift.telephony.AddressNormalizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Contact names/photos for senders, cached in Room by sender key. Degrades to "no contacts" without permission. */
class ContactsRepository(
    private val context: Context,
    private val db: AppDatabase,
    private val normalizer: AddressNormalizer,
) {
    fun hasPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED

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
        val rows = db.messages().phoneSenderKeys().map { key ->
            val hit = byKey[key]
            ContactCacheEntity(key, hit?.first, hit?.second)
        }
        rows.chunked(500).forEach { db.contacts().upsertAll(it) }
    }
}
