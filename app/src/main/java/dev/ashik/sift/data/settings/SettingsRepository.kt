package dev.ashik.sift.data.settings

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStoreFile
import dev.ashik.sift.spamcore.DecisionPolicy
import dev.ashik.sift.spamcore.Sensitivity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

data class AppSettings(
    val sensitivity: Sensitivity = Sensitivity.BALANCED,
    val spamNotifications: Boolean = false,
    val initialImportDone: Boolean = false,
) {
    val policy: DecisionPolicy get() = DecisionPolicy(sensitivity)
}

class SettingsRepository(context: Context) {
    private val store: DataStore<Preferences> =
        PreferenceDataStoreFactory.create(produceFile = { context.preferencesDataStoreFile("settings") })

    val flow: Flow<AppSettings> = store.data.map { p ->
        AppSettings(
            sensitivity = p[SENSITIVITY]?.let { runCatching { Sensitivity.valueOf(it) }.getOrNull() } ?: Sensitivity.BALANCED,
            spamNotifications = p[SPAM_NOTIFICATIONS] ?: false,
            initialImportDone = p[INITIAL_IMPORT_DONE] ?: false,
        )
    }

    suspend fun current(): AppSettings = flow.first()

    suspend fun setSensitivity(value: Sensitivity) = store.edit { it[SENSITIVITY] = value.name }
    suspend fun setSpamNotifications(value: Boolean) = store.edit { it[SPAM_NOTIFICATIONS] = value }
    suspend fun setInitialImportDone(value: Boolean) = store.edit { it[INITIAL_IMPORT_DONE] = value }

    private companion object {
        val SENSITIVITY = stringPreferencesKey("sensitivity")
        val SPAM_NOTIFICATIONS = booleanPreferencesKey("spam_notifications")
        val INITIAL_IMPORT_DONE = booleanPreferencesKey("initial_import_done")
    }
}
