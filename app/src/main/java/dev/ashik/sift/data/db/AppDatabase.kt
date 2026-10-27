package dev.ashik.sift.data.db

import android.content.Context
import androidx.room3.Database
import androidx.room3.Room
import androidx.room3.RoomDatabase
import androidx.room3.migration.Migration
import androidx.sqlite.SQLiteConnection
import androidx.sqlite.execSQL
import androidx.sqlite.driver.bundled.BundledSQLiteDriver
import kotlinx.coroutines.Dispatchers

@Database(
    entities = [
        MessageEntity::class,
        ClassificationEntity::class,
        ConversationEntity::class,
        ContactCacheEntity::class,
        FeedbackEventEntity::class,
        TrainingExampleEntity::class,
        MessageEmbeddingEntity::class,
    ],
    version = 4,
    exportSchema = true,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun messages(): MessageDao
    abstract fun classifications(): ClassificationDao
    abstract fun conversations(): ConversationDao
    abstract fun contacts(): ContactsCacheDao
    abstract fun feedback(): FeedbackDao
    abstract fun trainingExamples(): TrainingExampleDao
    abstract fun embeddings(): EmbeddingDao

    companion object {
        /**
         * v2: classification is always per message, so per-sender verdicts are gone. Verdicts that came from the
         * removed sender rules go back to PENDING; the startup recovery pass re-classifies them from their content.
         */
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override suspend fun migrate(connection: SQLiteConnection) {
                connection.execSQL("DROP TABLE IF EXISTS sender_verdicts")
                connection.execSQL("ALTER TABLE feedback_events DROP COLUMN prevSenderVerdict")
                connection.execSQL("ALTER TABLE feedback_events DROP COLUMN prevSenderSourceId")
                connection.execSQL(
                    "UPDATE classifications SET state = ${ClassState.PENDING}, effectiveLabel = ${LabelCode.HAM}, " +
                        "effectiveSource = '${Source.PENDING}' WHERE userLabel IS NULL AND reason IN " +
                        "('SENDER_MARKED_SPAM', 'SENDER_MARKED_NOT_SPAM', 'TRUSTED_CONTACT', 'TRUSTED_REPLIED')",
                )
            }
        }

        /** v3: the user's marks are the only training signal, so examples inferred from replies are dropped. */
        val MIGRATION_2_3 = object : Migration(2, 3) {
            override suspend fun migrate(connection: SQLiteConnection) {
                connection.execSQL("DELETE FROM training_examples WHERE implicit = 1")
                connection.execSQL("ALTER TABLE training_examples DROP COLUMN implicit")
            }
        }

        /** v4: Sift only receives, so the send queue and the conversations' can-reply flag are gone. */
        val MIGRATION_3_4 = object : Migration(3, 4) {
            override suspend fun migrate(connection: SQLiteConnection) {
                connection.execSQL("DROP TABLE IF EXISTS pending_sends")
                connection.execSQL("ALTER TABLE conversations DROP COLUMN canReply")
            }
        }

        fun build(context: Context, name: String? = "sift.db"): AppDatabase {
            val builder = if (name == null) {
                Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            } else {
                Room.databaseBuilder(context, AppDatabase::class.java, name)
            }
            return builder
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4)
                .setDriver(BundledSQLiteDriver())
                .setQueryCoroutineContext(Dispatchers.IO)
                .build()
        }
    }
}
