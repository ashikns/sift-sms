package dev.ashik.sift.data.db

import android.content.Context
import androidx.room3.Database
import androidx.room3.Room
import androidx.room3.RoomDatabase
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
    version = 5,
    exportSchema = false,
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
        fun build(context: Context, name: String? = "sift.db"): AppDatabase {
            val builder = if (name == null) {
                Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            } else {
                Room.databaseBuilder(context, AppDatabase::class.java, name)
            }
            return builder
                .fallbackToDestructiveMigration(dropAllTables = true)
                .setDriver(BundledSQLiteDriver())
                .setQueryCoroutineContext(Dispatchers.IO)
                .build()
        }
    }
}
