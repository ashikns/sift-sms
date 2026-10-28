package dev.ashik.sift.data.db

import androidx.room3.Dao
import androidx.room3.Query
import androidx.room3.Upsert
import kotlinx.coroutines.flow.Flow

@Dao
interface MessageDao {
    @Upsert
    suspend fun upsert(message: MessageEntity)

    @Upsert
    suspend fun upsertAll(messages: List<MessageEntity>)

    @Query("SELECT * FROM messages WHERE id = :id")
    suspend fun get(id: Long): MessageEntity?

    @Query("SELECT * FROM messages WHERE id IN (:ids)")
    suspend fun getAll(ids: List<Long>): List<MessageEntity>

    @Query("SELECT COUNT(*) FROM messages")
    suspend fun count(): Int

    @Query("SELECT id, fingerprint, mutableHash, date, threadId, missingSince FROM messages ORDER BY id")
    suspend fun allKeys(): List<MessageKey>

    @Query("DELETE FROM messages WHERE id IN (:ids)")
    suspend fun deleteIds(ids: List<Long>)

    @Query("SELECT id FROM messages WHERE threadId = :threadId")
    suspend fun idsInThread(threadId: Long): List<Long>

    @Query("UPDATE messages SET missingSince = :ts WHERE id IN (:ids) AND missingSince IS NULL")
    suspend fun markMissing(ids: List<Long>, ts: Long)

    @Query("UPDATE messages SET missingSince = NULL WHERE id IN (:ids)")
    suspend fun clearMissing(ids: List<Long>)

    @Query(
        "UPDATE messages SET type = :type, status = :status, read = :read, threadId = :threadId, " +
            "mutableHash = :mutableHash, missingSince = NULL WHERE id = :id",
    )
    suspend fun updateMutable(id: Long, type: Int, status: Int, read: Boolean, threadId: Long, mutableHash: Int)

    @Query("SELECT DISTINCT threadId FROM messages WHERE id IN (:ids)")
    suspend fun threadsOf(ids: List<Long>): List<Long>

    @Query("UPDATE messages SET read = 1 WHERE threadId = :threadId AND read = 0")
    suspend fun markThreadRead(threadId: Long): Int

    @Query("UPDATE messages SET read = 0, mutableHash = :mutableHash WHERE id = :id AND read = 1")
    suspend fun markUnread(id: Long, mutableHash: Int): Int

    @Query(
        "SELECT COUNT(*) FROM messages WHERE senderKey = :senderKey AND type = 1 AND id != :excludeId AND " +
            "(date < :before OR (date = :before AND id < :excludeId))",
    )
    suspend fun earlierIncomingCount(senderKey: String, excludeId: Long, before: Long): Int

    @Query(
        "SELECT m.id FROM messages m LEFT JOIN classifications c ON c.messageId = m.id " +
            "WHERE m.type = 1 AND c.userLabel IS NULL AND m.threadId NOT IN (:skipThreads) ORDER BY m.date DESC, m.id DESC",
    )
    suspend fun unlabeledIncomingIds(skipThreads: List<Long>): List<Long>

    @Query(
        "SELECT m.id FROM messages m LEFT JOIN classifications c ON c.messageId = m.id " +
            "WHERE m.type = 1 AND c.userLabel IS NULL AND m.threadId IN (:threadIds) ORDER BY m.date DESC, m.id DESC",
    )
    suspend fun unlabeledIncomingIdsIn(threadIds: List<Long>): List<Long>

    @Query(
        "SELECT m.id FROM messages m LEFT JOIN classifications c ON c.messageId = m.id " +
            "WHERE m.type = 1 AND (c.messageId IS NULL OR c.state = 0) ORDER BY m.date DESC, m.id DESC",
    )
    suspend fun unclassifiedIncomingIds(): List<Long>

    /** Incoming messages a rescan covers: every one without a user label, except pending ones (ClassifyPendingWorker's). */
    @Query(
        "SELECT COUNT(*) FROM messages m LEFT JOIN classifications c ON c.messageId = m.id " +
            "WHERE m.type = 1 AND c.userLabel IS NULL AND (c.messageId IS NULL OR c.state != 0)",
    )
    suspend fun rescanCount(): Int

    /** The [rescanCount] messages not classified again since [startedAt], newest first. */
    @Query(
        "SELECT m.id FROM messages m LEFT JOIN classifications c ON c.messageId = m.id " +
            "WHERE m.type = 1 AND c.userLabel IS NULL AND " +
            "(c.messageId IS NULL OR (c.state != 0 AND COALESCE(c.classifiedAt, 0) < :startedAt)) " +
            "ORDER BY m.date DESC, m.id DESC",
    )
    suspend fun rescanRemainingIds(startedAt: Long): List<Long>

    @Query(
        "SELECT m.id, m.threadId, m.address, m.body, m.date, m.type, m.status, m.read, " +
            "COALESCE(c.effectiveLabel, 0) AS effectiveLabel, c.effectiveSource AS effectiveSource, " +
            "c.modelScore AS modelScore, c.reason AS reason, c.cues AS cues, " +
            "c.userLabel AS userLabel, c.state AS classState, c.modelVersion AS modelVersion " +
            "FROM messages m LEFT JOIN classifications c ON c.messageId = m.id " +
            "WHERE m.threadId = :threadId ORDER BY m.date DESC, m.id DESC",
    )
    fun observeThread(threadId: Long): Flow<List<ThreadMessageRow>>

    @Query(
        "SELECT m.id, m.threadId, m.address, m.body, m.date, m.type, m.status, m.read, " +
            "COALESCE(c.effectiveLabel, 0) AS effectiveLabel, c.effectiveSource AS effectiveSource, " +
            "c.modelScore AS modelScore, c.reason AS reason, c.cues AS cues, " +
            "c.userLabel AS userLabel, c.state AS classState, c.modelVersion AS modelVersion " +
            "FROM messages m LEFT JOIN classifications c ON c.messageId = m.id " +
            "WHERE m.threadId = :threadId AND m.type = 1 AND m.read = 0 AND COALESCE(c.effectiveLabel, 0) != 2 " +
            "ORDER BY m.date DESC, m.id DESC LIMIT :limit",
    )
    suspend fun unreadNotSpam(threadId: Long, limit: Int): List<ThreadMessageRow>

    @Query(
        "SELECT m.id, m.threadId, m.address, m.body, m.date, m.type, m.status, m.read, " +
            "COALESCE(c.effectiveLabel, 0) AS effectiveLabel, c.effectiveSource AS effectiveSource, " +
            "c.modelScore AS modelScore, c.reason AS reason, c.cues AS cues, " +
            "c.userLabel AS userLabel, c.state AS classState, c.modelVersion AS modelVersion " +
            "FROM messages m LEFT JOIN classifications c ON c.messageId = m.id ORDER BY m.date DESC, m.id DESC LIMIT :limit",
    )
    suspend fun debugMessages(limit: Int): List<ThreadMessageRow>

    @Query("SELECT (date / 604800000) AS week, COUNT(*) AS count FROM messages WHERE type = 1 GROUP BY week")
    suspend fun incomingPerWeek(): List<WeekCount>

    @Query("SELECT DISTINCT senderKey FROM messages WHERE senderKind = 'PHONE'")
    suspend fun phoneSenderKeys(): List<String>

    @Query("SELECT threadId FROM messages WHERE senderKey = :senderKey ORDER BY date DESC LIMIT 1")
    suspend fun threadForSender(senderKey: String): Long?
}

@Dao
interface ClassificationDao {
    @Upsert
    suspend fun upsert(classification: ClassificationEntity)

    @Upsert
    suspend fun upsertAll(classifications: List<ClassificationEntity>)

    @Query("SELECT * FROM classifications WHERE messageId = :id")
    suspend fun get(id: Long): ClassificationEntity?

    @Query("SELECT * FROM classifications WHERE messageId IN (:ids)")
    suspend fun getAll(ids: List<Long>): List<ClassificationEntity>

    @Query("SELECT messageId FROM classifications WHERE state = 0 ORDER BY messageId DESC")
    suspend fun pendingIds(): List<Long>

    @Query("SELECT COUNT(*) FROM classifications WHERE state = 0")
    suspend fun pendingCount(): Int

    @Query("UPDATE classifications SET notified = 1 WHERE messageId = :id")
    suspend fun markNotified(id: Long)

    /** The user's labels are gone and the messages wait to be judged again (state 0, pending). */
    @Query("UPDATE classifications SET userLabel = NULL, userLabeledAt = NULL, state = 0 WHERE messageId IN (:ids)")
    suspend fun clearUserLabels(ids: List<Long>)

    @Query("DELETE FROM classifications WHERE messageId IN (:ids)")
    suspend fun deleteIds(ids: List<Long>)
}

@Dao
interface ConversationDao {
    @Query(
        "SELECT c.*, cc.displayName AS displayName, cc.photoUri AS photoUri FROM conversations c " +
            "LEFT JOIN contacts_cache cc ON cc.senderKey = c.senderKey " +
            "WHERE c.folder = :folder ORDER BY c.latestDate DESC, c.threadId DESC",
    )
    fun observeFolder(folder: Int): Flow<List<ConversationRow>>

    @Query(
        "SELECT c.*, cc.displayName AS displayName, cc.photoUri AS photoUri FROM conversations c " +
            "LEFT JOIN contacts_cache cc ON cc.senderKey = c.senderKey WHERE c.threadId = :threadId",
    )
    fun observe(threadId: Long): Flow<ConversationRow?>

    @Query("SELECT * FROM conversations ORDER BY latestDate DESC")
    suspend fun all(): List<ConversationEntity>

    @Query("SELECT threadId, folder FROM conversations")
    suspend fun folders(): List<ThreadFolder>

    @Query("DELETE FROM conversations WHERE threadId IN (:ids)")
    suspend fun deleteThreads(ids: List<Long>)

    @Query(PROJECTION_INSERT + PROJECTION_SELECT + " WHERE threadId IN (:ids) GROUP BY threadId) t" + PROJECTION_JOINS)
    suspend fun insertProjection(ids: List<Long>)

    companion object {
        private const val PROJECTION_INSERT =
            "INSERT OR REPLACE INTO conversations (threadId, senderKey, address, latestMessageId, latestDate, snippet, " +
                "latestType, unreadCount, messageCount, folder, latestLabel) "
        private const val PROJECTION_SELECT =
            "SELECT t.threadId, COALESCE(li.senderKey, lm.senderKey), COALESCE(li.address, lm.address), lm.id, lm.date, " +
                "lm.body, lm.type, " +
                "(SELECT COUNT(*) FROM messages u WHERE u.threadId = t.threadId AND u.type = 1 AND u.read = 0), t.cnt, " +
                "CASE WHEN COALESCE(cl.effectiveLabel, 0) = 2 THEN 1 ELSE 0 END, COALESCE(cl.effectiveLabel, 0) " +
                "FROM (SELECT threadId, COUNT(*) AS cnt FROM messages"
        private const val PROJECTION_JOINS =
            " JOIN messages lm ON lm.id = (SELECT id FROM messages WHERE threadId = t.threadId ORDER BY date DESC, id DESC LIMIT 1)" +
                " LEFT JOIN messages li ON li.id = (SELECT id FROM messages WHERE threadId = t.threadId AND type = 1 " +
                "ORDER BY date DESC, id DESC LIMIT 1)" +
                " LEFT JOIN classifications cl ON cl.messageId = li.id"
    }
}

@Dao
interface ContactsCacheDao {
    @Upsert
    suspend fun upsertAll(rows: List<ContactCacheEntity>)
}

@Dao
interface FeedbackDao {
    @Upsert
    suspend fun insertAll(events: List<FeedbackEventEntity>)

    /** Marks whose learning entries were deleted: no longer counted. */
    @Query("DELETE FROM feedback_events WHERE messageId IN (:ids)")
    suspend fun retractMarks(ids: List<Long>)

    @Query("SELECT COUNT(*) FROM feedback_events WHERE label = :label")
    fun observeExplicitCount(label: Int): Flow<Int>

    @Query(
        "SELECT (createdAt / 604800000) AS week, COUNT(*) AS count FROM feedback_events " +
            "WHERE (label = 1 AND COALESCE(engineLabelAtMark, 0) != 2) OR " +
            "(label = 0 AND engineLabelAtMark = 2) GROUP BY week",
    )
    suspend fun correctionsPerWeek(): List<WeekCount>
}

@Dao
interface TrainingExampleDao {
    @Upsert
    suspend fun upsertAll(rows: List<TrainingExampleEntity>)

    @Query("SELECT * FROM training_examples ORDER BY createdAt")
    suspend fun all(): List<TrainingExampleEntity>

    @Query("DELETE FROM training_examples WHERE messageId IN (:ids)")
    suspend fun deleteIds(ids: List<Long>)

    @Query("SELECT * FROM training_examples WHERE messageId = :id")
    suspend fun get(id: Long): TrainingExampleEntity?

    @Query(
        "SELECT t.messageId, t.body, t.label, t.createdAt, COALESCE(m.address, t.senderKey) AS address, cc.displayName AS displayName " +
            "FROM training_examples t LEFT JOIN messages m ON m.id = t.messageId " +
            "LEFT JOIN contacts_cache cc ON cc.senderKey = t.senderKey ORDER BY t.createdAt DESC, t.messageId DESC",
    )
    fun observeEntries(): Flow<List<LearningEntryRow>>
}

@Dao
interface EmbeddingDao {
    @Upsert
    suspend fun upsert(row: MessageEmbeddingEntity)

    @Query("SELECT * FROM message_embeddings WHERE messageId = :id")
    suspend fun get(id: Long): MessageEmbeddingEntity?

    @Query("DELETE FROM message_embeddings WHERE messageId IN (:ids)")
    suspend fun deleteIds(ids: List<Long>)
}
