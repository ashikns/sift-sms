@file:Suppress("ArrayInDataClass")

package dev.ashik.sift.data.db

import androidx.room3.Embedded
import androidx.room3.Entity
import androidx.room3.Index
import androidx.room3.PrimaryKey

/**
 * Mirror of one row of the system SMS provider. `id` is the provider `_id`; because the provider reuses ids,
 * [fingerprint] (address + date) detects a reused id pointing at a different message.
 */
@Entity(
    tableName = "messages",
    indices = [Index(value = ["threadId", "date"]), Index(value = ["senderKey"]), Index(value = ["type", "date"])],
)
data class MessageEntity(
    @PrimaryKey val id: Long,
    val threadId: Long,
    val address: String,
    val senderKey: String,
    val senderKind: String,
    val body: String,
    val date: Long,
    val type: Int,
    val status: Int,
    val read: Boolean,
    val fingerprint: Long,
    val mutableHash: Int,
    /** Set when the row disappeared from the provider but is too young to delete (OTP protection can hide rows). */
    val missingSince: Long? = null,
)

/** The spam verdict and user label for one incoming message. */
@Entity(tableName = "classifications")
data class ClassificationEntity(
    @PrimaryKey val messageId: Long,
    val state: Int = ClassState.PENDING,
    val modelLabel: Int? = null,
    val modelScore: Float? = null,
    val reason: String? = null,
    /** JSON array of {"text","weight"} cues. */
    val cues: String? = null,
    val similarMessageId: Long? = null,
    val modelVersion: Int? = null,
    val userLabel: Int? = null,
    val userLabeledAt: Long? = null,
    val effectiveLabel: Int = LabelCode.HAM,
    val effectiveSource: String = Source.PENDING,
    /** True once a notification was posted, or when the message must never notify (imported history). */
    val notified: Boolean = false,
    val classifiedAt: Long? = null,
)

/** One row per thread, rebuilt by the projector whenever the thread's messages or verdicts change. */
@Entity(tableName = "conversations", indices = [Index(value = ["folder", "latestDate"])])
data class ConversationEntity(
    @PrimaryKey val threadId: Long,
    val senderKey: String,
    val address: String,
    val latestMessageId: Long,
    val latestDate: Long,
    val snippet: String,
    val latestType: Int,
    val unreadCount: Int,
    val messageCount: Int,
    val folder: Int,
    /** Effective label of the latest incoming message (drives the "Possible spam" badge). */
    val latestLabel: Int,
)

@Entity(tableName = "contacts_cache")
data class ContactCacheEntity(
    @PrimaryKey val senderKey: String,
    val displayName: String?,
    val photoUri: String?,
)

/** One explicit mark by the user, for stats. */
@Entity(tableName = "feedback_events", indices = [Index(value = ["createdAt"])])
data class FeedbackEventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val messageId: Long,
    /** The user label code. */
    val label: Int,
    /** What the engine said before the mark: a disagreement counts as a correction in stats. */
    val engineLabelAtMark: Int?,
    val createdAt: Long,
)

/** A labeled message the spam engine learns from (spamcore's ExampleStore). */
@Entity(tableName = "training_examples")
data class TrainingExampleEntity(
    @PrimaryKey val messageId: Long,
    val body: String,
    val senderKey: String,
    val senderKind: String,
    val international: Boolean,
    val firstContact: Boolean,
    val label: Int,
    val createdAt: Long,
    /** float32 little-endian, or null if no embedding was available. */
    val embedding: ByteArray?,
)

/** Cached encoder output per message, so re-scoring after learning skips the encoder. */
@Entity(tableName = "message_embeddings")
data class MessageEmbeddingEntity(
    @PrimaryKey val messageId: Long,
    val fingerprint: Long,
    val bundleVersion: String,
    val vector: ByteArray,
    val createdAt: Long,
)

// ---- Query projections -------------------------------------------------------------------------------------------

data class ConversationRow(
    @Embedded val conversation: ConversationEntity,
    val displayName: String?,
    val photoUri: String?,
)

data class ThreadMessageRow(
    val id: Long,
    val threadId: Long,
    val address: String,
    val body: String,
    val date: Long,
    val type: Int,
    val status: Int,
    val read: Boolean,
    val effectiveLabel: Int,
    val effectiveSource: String?,
    val modelScore: Float?,
    val reason: String?,
    val cues: String?,
    val userLabel: Int?,
    val classState: Int?,
    val modelVersion: Int?,
)

data class MessageKey(
    val id: Long,
    val fingerprint: Long,
    val mutableHash: Int,
    val date: Long,
    val threadId: Long,
    val missingSince: Long?,
)

data class ThreadFolder(val threadId: Long, val folder: Int)

/** A learning entry as the review screen lists it: no embedding, plus the sender's contact name if any. */
data class LearningEntryRow(
    val messageId: Long,
    val body: String,
    val label: Int,
    val createdAt: Long,
    /** The message's address while it's still stored (formatted like the rest of the app), else the sender key. */
    val address: String,
    val displayName: String?,
)

data class WeekCount(val week: Long, val count: Int)
