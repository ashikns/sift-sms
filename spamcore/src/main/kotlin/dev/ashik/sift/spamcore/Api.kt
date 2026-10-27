package dev.ashik.sift.spamcore

import kotlinx.coroutines.flow.StateFlow

/** How the app's address normalizer classified a sender. */
enum class SenderKind { PHONE, SHORT_CODE, ALPHANUMERIC, EMAIL, UNKNOWN }

/** A label the user gave a message. */
enum class UserLabel { HAM, SPAM }

/** The three-way verdict shown in the UI. SUSPICIOUS stays in the inbox with a "Possible spam" badge. */
enum class SpamLabel { HAM, SUSPICIOUS, SPAM }

enum class Sensitivity { CONSERVATIVE, BALANCED, AGGRESSIVE }

/**
 * Why a verdict was reached; drives the "Why?" explanation in the UI. Every verdict is about one message's own
 * content, since one sender can send both a promo and a one-time passcode.
 */
enum class Reason {
    SIMILAR_TO_SPAM,
    SIMILAR_TO_HAM,
    MODEL,
    MODEL_FALLBACK,
    OTP_GUARDRAIL,
    UNAVAILABLE,
}

/**
 * Facts about the channel a message arrived on. They feed the message's own score (sender-type features); the sender
 * never decides a verdict by itself.
 *
 * @param key normalized sender key: E.164 number, short-code digits, upper-cased alphanumeric ID, or lower-cased email.
 *   Only used so "not spam" near-duplicate matches can require the same sender.
 * @param international the number's country differs from the SIM's.
 * @param firstContact no earlier message from this sender exists.
 */
data class SenderInfo(
    val key: String,
    val kind: SenderKind,
    val international: Boolean = false,
    val firstContact: Boolean = false,
)

data class DecisionPolicy(
    val sensitivity: Sensitivity = Sensitivity.BALANCED,
)

data class SmsInput(
    val body: String,
    val sender: SenderInfo,
)

/** One human-readable cue behind a verdict, e.g. "Shortened link", with its contribution in logits. */
data class Cue(val text: String, val weight: Float)

data class Verdict(
    val label: SpamLabel,
    /** Model P(spam), or NaN when the model wasn't consulted (rule-based verdicts). */
    val probability: Float,
    val reason: Reason,
    val cues: List<Cue> = emptyList(),
    /** For SIMILAR_TO_* reasons: the labeled message this one matched. */
    val similarMessageId: Long? = null,
    /** Version of the personalized model that produced the verdict (0 = base model). */
    val modelVersion: Int = 0,
)

/** A verdict plus the message's embedding, which the app caches so re-scoring later skips the encoder. */
class Scored(val verdict: Verdict, val embedding: FloatArray?)

/**
 * A message the user explicitly marked Spam or Not spam: the only thing the engine learns from after installation.
 * Bodies are kept so learning survives deleting the message.
 */
data class TrainingExample(
    val messageId: Long,
    val body: String,
    val senderKind: SenderKind,
    val international: Boolean,
    val firstContact: Boolean,
    val label: UserLabel,
    val createdAt: Long,
    val embedding: FloatArray?,
    /** The sender's normalized key (see [SenderInfo.key]); lets "not spam" memory match the same sender. */
    val senderKey: String = "",
) {
    override fun equals(other: Any?) = other is TrainingExample && other.messageId == messageId
    override fun hashCode() = messageId.hashCode()
}

data class EngineInfo(
    val bundleVersion: String,
    val modelVersion: Int,
    /** Number of messages the user has marked. */
    val examples: Int,
    val lastTrainedAt: Long?,
    val encoderAvailable: Boolean,
)

/** Persistent storage for training examples, implemented by the app (Room). */
interface ExampleStore {
    suspend fun loadAll(): List<TrainingExample>
    suspend fun upsert(examples: List<TrainingExample>)
    suspend fun delete(messageIds: Collection<Long>)
}

interface SpamEngine {
    val info: StateFlow<EngineInfo>

    /** Loads the model and learned state; safe to call repeatedly. */
    suspend fun warmUp()

    /**
     * Classifies one message. Pass a cached [embedding] to skip the encoder (re-scoring).
     * Never throws: failures produce a fail-open HAM verdict with [Reason.UNAVAILABLE].
     */
    suspend fun classify(input: SmsInput, policy: DecisionPolicy, embedding: FloatArray? = null): Scored

    /** Adds or replaces the user's marks. Near-duplicate memory updates immediately; retraining is debounced. */
    suspend fun learn(examples: List<TrainingExample>)

    suspend fun forget(messageIds: Collection<Long>)

    /** Suspends until any pending retrain has finished. */
    suspend fun awaitIdle()
}
