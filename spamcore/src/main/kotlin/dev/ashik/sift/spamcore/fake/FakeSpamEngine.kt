package dev.ashik.sift.spamcore.fake

import dev.ashik.sift.spamcore.Cue
import dev.ashik.sift.spamcore.DecisionPolicy
import dev.ashik.sift.spamcore.EngineInfo
import dev.ashik.sift.spamcore.ExampleStore
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.Scored
import dev.ashik.sift.spamcore.Sensitivity
import dev.ashik.sift.spamcore.SmsInput
import dev.ashik.sift.spamcore.SpamEngine
import dev.ashik.sift.spamcore.SpamLabel
import dev.ashik.sift.spamcore.TrainingExample
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.spamcore.Verdict
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.exp

/**
 * Keyword-based stand-in for the real engine, used in app tests. Follows the same decision order as the real
 * engine, always on the message's own content: a similar labeled message (token Jaccard), then a keyword score
 * against per-sensitivity thresholds.
 */
class FakeSpamEngine(private val store: ExampleStore? = null) : SpamEngine {
    private val mutex = Mutex()
    private val examples = LinkedHashMap<Long, TrainingExample>()
    private var version = 0
    private var loaded = false

    private val _info = MutableStateFlow(EngineInfo("fake-keywords", 0, 0, null, encoderAvailable = false))
    override val info: StateFlow<EngineInfo> = _info

    override suspend fun warmUp() {
        mutex.withLock {
            if (loaded) return
            store?.loadAll()?.forEach { examples[it.messageId] = it }
            loaded = true
            publish()
        }
    }

    override suspend fun classify(input: SmsInput, policy: DecisionPolicy, embedding: FloatArray?): Scored {
        warmUp()
        val tokens = tokens(input.body)
        val match = mutex.withLock {
            examples.values
                .map { it to jaccard(tokens, tokens(it.body)) }
                .maxByOrNull { it.second }
        }
        if (match != null && match.second >= 0.6) {
            val (example, _) = match
            val label = if (example.label == UserLabel.SPAM) SpamLabel.SPAM else SpamLabel.HAM
            val reason = if (example.label == UserLabel.SPAM) Reason.SIMILAR_TO_SPAM else Reason.SIMILAR_TO_HAM
            return Scored(verdict(label, reason).copy(similarMessageId = example.messageId), null)
        }

        val lower = input.body.lowercase()
        val cues = KEYWORDS.filter { (k, _) -> k in lower }.map { (k, w) -> Cue("Contains \"$k\"", w) }
        val z = -3.0f + cues.sumOf { it.weight.toDouble() }.toFloat()
        val p = (1.0 / (1.0 + exp(-z.toDouble()))).toFloat()
        val spamAt = when (policy.sensitivity) {
            Sensitivity.CONSERVATIVE -> 0.9f
            Sensitivity.BALANCED -> 0.7f
            Sensitivity.AGGRESSIVE -> 0.5f
        }
        val label = when {
            p >= spamAt -> SpamLabel.SPAM
            p >= 0.35f -> SpamLabel.SUSPICIOUS
            else -> SpamLabel.HAM
        }
        return Scored(Verdict(label, p, Reason.MODEL, cues, modelVersion = version), null)
    }

    override suspend fun learn(examples: List<TrainingExample>) {
        mutex.withLock {
            examples.forEach { this.examples[it.messageId] = it }
            version++
            publish()
        }
        store?.upsert(examples)
    }

    override suspend fun forget(messageIds: Collection<Long>) {
        mutex.withLock {
            messageIds.forEach { examples.remove(it) }
            version++
            publish()
        }
        store?.delete(messageIds)
    }

    override suspend fun reset() {
        mutex.withLock {
            examples.clear()
            version = 0
            publish()
        }
        store?.clear()
    }

    override suspend fun awaitIdle() = Unit

    private fun publish() {
        _info.value = EngineInfo(
            bundleVersion = "fake-keywords",
            modelVersion = version,
            examples = examples.size,
            lastTrainedAt = if (version > 0) System.currentTimeMillis() else null,
            encoderAvailable = false,
        )
    }

    private fun verdict(label: SpamLabel, reason: Reason) = Verdict(label, Float.NaN, reason, modelVersion = version)

    private fun tokens(text: String): Set<String> =
        text.lowercase().split(Regex("[^a-z0-9]+")).filter { it.length > 2 }.toSet()

    private fun jaccard(a: Set<String>, b: Set<String>): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val inter = a.count { it in b }
        return inter.toDouble() / (a.size + b.size - inter)
    }

    private companion object {
        val KEYWORDS = listOf(
            "winner" to 2.0f, "won" to 1.0f, "prize" to 2.0f, "claim" to 1.5f, "gift card" to 2.0f,
            "free" to 1.0f, "urgent" to 1.5f, "verify" to 1.0f, "suspended" to 1.5f, "unpaid" to 1.5f,
            "toll" to 1.0f, "redeliver" to 2.0f, "click" to 1.0f, "bit.ly" to 2.0f, "tinyurl" to 2.0f,
            "http" to 1.0f, "crypto" to 1.5f, "loan" to 1.0f, "reply stop" to 0.5f, "$" to 0.5f,
        )
    }
}
