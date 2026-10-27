package dev.ashik.sift.spamcore.memory

import dev.ashik.sift.spamcore.features.SparseVector

/**
 * Near-duplicate memory of messages the user labeled (explicit labels only), matched by n-gram cosine.
 * Immutable: updates return a new instance, so readers never lock. Mirrored by ml/scripts/s08_simulate.py Memory.
 */
class LabeledMemory private constructor(
    private val entries: Map<Long, Entry>,
    val tauSpam: Double,
    val tauHam: Double,
) {
    class Entry(val key: Long, val label: Int, val sparse: SparseVector, val senderKey: String, val domains: Set<String>)

    data class Match(val label: Int, val key: Long, val cosine: Double)

    constructor(tauSpam: Double, tauHam: Double) : this(emptyMap(), tauSpam, tauHam)

    val size: Int get() = entries.size

    fun with(entry: Entry) = LabeledMemory(LinkedHashMap(entries).apply { put(entry.key, entry) }, tauSpam, tauHam)

    fun withAll(added: Collection<Entry>) =
        LabeledMemory(LinkedHashMap(entries).apply { added.forEach { put(it.key, it) } }, tauSpam, tauHam)

    fun without(keys: Collection<Long>) =
        LabeledMemory(LinkedHashMap(entries).apply { keys.forEach { remove(it) } }, tauSpam, tauHam)

    /** Ham matches also need the same sender, or no link domain the matched message didn't have. */
    fun match(sparse: SparseVector, senderKey: String, domains: Set<String>): Match? {
        var bestSpam: Match? = null
        var bestHam: Match? = null
        for (e in entries.values) {
            val cos = SparseVector.cosine(sparse, e.sparse)
            if (e.label == 1) {
                if (cos >= tauSpam && (bestSpam == null || cos > bestSpam.cosine)) bestSpam = Match(1, e.key, cos)
            } else if (cos >= tauHam && (senderKey == e.senderKey || e.domains.containsAll(domains))) {
                if (bestHam == null || cos > bestHam.cosine) bestHam = Match(0, e.key, cos)
            }
        }
        if (bestHam != null && (bestSpam == null || bestHam.cosine >= bestSpam.cosine)) return bestHam
        return bestSpam
    }
}
