package dev.ashik.sift.spamcore.train

import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.head.LinearHead
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/** A marked message with its features computed, ready for training. Label: 1 = spam, 0 = ham. */
class PreparedExample(
    val key: Long,
    val label: Int,
    val createdAt: Long,
    val embedding: FloatArray?,
    val meta: FloatArray,
    val sparse: SparseVector,
    /** The sender × content (SND) vector; empty without a sender identity. */
    val snd: SparseVector,
    /** The sender's identity (ml/SPEC.md sid), for the memory's "not spam" rule. */
    val sid: String? = null,
    val domains: Set<String> = emptySet(),
)

/** Prior strengths per weight block. A block whose prior is ≥ [FROZEN] is pinned at its base weights. */
data class Priors(val embed: Double = 1.0, val meta: Double = 4.0, val hash: Double = 0.5, val snd: Double = 0.5) {
    companion object {
        const val FROZEN = 1e5
    }
}

data class Weighting(
    val spam: Double = 1.0,
    val ham: Double = 1.0,
    val clusterCap: Double = 2.0,
    val clusterCos: Double = 0.8,
)

/** Mirrors ml/src/smsml/head.py example_weights: base weights over all marks, leader-clustered caps. */
object ExampleWeighting {
    fun weights(examples: List<PreparedExample>, w: Weighting): DoubleArray {
        val n = examples.size
        val base = DoubleArray(n) { i -> if (examples[i].label == 1) w.spam else w.ham }
        val leaders = ArrayList<Int>()
        val clusterOf = IntArray(n) { -1 }
        for (i in 0 until n) {
            val e = examples[i]
            var joined = -1
            for ((c, lead) in leaders.withIndex()) {
                val le = examples[lead]
                if (le.label == e.label && SparseVector.cosine(le.sparse, e.sparse) >= w.clusterCos) {
                    joined = c
                    break
                }
            }
            if (joined == -1) {
                clusterOf[i] = leaders.size
                leaders.add(i)
            } else {
                clusterOf[i] = joined
            }
        }
        val totals = DoubleArray(leaders.size)
        for (i in 0 until n) totals[clusterOf[i]] += base[i]
        val out = base.copyOf()
        for (i in 0 until n) {
            if (totals[clusterOf[i]] > w.clusterCap) out[i] = base[i] * (w.clusterCap / totals[clusterOf[i]])
        }
        return out
    }
}

/**
 * MAP logistic regression anchored to the base head (see ml/src/smsml/head.py personalize):
 *   minimize Σ cᵢ·logloss(zᵢ, yᵢ) + ½ Σ_g λ_g ‖w_g − w_base,g‖²
 * over the embedding block, the content metadata, the active hashed n-gram weights and the active sender × content
 * (SND) weights. The bias and the sender-type features stay frozen at the base weights, so marks only teach the model
 * about message content (a mark on one short-code promo must not shift every short-code message), and about a sender
 * only together with that message's words and link domains. Solved with L-BFGS (memory 7) and Armijo backtracking, in
 * double precision and deterministic order.
 */
class MapLogisticTrainer(
    private val memory: Int = 7,
    private val maxIterations: Int = 150,
    private val gradTolerance: Double = 1e-6,
) {
    data class Result(val head: LinearHead, val iterations: Int, val objective: Double, val gradNorm: Double)

    fun train(base: LinearHead, examples: List<PreparedExample>, weights: DoubleArray, priors: Priors, cancelled: () -> Boolean = { false }): Result {
        val use = examples.indices.filter { weights[it] > 0.0 }
        if (use.isEmpty()) return Result(base, 0, 0.0, 0.0)
        val active = use.flatMap { examples[it].sparse.indices.asIterable() }.toSortedSet().toIntArray()
        val sndActive = use.flatMap { examples[it].snd.indices.asIterable() }.toSortedSet().toIntArray()
        val pos = HashMap<Int, Int>(active.size * 2).apply { active.forEachIndexed { j, k -> put(k, j) } }
        val sndPos = HashMap<Int, Int>(sndActive.size * 2).apply { sndActive.forEachIndexed { j, k -> put(k, j) } }
        val de = LinearHead.EMBED_DIM
        val dm = Features.N_META
        val hOff = de + dm
        val sOff = hOff + active.size
        val d = sOff + sndActive.size
        val theta0 = DoubleArray(d)
        for (i in 0 until de) theta0[i] = base.wEmbed[i].toDouble()
        for (j in 0 until dm) theta0[de + j] = base.wMeta[j].toDouble()
        for ((j, k) in active.withIndex()) theta0[hOff + j] = base.wHash[k].toDouble()
        for ((j, k) in sndActive.withIndex()) theta0[sOff + j] = base.wSnd[k].toDouble()
        val lambda = DoubleArray(d) { i ->
            when {
                i < de -> priors.embed
                i < hOff -> priors.meta
                i < sOff -> priors.hash
                else -> priors.snd
            }
        }
        // Frozen coordinates get a zero gradient, so L-BFGS never moves them off the base weights: always the
        // sender-type features, plus any block whose prior is at least Priors.FROZEN.
        val frozen = BooleanArray(d) { i ->
            when {
                i < de -> priors.embed >= Priors.FROZEN
                i < hOff -> (i - de) in Features.SENDER_RANGE || priors.meta >= Priors.FROZEN
                i < sOff -> priors.hash >= Priors.FROZEN
                else -> priors.snd >= Priors.FROZEN
            }
        }
        val rows = use.map { i ->
            val e = examples[i]
            val emb = e.embedding?.let { em -> DoubleArray(de) { em[it].toDouble() - base.muEmbed[it] } }
            val meta = DoubleArray(dm) { e.meta[it].toDouble() - base.muMeta[it] }
            val hp = IntArray(e.sparse.size) { pos.getValue(e.sparse.indices[it]) }
            val hv = DoubleArray(e.sparse.size) { e.sparse.values[it].toDouble() }
            val sp = IntArray(e.snd.size) { sndPos.getValue(e.snd.indices[it]) }
            val sv = DoubleArray(e.snd.size) { e.snd.values[it].toDouble() }
            Row(emb, meta, hp, hv, sp, sv, e.label.toDouble(), weights[i])
        }
        val problem = Problem(base.bias, rows, theta0, lambda, frozen, de, dm, sOff)

        val theta = theta0.copyOf()
        val g = DoubleArray(d)
        var f = problem.eval(theta, g)
        val sList = ArrayDeque<DoubleArray>()
        val yList = ArrayDeque<DoubleArray>()
        val rhoList = ArrayDeque<Double>()
        val dir = DoubleArray(d)
        val alpha = DoubleArray(memory)
        val thetaNew = DoubleArray(d)
        val gNew = DoubleArray(d)
        var iter = 0
        while (iter < maxIterations && !cancelled()) {
            if (infNorm(g) <= gradTolerance) break
            // two-loop recursion: dir = -H·g
            for (i in 0 until d) dir[i] = -g[i]
            for (m in sList.indices.reversed()) {
                alpha[m] = rhoList[m] * dot(sList[m], dir)
                axpy(-alpha[m], yList[m], dir)
            }
            if (sList.isNotEmpty()) {
                val gamma = dot(sList.last(), yList.last()) / dot(yList.last(), yList.last())
                for (i in 0 until d) dir[i] *= gamma
            } else {
                val gn = sqrt(dot(g, g))
                if (gn > 0) for (i in 0 until d) dir[i] /= max(1.0, gn)
            }
            for (m in sList.indices) {
                val beta = rhoList[m] * dot(yList[m], dir)
                axpy(alpha[m] - beta, sList[m], dir)
            }
            var slope = dot(g, dir)
            if (slope >= 0) { // not a descent direction: reset to steepest descent
                sList.clear(); yList.clear(); rhoList.clear()
                for (i in 0 until d) dir[i] = -g[i]
                slope = dot(g, dir)
            }
            var step = 1.0
            var fNew: Double
            var tries = 0
            while (true) {
                for (i in 0 until d) thetaNew[i] = theta[i] + step * dir[i]
                fNew = problem.eval(thetaNew, gNew)
                if (fNew <= f + 1e-4 * step * slope || tries >= 40) break
                step *= 0.5
                tries++
            }
            if (fNew > f) break // no progress possible at this precision
            val s = DoubleArray(d) { thetaNew[it] - theta[it] }
            val y = DoubleArray(d) { gNew[it] - g[it] }
            val sy = dot(s, y)
            if (sy > 1e-12) {
                if (sList.size == memory) { sList.removeFirst(); yList.removeFirst(); rhoList.removeFirst() }
                sList.addLast(s); yList.addLast(y); rhoList.addLast(1.0 / sy)
            }
            val decrease = f - fNew
            thetaNew.copyInto(theta)
            gNew.copyInto(g)
            f = fNew
            iter++
            if (decrease <= 1e-15 * max(1.0, abs(f))) break // stagnated at double precision
        }
        val wEmbed = FloatArray(de) { theta[it].toFloat() }
        val wMeta = FloatArray(dm) { theta[de + it].toFloat() }
        val wHash = base.wHash.copyOf()
        for ((j, k) in active.withIndex()) wHash[k] = theta[hOff + j].toFloat()
        val wSnd = if (sndActive.isEmpty()) base.wSnd else base.wSnd.copyOf()
        for ((j, k) in sndActive.withIndex()) wSnd[k] = theta[sOff + j].toFloat()
        return Result(base.copyWith(wEmbed = wEmbed, wMeta = wMeta, wHash = wHash, wSnd = wSnd), iter, f, infNorm(g))
    }

    private class Row(
        val emb: DoubleArray?, val meta: DoubleArray, val hp: IntArray, val hv: DoubleArray, val sp: IntArray,
        val sv: DoubleArray, val y: Double, val c: Double,
    )

    private class Problem(
        val bias: Double, val rows: List<Row>, val theta0: DoubleArray, val lambda: DoubleArray, val frozen: BooleanArray,
        val de: Int, val dm: Int, val sOff: Int,
    ) {
        /** Returns the objective and writes the gradient into [g]. */
        fun eval(theta: DoubleArray, g: DoubleArray): Double {
            var loss = 0.0
            for (i in g.indices) {
                val diff = theta[i] - theta0[i]
                loss += 0.5 * lambda[i] * diff * diff
                g[i] = lambda[i] * diff
            }
            val hOff = de + dm
            for (r in rows) {
                var z = bias
                if (r.emb != null) for (i in 0 until de) z += theta[i] * r.emb[i]
                for (j in 0 until dm) z += theta[de + j] * r.meta[j]
                for (k in r.hp.indices) z += theta[hOff + r.hp[k]] * r.hv[k]
                for (k in r.sp.indices) z += theta[sOff + r.sp[k]] * r.sv[k]
                loss += r.c * (softplus(z) - r.y * z)
                val res = r.c * (LinearHead.sigmoid(z) - r.y)
                if (r.emb != null) for (i in 0 until de) g[i] += res * r.emb[i]
                for (j in 0 until dm) g[de + j] += res * r.meta[j]
                for (k in r.hp.indices) g[hOff + r.hp[k]] += res * r.hv[k]
                for (k in r.sp.indices) g[sOff + r.sp[k]] += res * r.sv[k]
            }
            for (i in g.indices) if (frozen[i]) g[i] = 0.0
            return loss
        }

        private fun softplus(z: Double) = max(z, 0.0) + ln(1.0 + exp(-abs(z)))
    }

    private companion object {
        fun dot(a: DoubleArray, b: DoubleArray): Double {
            var s = 0.0
            for (i in a.indices) s += a[i] * b[i]
            return s
        }

        fun axpy(a: Double, x: DoubleArray, y: DoubleArray) {
            for (i in x.indices) y[i] += a * x[i]
        }

        fun infNorm(a: DoubleArray): Double {
            var m = 0.0
            for (v in a) m = max(m, abs(v))
            return m
        }
    }
}
