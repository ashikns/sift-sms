package dev.ashik.sift.spamcore.head

import dev.ashik.sift.spamcore.Cue
import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.head.HeadCodec.Block
import kotlin.math.exp

/**
 * score_v1 (ml/SPEC.md); mirrors ml/src/smsml/head.py. Immutable: personalization produces a new instance that
 * shares the (large) arrays it didn't change.
 */
class LinearHead(
    val bias: Double,
    val wEmbed: FloatArray,
    val muEmbed: FloatArray,
    val wMeta: FloatArray,
    val muMeta: FloatArray,
    val wHash: FloatArray,
    val idf: FloatArray,
    val fbBias: Double,
    val fbScale: Double,
    val wMetaFb: FloatArray,
) {
    init {
        require(wEmbed.size == EMBED_DIM && muEmbed.size == EMBED_DIM) { "embedding block size" }
        require(wMeta.size == Features.N_META && muMeta.size == Features.N_META && wMetaFb.size == Features.N_META)
        require(wHash.size == Features.BUCKETS && idf.size == Features.BUCKETS) { "hash block size" }
    }

    /** Logit with the embedding (accumulated in double, SPEC order). A null embedding contributes 0. */
    fun z(embedding: FloatArray?, meta: FloatArray, sparse: SparseVector): Double {
        var z = bias
        if (embedding != null) for (i in 0 until EMBED_DIM) z += wEmbed[i].toDouble() * (embedding[i].toDouble() - muEmbed[i])
        for (j in 0 until Features.N_META) z += wMeta[j].toDouble() * (meta[j].toDouble() - muMeta[j])
        for (k in 0 until sparse.size) z += wHash[sparse.indices[k]].toDouble() * sparse.values[k]
        return z
    }

    /** Lexical-only fallback logit, used when the encoder is unavailable. */
    fun zFallback(meta: FloatArray, sparse: SparseVector): Double {
        var h = 0.0
        for (k in 0 until sparse.size) h += wHash[sparse.indices[k]].toDouble() * sparse.values[k]
        var z = fbBias + fbScale * h
        for (j in 0 until Features.N_META) z += wMetaFb[j].toDouble() * (meta[j].toDouble() - muMeta[j])
        return z
    }

    /**
     * The largest positive contributions behind a score, as readable cues: metadata features by name, the
     * embedding block as a whole ("wording"), and the strongest n-gram words. [featureStrings] maps bucket index
     * to the feature strings that hashed there.
     */
    fun explain(
        embedding: FloatArray?,
        meta: FloatArray,
        sparse: SparseVector,
        featureStrings: Map<Int, List<String>>,
        limit: Int = 4,
    ): List<Cue> {
        val cues = ArrayList<Cue>()
        for (j in 0 until Features.N_META) {
            val c = wMeta[j].toDouble() * (meta[j].toDouble() - muMeta[j])
            val text = META_CUES[Features.META_NAMES[j]]
            if (text != null && c > 0.25) cues += Cue(text, c.toFloat())
        }
        if (embedding != null) {
            var c = 0.0
            for (i in 0 until EMBED_DIM) c += wEmbed[i].toDouble() * (embedding[i].toDouble() - muEmbed[i])
            if (c > 0.5) cues += Cue("Wording resembles known spam", c.toFloat())
        }
        val words = ArrayList<Pair<String, Double>>()
        for (k in 0 until sparse.size) {
            val c = wHash[sparse.indices[k]].toDouble() * sparse.values[k]
            if (c <= 0.15) continue
            val label = featureStrings[sparse.indices[k]]?.firstOrNull { it.startsWith("w:") || it.startsWith("b:") || it.startsWith("u:") }
                ?: continue
            words += label.substring(2) to c
        }
        words.sortedByDescending { it.second }.take(3).forEach { (w, c) -> cues += Cue("Contains “$w”", c.toFloat()) }
        return cues.sortedByDescending { it.weight }.take(limit)
    }

    fun copyWith(wEmbed: FloatArray = this.wEmbed, wMeta: FloatArray = this.wMeta, wHash: FloatArray = this.wHash) =
        LinearHead(bias, wEmbed, muEmbed, wMeta, muMeta, wHash, idf, fbBias, fbScale, wMetaFb)

    fun toBlocks(): LinkedHashMap<String, Block> = linkedMapOf(
        "bias" to Block.F64(doubleArrayOf(bias)),
        "w_embed" to Block.F32(wEmbed),
        "mu_embed" to Block.F32(muEmbed),
        "w_meta" to Block.F32(wMeta),
        "mu_meta" to Block.F32(muMeta),
        "w_hash" to Block.F32(wHash),
        "idf" to Block.F32(idf),
        "fb_bias" to Block.F64(doubleArrayOf(fbBias)),
        "fb_scale" to Block.F64(doubleArrayOf(fbScale)),
        "w_meta_fb" to Block.F32(wMetaFb),
    )

    companion object {
        const val EMBED_DIM = 384

        fun sigmoid(z: Double): Double = if (z >= 0) 1.0 / (1.0 + exp(-z)) else exp(z).let { it / (1.0 + it) }

        fun fromBlocks(b: Map<String, Block>): LinearHead {
            fun f32(name: String) = (b[name] as? Block.F32)?.data ?: throw HeadCodec.FormatException("missing $name")
            fun f64(name: String) = (b[name] as? Block.F64)?.data?.get(0) ?: throw HeadCodec.FormatException("missing $name")
            return LinearHead(
                f64("bias"), f32("w_embed"), f32("mu_embed"), f32("w_meta"), f32("mu_meta"), f32("w_hash"), f32("idf"),
                f64("fb_bias"), f64("fb_scale"), f32("w_meta_fb"),
            )
        }

        private val META_CUES = mapOf(
            "url_count" to "Contains links",
            "url_shortener" to "Shortened link hides the destination",
            "url_ip_host" to "Link points to a raw IP address",
            "url_risky_tld" to "Link uses a domain ending common in scams",
            "url_punycode_or_at" to "Link uses a disguised address",
            "url_brand_lookalike" to "Link imitates a well-known brand",
            "phone_number" to "Asks you to call a number",
            "email_address" to "Contains an email address",
            "currency_amount" to "Mentions money",
            "optout_stop" to "Bulk marketing (“reply STOP”)",
            "upper_ratio" to "Lots of capital letters",
            "excl_count" to "Lots of exclamation marks",
            "invisible" to "Hidden characters in the text",
            "lookalike" to "Look-alike letters from other alphabets",
            "dropped_ratio" to "Unusual symbols",
            "snd_short_code" to "Sent from a short code",
            "snd_alphanumeric" to "Sent from a named sender ID",
            "snd_email" to "Sent from an email address",
            "snd_international" to "Sent from an international number",
            "snd_first_contact" to "First message from this sender",
        )
    }
}
