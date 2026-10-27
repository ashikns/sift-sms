package dev.ashik.sift.spamcore.model

import dev.ashik.sift.spamcore.combine.Thresholds
import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.head.HeadCodec
import dev.ashik.sift.spamcore.head.LinearHead
import dev.ashik.sift.spamcore.normalize.Resources
import dev.ashik.sift.spamcore.normalize.TextNormalizer
import dev.ashik.sift.spamcore.tokenize.WordPieceTokenizer
import dev.ashik.sift.spamcore.train.Priors
import dev.ashik.sift.spamcore.train.Weighting
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

/** Where bundle files come from: app assets on Android, a directory on the desktop. */
interface BundleSource {
    /** @throws FileNotFoundException if [name] doesn't exist. */
    fun open(name: String): InputStream

    /** A read-only direct mapping of [name] for zero-copy model loading, or null to read the bytes instead. */
    fun map(name: String): ByteBuffer? = null
}

class DirectoryBundleSource(private val dir: File) : BundleSource {
    override fun open(name: String): InputStream {
        val f = File(dir, name)
        if (!f.isFile) throw FileNotFoundException(f.path)
        return f.inputStream()
    }

    override fun map(name: String): ByteBuffer? = RandomAccessFile(File(dir, name), "r").use { raf ->
        raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
    }
}

@Serializable
data class ModelConfig(
    val schema: Int,
    @SerialName("bundle_version") val bundleVersion: String,
    val encoder: EncoderConfig,
    val tokenizer: FileRef,
    val normalizer: NormalizerConfig,
    val features: FeatureConfig,
    val head: FileRef,
    val thresholds: ThresholdConfig,
    val personalization: PersonalizationConfig,
    val knn: KnnConfig,
)

@Serializable
data class FileRef(val file: String, val sha256: String)

@Serializable
data class EncoderConfig(
    val file: String,
    val sha256: String,
    val inputs: List<String>,
    val output: String,
)

@Serializable
data class NormalizerConfig(val version: String, val resources: Map<String, String>)

@Serializable
data class FeatureConfig(
    val version: String,
    @SerialName("hash_bits") val hashBits: Int,
    @SerialName("hash_seed") val hashSeed: Long,
    @SerialName("meta_names") val metaNames: List<String>,
)

@Serializable
data class ThresholdSet(val conservative: Double, val balanced: Double, val aggressive: Double, val warn: Double) {
    fun toThresholds() = Thresholds(conservative, balanced, aggressive, warn)
}

@Serializable
data class ThresholdConfig(val full: ThresholdSet, val fallback: ThresholdSet)

@Serializable
data class PersonalizationConfig(
    val priors: PriorsConfig,
    val weights: WeightsConfig,
    @SerialName("debounce_ms") val debounceMs: Long,
)

@Serializable
data class PriorsConfig(val embed: Double, val meta: Double, val hash: Double, val snd: Double) {
    fun toPriors() = Priors(embed, meta, hash, snd)
}

@Serializable
data class WeightsConfig(
    val spam: Double,
    val ham: Double,
    @SerialName("cluster_cap") val clusterCap: Double,
    @SerialName("cluster_cos") val clusterCos: Double,
    @SerialName("max_examples") val maxExamples: Int,
) {
    fun toWeighting() = Weighting(spam, ham, clusterCap, clusterCos, maxExamples)
}

@Serializable
data class KnnConfig(@SerialName("tau_spam") val tauSpam: Double, @SerialName("tau_ham") val tauHam: Double)

/** A loaded, verified model bundle (ml/scripts/s11_package.py writes it). */
class ModelBundle(
    val config: ModelConfig,
    val source: BundleSource,
    val resources: Resources,
    val normalizer: TextNormalizer,
    val tokenizer: WordPieceTokenizer,
    val baseHead: LinearHead,
) {
    companion object {
        const val SCHEMA = 1
        const val RESOURCE_DIR = "resources"
        val JSON = Json { ignoreUnknownKeys = true }

        class BundleException(message: String) : Exception(message)

        /**
         * Loads and verifies every small file's SHA-256; the encoder is only hashed when [verifyEncoder]. Checksums are
         * skipped when the bundle's version equals [alreadyVerified] (it was verified on an earlier start).
         */
        fun load(source: BundleSource, verifyEncoder: Boolean = false, alreadyVerified: String? = null): ModelBundle {
            val config = source.open("model.json").use { JSON.decodeFromString(ModelConfig.serializer(), it.readBytes().decodeToString()) }
            val verify = config.bundleVersion != alreadyVerified
            if (config.schema != SCHEMA) throw BundleException("unsupported bundle schema ${config.schema}")
            if (config.features.version != Features.VERSION || config.normalizer.version != "norm_v1") {
                throw BundleException("unsupported feature/normalizer version ${config.features.version}/${config.normalizer.version}")
            }
            if (config.features.hashBits != Features.BITS || config.features.hashSeed != Features.SEED.toLong() ||
                config.features.metaNames != Features.META_NAMES
            ) {
                throw BundleException("feature spec mismatch")
            }
            fun bytes(name: String, sha: String): ByteArray {
                val data = source.open(name).use { it.readBytes() }
                if (verify && sha256(data) != sha) throw BundleException("checksum mismatch for $name")
                return data
            }
            val resourceBytes = config.normalizer.resources.mapValues { (name, sha) -> bytes("$RESOURCE_DIR/$name", sha) }
            val resources = Resources.load { name -> resourceBytes[name]?.inputStream() }
            val tokenizer = WordPieceTokenizer.load(bytes(config.tokenizer.file, config.tokenizer.sha256).inputStream())
            val head = LinearHead.fromBlocks(HeadCodec.decode(bytes(config.head.file, config.head.sha256)))
            if (verifyEncoder) bytes(config.encoder.file, config.encoder.sha256)
            return ModelBundle(config, source, resources, TextNormalizer(resources), tokenizer, head)
        }

        fun sha256(data: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }
    }
}
