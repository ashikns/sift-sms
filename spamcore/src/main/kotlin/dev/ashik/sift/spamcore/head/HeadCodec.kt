package dev.ashik.sift.spamcore.head

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32

/** SPMH v1 container (ml/SPEC.md); mirrors ml/src/smsml/bundle.py. */
object HeadCodec {
    private const val FORMAT = 1
    private const val ENTRY = 32
    private val MAGIC = byteArrayOf('S'.code.toByte(), 'P'.code.toByte(), 'M'.code.toByte(), 'H'.code.toByte())

    sealed interface Block {
        class F32(val data: FloatArray) : Block
        class I32(val data: IntArray) : Block
        class U8(val data: ByteArray) : Block
        class F64(val data: DoubleArray) : Block
    }

    class FormatException(message: String) : Exception(message)

    private fun align(n: Int) = (n + 15) / 16 * 16

    fun encode(blocks: Map<String, Block>): ByteArray {
        val names = blocks.keys.toList()
        var offset = align(8 + ENTRY * names.size)
        val offsets = IntArray(names.size)
        val sizes = IntArray(names.size)
        names.forEachIndexed { i, name ->
            offsets[i] = offset
            sizes[i] = byteSize(blocks.getValue(name))
            offset = align(offset + sizes[i])
        }
        val buf = ByteBuffer.allocate(offset + 4).order(ByteOrder.LITTLE_ENDIAN)
        buf.put(MAGIC).putShort(FORMAT.toShort()).putShort(names.size.toShort())
        names.forEachIndexed { i, name ->
            val bytes = name.toByteArray(Charsets.US_ASCII)
            require(bytes.size <= 16) { "block name too long: $name" }
            buf.put(bytes.copyOf(16))
            val b = blocks.getValue(name)
            buf.putInt(code(b)).putInt(count(b)).putLong(offsets[i].toLong())
        }
        names.forEachIndexed { i, name ->
            buf.position(offsets[i])
            when (val b = blocks.getValue(name)) {
                is Block.F32 -> b.data.forEach { buf.putFloat(it) }
                is Block.I32 -> b.data.forEach { buf.putInt(it) }
                is Block.U8 -> buf.put(b.data)
                is Block.F64 -> b.data.forEach { buf.putDouble(it) }
            }
        }
        val crc = CRC32().apply { update(buf.array(), 0, offset) }.value
        buf.position(offset)
        buf.putInt(crc.toInt())
        return buf.array()
    }

    fun decode(data: ByteArray): Map<String, Block> {
        if (data.size < 12 || !data.copyOfRange(0, 4).contentEquals(MAGIC)) throw FormatException("not an SPMH file")
        val buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        val stored = buf.getInt(data.size - 4).toLong() and 0xffffffffL
        val crc = CRC32().apply { update(data, 0, data.size - 4) }.value
        if (crc != stored) throw FormatException("SPMH CRC mismatch")
        val format = buf.getShort(4).toInt()
        if (format != FORMAT) throw FormatException("unsupported SPMH format $format")
        val n = buf.getShort(6).toInt() and 0xffff
        val out = LinkedHashMap<String, Block>()
        for (i in 0 until n) {
            val base = 8 + i * ENTRY
            val name = String(data, base, 16, Charsets.US_ASCII).trimEnd('\u0000')
            val code = buf.getInt(base + 16)
            val count = buf.getInt(base + 20)
            val off = buf.getLong(base + 24).toInt()
            val view = ByteBuffer.wrap(data, off, data.size - off).order(ByteOrder.LITTLE_ENDIAN)
            out[name] = when (code) {
                1 -> Block.F32(FloatArray(count).also { view.asFloatBuffer().get(it) })
                2 -> Block.I32(IntArray(count).also { view.asIntBuffer().get(it) })
                3 -> Block.U8(data.copyOfRange(off, off + count))
                4 -> Block.F64(DoubleArray(count).also { view.asDoubleBuffer().get(it) })
                else -> throw FormatException("unknown dtype $code for $name")
            }
        }
        return out
    }

    /** Writes via a temp file, fsync, then an atomic rename. */
    fun writeAtomic(file: File, blocks: Map<String, Block>) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        FileOutputStream(tmp).use { out ->
            out.write(encode(blocks))
            out.fd.sync()
        }
        if (!tmp.renameTo(file)) {
            file.delete()
            check(tmp.renameTo(file)) { "could not move $tmp to $file" }
        }
    }

    private fun code(b: Block) = when (b) {
        is Block.F32 -> 1
        is Block.I32 -> 2
        is Block.U8 -> 3
        is Block.F64 -> 4
    }

    private fun count(b: Block) = when (b) {
        is Block.F32 -> b.data.size
        is Block.I32 -> b.data.size
        is Block.U8 -> b.data.size
        is Block.F64 -> b.data.size
    }

    private fun byteSize(b: Block) = when (b) {
        is Block.F32 -> b.data.size * 4
        is Block.I32 -> b.data.size * 4
        is Block.U8 -> b.data.size
        is Block.F64 -> b.data.size * 8
    }
}
