package dev.ashik.sift.spamcore.features

/** MurmurHash3 x86_32 (ml/src/smsml/hashing.py). Returns the hash as an unsigned value in a Long. */
object Murmur3 {
    fun hash32(data: ByteArray, seed: Int): Long {
        val c1 = 0xcc9e2d51.toInt()
        val c2 = 0x1b873593
        var h = seed
        val nblocks = data.size / 4
        for (i in 0 until nblocks) {
            val o = i * 4
            var k = (data[o].toInt() and 0xff) or ((data[o + 1].toInt() and 0xff) shl 8) or
                ((data[o + 2].toInt() and 0xff) shl 16) or ((data[o + 3].toInt() and 0xff) shl 24)
            k *= c1
            k = Integer.rotateLeft(k, 15)
            k *= c2
            h = h xor k
            h = Integer.rotateLeft(h, 13)
            h = h * 5 + 0xe6546b64.toInt()
        }
        val tail = nblocks * 4
        var k = 0
        val rem = data.size and 3
        if (rem == 3) k = k xor ((data[tail + 2].toInt() and 0xff) shl 16)
        if (rem >= 2) k = k xor ((data[tail + 1].toInt() and 0xff) shl 8)
        if (rem >= 1) {
            k = k xor (data[tail].toInt() and 0xff)
            k *= c1
            k = Integer.rotateLeft(k, 15)
            k *= c2
            h = h xor k
        }
        h = h xor data.size
        h = h xor (h ushr 16)
        h *= 0x85ebca6b.toInt()
        h = h xor (h ushr 13)
        h *= 0xc2b2ae35.toInt()
        h = h xor (h ushr 16)
        return h.toLong() and 0xffffffffL
    }
}
