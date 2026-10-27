package dev.ashik.sift.spamcore.tokenize

import dev.ashik.sift.spamcore.normalize.TextNormalizer
import java.io.InputStream

/** tok_v1 (ml/SPEC.md): BERT-uncased WordPiece over norm_v1's `enc`. Mirrors ml/src/smsml/tokenizer.py. */
class WordPieceTokenizer(private val vocab: Map<String, Int>) {

    fun encode(enc: String): LongArray {
        val pieces = ArrayList<Int>(64)
        for (word in words(enc)) {
            wordpiece(word, pieces)
            if (pieces.size >= MAX_PIECES) break
        }
        val n = minOf(pieces.size, MAX_PIECES)
        val ids = LongArray(n + 2)
        ids[0] = CLS.toLong()
        for (i in 0 until n) ids[i + 1] = pieces[i].toLong()
        ids[n + 1] = SEP.toLong()
        return ids
    }

    private fun wordpiece(word: String, out: MutableList<Int>) {
        if (word.length > MAX_WORD_CPS) {
            out.add(UNK)
            return
        }
        val start0 = out.size
        var start = 0
        while (start < word.length) {
            var end = word.length
            var found = -1
            while (start < end) {
                val sub = if (start == 0) word.substring(0, end) else "##" + word.substring(start, end)
                val id = vocab[sub]
                if (id != null) {
                    found = id
                    break
                }
                end--
            }
            if (found == -1) {
                while (out.size > start0) out.removeAt(out.size - 1)
                out.add(UNK)
                return
            }
            out.add(found)
            start = end
        }
    }

    companion object {
        const val CLS = 101
        const val SEP = 102
        const val UNK = 100
        const val MAX_PIECES = 126
        const val MAX_WORD_CPS = 100

        fun isAsciiPunct(c: Char): Boolean {
            val o = c.code
            return o in 33..47 || o in 58..64 || o in 91..96 || o in 123..126
        }

        fun words(enc: String): List<String> {
            val out = ArrayList<String>()
            for (w in TextNormalizer.asciiLower(enc).split(' ')) {
                if (w.isEmpty()) continue
                val cur = StringBuilder()
                for (c in w) {
                    if (isAsciiPunct(c)) {
                        if (cur.isNotEmpty()) {
                            out.add(cur.toString())
                            cur.setLength(0)
                        }
                        out.add(c.toString())
                    } else {
                        cur.append(c)
                    }
                }
                if (cur.isNotEmpty()) out.add(cur.toString())
            }
            return out
        }

        /** Line index = id; splits on LF only and drops a trailing CR, like the Python loader. */
        fun load(stream: InputStream): WordPieceTokenizer {
            val text = stream.readBytes().toString(Charsets.UTF_8)
            val lines = text.split('\n').toMutableList()
            if (lines.isNotEmpty() && lines.last().isEmpty()) lines.removeAt(lines.size - 1)
            val vocab = HashMap<String, Int>(lines.size * 2)
            lines.forEachIndexed { i, line ->
                val token = if (line.endsWith('\r')) line.dropLast(1) else line
                vocab.putIfAbsent(token, i)
            }
            return WordPieceTokenizer(vocab)
        }
    }
}
