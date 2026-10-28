package dev.ashik.sift.spamcore.normalize

import java.io.InputStream

/** The resource lists norm_v1 / feat_v3 depend on (ml/SPEC.md). Same files and parsing as ml/src/smsml/resources.py. */
class Resources(
    val gsm7: Set<Int>,
    val confusables: Map<Int, String>,
    val tlds: Set<String>,
    val tldStoplist: Set<String>,
    val psl: Set<String>,
    val shorteners: Set<String>,
    val riskyTlds: Set<String>,
    val brands: List<Brand>,
) {
    /** A brand keyword, its official registrable domains, and whether it matches as a host substring. */
    data class Brand(val keyword: String, val officials: Set<String>, val substring: Boolean)

    /** Every brand's official registrable domains. */
    val officialDomains: Set<String> = brands.flatMapTo(HashSet()) { it.officials }

    companion object {
        val FILES = listOf(
            "gsm7_v1.txt", "confusables_v1.tsv", "tlds_v1.txt", "tld_stoplist_v1.txt", "psl_lite_v1.txt",
            "shorteners_v1.txt", "risky_tlds_v1.txt", "brands_v1.tsv",
        )

        /** Non-empty lines with `#` comments and surrounding whitespace removed. */
        fun readLines(stream: InputStream): List<String> =
            stream.bufferedReader(Charsets.UTF_8).useLines { lines ->
                lines.map { it.substringBefore('#').trim() }.filter { it.isNotEmpty() }.toList()
            }

        /** [open] returns the named file's stream, or null if it doesn't exist (treated as an empty list). */
        fun load(open: (String) -> InputStream?): Resources {
            fun lines(name: String): List<String> = open(name)?.use { readLines(it) } ?: emptyList()
            return Resources(
                gsm7 = lines("gsm7_v1.txt").map { it.toInt(16) }.toHashSet(),
                confusables = lines("confusables_v1.tsv").associate { line ->
                    val (src, dst) = line.split('\t')
                    src.toInt(16) to dst.trim()
                },
                tlds = lines("tlds_v1.txt").toHashSet(),
                tldStoplist = lines("tld_stoplist_v1.txt").toHashSet(),
                psl = lines("psl_lite_v1.txt").toHashSet(),
                shorteners = lines("shorteners_v1.txt").toHashSet(),
                riskyTlds = lines("risky_tlds_v1.txt").toHashSet(),
                brands = lines("brands_v1.tsv").map { line ->
                    val cols = line.split('\t')
                    val keyword = cols[0].trim()
                    val mode = cols.getOrNull(2)?.trim() ?: if (keyword.length >= 5) "sub" else "word"
                    require(mode == "sub" || mode == "word") { "bad brand match mode $mode for $keyword" }
                    Brand(keyword, cols[1].split(',').map { it.trim() }.filter { it.isNotEmpty() }.toHashSet(), mode == "sub")
                },
            )
        }
    }
}
