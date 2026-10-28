package dev.ashik.sift.ui.conversations

import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.SearchableMessage

/**
 * Search text that ignores case, spaces and punctuation: "hdfc salary" finds "HDFC-Salary", and "98765 43210" finds
 * "+919876543210".
 */
object SearchText {
    /** [text] lowercased, with only its letters, digits and combining marks (an Indic vowel sign is part of its word). */
    fun key(text: String): String = fold(text, null)

    /** Where [query] occurs in [text], as ranges of [text] that span the spaces and punctuation inside each match. */
    fun ranges(text: String, query: String?): List<IntRange> {
        val q = key(query.orEmpty())
        if (q.isEmpty()) return emptyList()
        val at = IntArray(text.length)
        val folded = fold(text, at)
        val found = mutableListOf<IntRange>()
        var from = folded.indexOf(q)
        while (from >= 0) {
            found += at[from]..at[from + q.length - 1]
            from = folded.indexOf(q, from + q.length)
        }
        return found
    }

    /**
     * [text] for a two-line snippet that shows its first match of [query]: from the start if the match is near it, else
     * "…" and the first word starting less than [LEAD] chars before the match.
     */
    fun excerpt(text: String, query: String?): String {
        val match = ranges(text, query).firstOrNull()?.first ?: return text
        if (match <= LEAD) return text
        val from = match - LEAD
        val start = (from + 1..match).firstOrNull { !text[it].isWhitespace() && text[it - 1].isWhitespace() } ?: from
        return "…" + text.substring(start)
    }

    private const val LEAD = 30

    /** Variation selectors are marks too, but they only pick how an emoji looks. */
    private val MARKS = setOf(CharCategory.NON_SPACING_MARK, CharCategory.COMBINING_SPACING_MARK)

    private fun fold(text: String, at: IntArray?): String {
        val out = StringBuilder(text.length)
        text.forEachIndexed { i, c ->
            if (c.isLetterOrDigit() || (c.category in MARKS && c !in '︀'..'️')) {
                at?.set(out.length, i)
                out.append(c.lowercaseChar())
            }
        }
        return out.toString()
    }
}

/** A conversation a search found: by its sender ([match] null), or by [match], its newest matching message in the tab. */
data class SearchHit(val row: ConversationRow, val match: SearchableMessage?) {
    /** The date the row shows. */
    val date: Long get() = match?.date ?: row.conversation.latestDate
}

/** Each tab's conversations that match a search. */
data class SearchResults(val inbox: List<SearchHit>, val spam: List<SearchHit>)

/** Every message's [SearchText.key], newest first, to find the tabs' conversations by. */
class SearchIndex(private val messages: List<SearchableMessage>) {
    private val keys = messages.map { SearchText.key(it.body) }

    /**
     * The conversations among each tab's rows that match [query], a non-empty [SearchText.key]: by the sender's name or
     * number, else by a message in that tab. Newest first, by the date each row shows.
     */
    fun search(inbox: List<ConversationRow>, spam: List<ConversationRow>, query: String): SearchResults {
        val newest = HashMap<Pair<Long, Int>, SearchableMessage>()
        messages.forEachIndexed { i, m -> if (query in keys[i]) newest.putIfAbsent(m.threadId to m.folder, m) }
        fun hits(rows: List<ConversationRow>) = rows
            .mapNotNull { row ->
                val c = row.conversation
                if (query in SearchText.key(row.displayName.orEmpty()) || query in SearchText.key(c.address)) SearchHit(row, null)
                else newest[c.threadId to c.folder]?.let { SearchHit(row, it) }
            }
            .sortedWith(compareByDescending<SearchHit> { it.date }.thenByDescending { it.row.conversation.threadId })
        return SearchResults(hits(inbox), hits(spam))
    }
}
