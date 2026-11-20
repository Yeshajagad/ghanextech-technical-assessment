from rag.chunker import CHUNK_SIZE, chunk_pages, split_text


def test_short_text_is_one_chunk():
    assert split_text("Just one short sentence.") == ["Just one short sentence."]


def test_chunks_respect_size_and_overlap():
    sentences = [f"This is sentence number {i} about topic {i % 5}." for i in range(80)]
    chunks = split_text(" ".join(sentences))

    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    # Consecutive chunks share some text because of the overlap.
    for first, second in zip(chunks, chunks[1:]):
        last_sentence = first.split(". ")[-1]
        assert last_sentence in second


def test_very_long_word_is_hard_cut():
    chunks = split_text("x" * 2000)
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    assert "".join(chunks) == "x" * 2000


def test_chunk_pages_keeps_source_and_page():
    chunks = chunk_pages([(1, "Page one text."), (2, "Page two text.")], "a.pdf")
    assert [(c["source"], c["page"]) for c in chunks] == [("a.pdf", 1), ("a.pdf", 2)]