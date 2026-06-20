import pytest
from portfolio_rag.bm25_index import BM25Index, Chunk


def _make_chunk(chunk_id: int, project: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, project=project, filename="README.md", section="Overview", text=text)


def test_build_empty_index():
    idx = BM25Index()
    idx.build()
    assert idx._built
    assert idx._avgdl == 1.0


def test_search_empty_index_returns_empty():
    idx = BM25Index()
    idx.build()
    assert idx.search("fuzzy matching") == []


def test_search_single_chunk_matches():
    idx = BM25Index()
    idx.add_chunk(_make_chunk(0, "GhostTrace", "Uses rapidfuzz for fuzzy entity matching and OFAC SDN screening."))
    results = idx.search("fuzzy entity matching")
    assert len(results) == 1
    chunk, score = results[0]
    assert chunk.project == "GhostTrace"
    assert score > 0


def test_search_ranks_more_relevant_first():
    idx = BM25Index()
    idx.add_chunk(_make_chunk(0, "ProjectA", "Uses rapidfuzz for entity matching with token sort ratio."))
    idx.add_chunk(_make_chunk(1, "ProjectB", "A financial monitoring tool with Monte Carlo simulation."))
    idx.add_chunk(_make_chunk(2, "ProjectC", "Entity matching and fuzzy string comparison using rapidfuzz library."))
    results = idx.search("fuzzy entity matching rapidfuzz")
    assert len(results) >= 2
    # ProjectC and ProjectA both match; ProjectB should score lower
    top_projects = [c.project for c, _ in results[:2]]
    assert "ProjectB" not in top_projects


def test_search_returns_only_positive_scores():
    idx = BM25Index()
    idx.add_chunk(_make_chunk(0, "Alpha", "BM25 retrieval over orbital mechanics and sgp4."))
    idx.add_chunk(_make_chunk(1, "Beta", "OFAC SDN screening with rapidfuzz."))
    results = idx.search("completely unrelated xyzzy")
    assert all(score > 0 for _, score in results) or results == []


def test_search_top_k_limits_results():
    idx = BM25Index()
    for i in range(10):
        idx.add_chunk(_make_chunk(i, f"Project{i}", f"Claude Haiku project {i} with FastAPI and SQLite."))
    results = idx.search("Claude FastAPI", top_k=3)
    assert len(results) <= 3


def test_tokenization_is_case_insensitive():
    idx = BM25Index()
    idx.add_chunk(_make_chunk(0, "P1", "CFIUS Part 800 TVC Risk Scoring Engine."))
    results_upper = idx.search("CFIUS TVC")
    results_lower = idx.search("cfius tvc")
    assert len(results_upper) == len(results_lower)
    if results_upper and results_lower:
        assert abs(results_upper[0][1] - results_lower[0][1]) < 1e-9


def test_project_name_included_in_scoring():
    idx = BM25Index()
    idx.add_chunk(_make_chunk(0, "GhostTrace", "EDGAR beneficial ownership extraction system."))
    idx.add_chunk(_make_chunk(1, "SENTINEL", "RSS feed monitoring for influence operations."))
    # GhostTrace should score higher on a GhostTrace query
    results = idx.search("GhostTrace EDGAR")
    assert results[0][0].project == "GhostTrace"


def test_chunk_ids_preserved():
    idx = BM25Index()
    c = _make_chunk(42, "TestProject", "Some content about orbital mechanics.")
    idx.add_chunk(c)
    results = idx.search("orbital mechanics")
    assert results[0][0].chunk_id == 42


def test_idf_penalizes_common_terms():
    idx = BM25Index()
    for i in range(10):
        idx.add_chunk(_make_chunk(i, f"P{i}", f"FastAPI project {i} with Claude Haiku."))
    idx.add_chunk(_make_chunk(10, "Unique", "sgp4 orbital propagation for conjunction analysis."))
    results = idx.search("sgp4 orbital")
    # Unique project should rank first since those terms appear in only one doc
    assert results[0][0].project == "Unique"


def test_avgdl_computed_correctly():
    idx = BM25Index()
    idx.add_chunk(_make_chunk(0, "A", "one two three"))
    idx.add_chunk(_make_chunk(1, "B", "alpha beta gamma delta epsilon"))
    idx.build()
    # combined per chunk = "{project} {section} {text}", section="Overview" for both
    # chunk 0: "A Overview one two three" = 5 tokens
    # chunk 1: "B Overview alpha beta gamma delta epsilon" = 7 tokens
    # avgdl = (5 + 7) / 2 = 6.0
    assert idx._avgdl == pytest.approx(6.0)


def test_rebuild_after_add():
    idx = BM25Index()
    idx.add_chunk(_make_chunk(0, "Alpha", "quantum cryptography migration NIST FIPS."))
    idx.build()
    assert idx._built
    idx.add_chunk(_make_chunk(1, "Beta", "BM25 retrieval engine for portfolio search."))
    # _built is False after new add
    assert not idx._built
    results = idx.search("portfolio BM25")
    assert idx._built
    assert results[0][0].project == "Beta"
