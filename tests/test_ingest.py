import textwrap
from pathlib import Path

import pytest

from portfolio_rag.ingest import _split_sections, chunks_from_file, build_index


def test_split_sections_basic():
    md = textwrap.dedent("""\
        ## Overview
        This is the overview text.

        ## Architecture
        This is architecture details.
    """)
    sections = _split_sections(md)
    assert len(sections) == 2
    assert sections[0][0] == "## Overview"
    assert "overview text" in sections[0][1]
    assert sections[1][0] == "## Architecture"


def test_split_sections_with_intro():
    md = textwrap.dedent("""\
        Some intro text before any header.

        ## Section One
        Body of section one.
    """)
    sections = _split_sections(md)
    assert sections[0][0] == "(intro)"
    assert "intro text" in sections[0][1]
    assert sections[1][0] == "## Section One"


def test_split_sections_skips_tiny_sections():
    md = "## Tiny\nab\n## Real Section\nThis has enough content to be a valid chunk."
    sections = _split_sections(md)
    headers = [h for h, _ in sections]
    assert "## Real Section" in headers
    assert "## Tiny" not in headers


def test_split_sections_handles_triple_hash():
    md = textwrap.dedent("""\
        ## Top Level
        Some content.

        ### Sub Section
        Sub content here with enough text.
    """)
    sections = _split_sections(md)
    assert any("### Sub Section" in h for h, _ in sections)


def test_chunks_from_file(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text(textwrap.dedent("""\
        ## What It Does
        Monitors SAM.gov solicitations and extracts CMMC requirements.

        ## Tech Stack
        FastAPI, Claude Haiku, SQLite, Jinja2.
    """), encoding="utf-8")

    chunks = chunks_from_file(readme, "SAM Agent", "sam_agent/README.md")
    assert len(chunks) == 2
    assert chunks[0].project == "SAM Agent"
    assert chunks[0].filename == "sam_agent/README.md"
    assert "SAM.gov" in chunks[0].text
    assert chunks[1].section == "## Tech Stack"


def test_chunks_from_file_sets_section():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("## GhostTrace Overview\nEDGAR extraction and entity resolution via rapidfuzz.\n")
        name = f.name
    path = Path(name)
    chunks = chunks_from_file(path, "GhostTrace", "ghosttrace/README.md")
    assert chunks[0].section == "## GhostTrace Overview"
    path.unlink()


def test_build_index_returns_chunks():
    idx = build_index()
    assert len(idx.chunks) > 0
    # Portfolio CLAUDE.md alone has many sections
    assert len(idx.chunks) > 20


def test_build_index_covers_multiple_projects():
    idx = build_index()
    projects = {c.project for c in idx.chunks}
    # Should cover well-known projects
    for expected in ("Portfolio", "GhostTrace", "CFIUS Screener"):
        assert expected in projects, f"{expected} not found in indexed projects: {projects}"


def test_build_index_is_built():
    idx = build_index()
    assert idx._built


def test_build_index_search_works():
    idx = build_index()
    results = idx.search("fuzzy entity matching rapidfuzz", top_k=3)
    assert len(results) > 0
    top_chunk, top_score = results[0]
    assert top_score > 0
    # GhostTrace is the canonical rapidfuzz project
    top_projects = [c.project for c, _ in results]
    assert any(p in ("GhostTrace", "Portfolio") for p in top_projects)


def test_build_index_orbital_sentinel_query():
    idx = build_index()
    results = idx.search("Orbital Sentinel sgp4 conjunction Pc", top_k=3)
    assert len(results) > 0
    top_projects = [c.project for c, _ in results]
    assert any(p in ("Orbital Sentinel", "Portfolio") for p in top_projects)


def test_build_index_no_node_modules():
    idx = build_index()
    filenames = [c.filename for c in idx.chunks]
    assert not any("node_modules" in fn for fn in filenames)


def test_split_sections_empty_string():
    sections = _split_sections("")
    assert sections == []


def test_split_sections_no_headers():
    md = "Just plain text without any markdown headers at all."
    sections = _split_sections(md)
    # Goes into (intro) only if len >= 50
    assert len(sections) == 1
    assert sections[0][0] == "(intro)"
