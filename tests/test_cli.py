from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from main import cli


@pytest.fixture
def runner():
    return CliRunner()


def _mock_index(search_results=None):
    idx = MagicMock()
    idx.chunks = [MagicMock()] * 30
    from portfolio_rag.bm25_index import Chunk

    chunk = Chunk(
        chunk_id=0,
        project="GhostTrace",
        filename="ghosttrace/CLAUDE.md",
        section="## Architecture decisions",
        text="rapidfuzz for fuzzy entity matching and OFAC SDN screening.",
    )
    idx.search.return_value = [(chunk, 3.14)]
    return idx


def test_query_basic(runner):
    with patch("main.build_index", return_value=_mock_index()), \
         patch("main.answer_query", return_value="GhostTrace uses rapidfuzz. [GhostTrace — ghosttrace/CLAUDE.md]"):
        result = runner.invoke(cli, ["query", "which project uses rapidfuzz"])
    assert result.exit_code == 0
    assert "GhostTrace" in result.output


def test_query_show_sources(runner):
    with patch("main.build_index", return_value=_mock_index()), \
         patch("main.answer_query", return_value="Answer with citation."):
        result = runner.invoke(cli, ["query", "fuzzy matching", "--show-sources"])
    assert result.exit_code == 0
    assert "GhostTrace" in result.output


def test_query_no_results(runner):
    idx = MagicMock()
    idx.chunks = [MagicMock()] * 5
    idx.search.return_value = []
    with patch("main.build_index", return_value=idx):
        result = runner.invoke(cli, ["query", "xyzzy not in portfolio"])
    assert result.exit_code != 0 or "No matching" in result.output


def test_query_top_k_passed_to_search(runner):
    idx = _mock_index()
    with patch("main.build_index", return_value=idx), \
         patch("main.answer_query", return_value="Answer."):
        runner.invoke(cli, ["query", "agentic loop", "--top-k", "8"])
    idx.search.assert_called_once_with("agentic loop", top_k=8)


def test_transcript_command(runner, tmp_path):
    transcript_file = tmp_path / "meeting.txt"
    transcript_file.write_text(
        "We decided to use BM25 over FAISS. Action item: Jak to write tests by Friday.",
        encoding="utf-8",
    )
    with patch("main.extract_transcript_decisions", return_value="## Decision Points\n\n**Decision: Use BM25**\n"):
        result = runner.invoke(cli, ["transcript", str(transcript_file)])
    assert result.exit_code == 0
    assert "Decision" in result.output


def test_transcript_missing_file(runner):
    result = runner.invoke(cli, ["transcript", "nonexistent_file.txt"])
    assert result.exit_code != 0


def test_chat_exits_on_quit(runner):
    idx = _mock_index()
    with patch("main.build_index", return_value=idx), \
         patch("main.answer_query", return_value="Answer."):
        result = runner.invoke(cli, ["chat"], input="quit\n")
    assert result.exit_code == 0
    assert "Goodbye" in result.output


def test_chat_answers_question(runner):
    idx = _mock_index()
    with patch("main.build_index", return_value=idx), \
         patch("main.answer_query", return_value="The answer is GhostTrace. [GhostTrace — ghosttrace/CLAUDE.md]"):
        result = runner.invoke(cli, ["chat"], input="which project uses rapidfuzz\nquit\n")
    assert result.exit_code == 0
    assert "GhostTrace" in result.output


def test_chat_skips_empty_input(runner):
    idx = _mock_index()
    with patch("main.build_index", return_value=idx), \
         patch("main.answer_query", return_value="Answer.") as mock_answer:
        result = runner.invoke(cli, ["chat"], input="\n\nquit\n")
    assert result.exit_code == 0
    mock_answer.assert_not_called()
