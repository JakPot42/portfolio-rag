import os
import sys
from pathlib import Path

import click


def _load_dotenv() -> None:
    """Load a .env file from the repo root if it exists (no external dep needed)."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from portfolio_rag.claude_chat import answer_query, extract_transcript_decisions
from portfolio_rag.ingest import build_index

console = Console(legacy_windows=False)


@click.group()
def cli() -> None:
    """Portfolio RAG — Chat With Your Own Work.

    Retrieves from every README and CLAUDE.md in the portfolio,
    then asks Claude to answer using only that context with mandatory citations.
    """


@cli.command()
@click.argument("question")
@click.option("--top-k", default=5, show_default=True, help="Chunks to retrieve.")
@click.option("--show-sources", is_flag=True, help="Print retrieved chunks before the answer.")
def query(question: str, top_k: int, show_sources: bool) -> None:
    """Single-shot question against the portfolio corpus.

    \b
    Examples:
      python main.py query "which project handles fuzzy entity matching"
      python main.py query "what was the Pc threshold in Orbital Sentinel"
      python main.py query "summarize every project using a bounded agentic loop"
    """
    with console.status("[bold green]Indexing portfolio...[/bold green]"):
        idx = build_index()

    console.print(
        f"[dim]Index built: {len(idx.chunks)} chunks across the portfolio.[/dim]"
    )

    results = idx.search(question, top_k=top_k)

    if not results:
        console.print("[yellow]No matching chunks found in the portfolio.[/yellow]")
        sys.exit(1)

    if show_sources:
        table = Table(title="Retrieved Chunks", show_lines=True)
        table.add_column("Score", style="dim", width=6)
        table.add_column("Project", style="cyan")
        table.add_column("File", style="dim")
        table.add_column("Section")
        for chunk, score in results:
            table.add_row(
                f"{score:.2f}",
                chunk.project,
                chunk.filename,
                chunk.section[:70],
            )
        console.print(table)
    else:
        source_list = ", ".join(
            f"[cyan]{c.project}[/cyan]" for c, _ in results
        )
        console.print(f"[dim]Searching: {source_list}[/dim]")

    with console.status("[bold green]Asking Claude...[/bold green]"):
        chunks = [c for c, _ in results]
        answer = answer_query(question, chunks)

    console.print()
    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))


@cli.command()
@click.option("--top-k", default=5, show_default=True, help="Chunks to retrieve per query.")
def chat(top_k: int) -> None:
    """Interactive chat loop against the portfolio corpus."""
    with console.status("[bold green]Indexing portfolio...[/bold green]"):
        idx = build_index()

    console.print(
        Panel(
            f"[bold]Portfolio RAG — Interactive Mode[/bold]\n"
            f"[dim]{len(idx.chunks)} chunks indexed.[/dim] "
            f"Type [cyan]exit[/cyan] to quit.",
            border_style="blue",
        )
    )

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if question.lower() in {"exit", "quit", "q", ""}:
            if question.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Goodbye.[/dim]")
                break
            continue

        results = idx.search(question, top_k=top_k)

        if not results:
            console.print("[yellow]No matching chunks found.[/yellow]")
            continue

        source_list = ", ".join(f"{c.project}" for c, _ in results)
        console.print(f"[dim]Sources: {source_list}[/dim]")

        with console.status("[bold green]Thinking...[/bold green]"):
            chunks = [c for c, _ in results]
            answer = answer_query(question, chunks)

        console.print("\n[bold green]Claude:[/bold green]")
        console.print(Markdown(answer))


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, readable=True))
def transcript(filepath: str) -> None:
    """Extract decisions and action items from a meeting transcript.

    Output is formatted as CLAUDE.md-style Decision Points sections.

    \b
    Example:
      python main.py transcript meeting_notes.txt
    """
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    char_count = len(text)
    console.print(f"[dim]Processing {filepath} ({char_count:,} chars)...[/dim]")

    with console.status("[bold green]Extracting decisions and action items...[/bold green]"):
        result = extract_transcript_decisions(text)

    console.print()
    console.print(
        Panel(
            Markdown(result),
            title="Extracted Decisions & Action Items",
            border_style="yellow",
        )
    )


if __name__ == "__main__":
    cli()
