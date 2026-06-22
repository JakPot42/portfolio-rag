import re
from pathlib import Path

from portfolio_rag.bm25_index import BM25Index, Chunk

PROJECTS_ROOT = Path(r"C:\Users\JakPot\Projects")
SEAD3_README = Path(r"C:\Users\JakPot\Downloads\sead3_auditor_2\sead3_auditor\README.md")

DIR_TO_PROJECT: dict[str, str] = {
    "agora": "Agora",
    "ato_accelerator": "ATO Accelerator",
    "biomon": "BioMon",
    "cable_analyzer": "Cable Resilience Analyzer",
    "carta": "Carta",
    "cfius_screener": "CFIUS Screener",
    "citation_checker": "Citation Checker",
    "civic_rag": "Civic RAG",
    "critical_mineral": "Critical Mineral Monitor",
    "cve_prioritizer": "CVE Prioritizer",
    "dib_monitor": "DIB Monitor",
    "distributed_inference": "Distributed Inference",
    "friendshore": "FriendShore",
    "ghosttrace": "GhostTrace",
    "harvest_horizon": "Harvest Horizon",
    "lease_translator": "Lease Translator",
    "litigation_network": "Litigation Network",
    "mof_discovery": "MOF Discovery",
    "ncf_ttx": "NCF TTX Generator",
    "orbital_sentinel": "Orbital Sentinel",
    "race_condition": "race-condition",
    "rai_compliance": "RAI Compliance",
    "redteam_eval": "redteam-eval",
    "sam_agent": "SAM Acquisition Agent",
    "scotus_analyzer": "SCOTUS Analyzer",
    "scrna_explorer": "scRNA Explorer",
    "sentinel": "SENTINEL",
    "z3_contract": "z3-contract",
    "contextslim": "ContextSlim",
    "fault_injector": "Fault Injector",
    "osint_triage": "OSINT Triage",
    "portfolio_rag": "Portfolio RAG",
}

# Directories to skip when scanning project folders
_SKIP_DIRS = {"node_modules", ".pytest_cache", "__pycache__", ".git", "venv", ".venv"}


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on ## headers. Returns (header, body) pairs."""
    parts = re.split(r"^(##+ .+)$", text, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []

    if parts[0].strip():
        sections.append(("(intro)", parts[0]))

    i = 1
    while i + 1 < len(parts):
        header = parts[i].strip()
        body = parts[i + 1]
        combined = f"{header}\n{body}".strip()
        if len(combined) >= 30:
            sections.append((header, combined))
        i += 2

    return sections


def chunks_from_file(filepath: Path, project: str, filename: str) -> list[Chunk]:
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    sections = _split_sections(text)
    return [
        Chunk(chunk_id=0, project=project, filename=filename, section=header, text=body)
        for header, body in sections
    ]


def build_index() -> BM25Index:
    idx = BM25Index()
    chunk_id = 0

    def _add(chunks: list[Chunk]) -> None:
        nonlocal chunk_id
        for c in chunks:
            c.chunk_id = chunk_id
            chunk_id += 1
            idx.add_chunk(c)

    # Master portfolio CLAUDE.md — richest single source
    master = PROJECTS_ROOT / "CLAUDE.md"
    if master.exists():
        _add(chunks_from_file(master, "Portfolio", "CLAUDE.md"))

    # Per-project README.md and CLAUDE.md (top-level only, skip node_modules etc.)
    for entry in sorted(PROJECTS_ROOT.iterdir()):
        if not entry.is_dir() or entry.name in _SKIP_DIRS or entry.name == "JakPot42":
            continue
        project = DIR_TO_PROJECT.get(entry.name, entry.name.replace("_", " ").title())
        for fname in ("README.md", "CLAUDE.md"):
            fpath = entry / fname
            if fpath.exists():
                _add(chunks_from_file(fpath, project, f"{entry.name}/{fname}"))

    # SEAD 3 lives outside the main Projects dir
    if SEAD3_README.exists():
        _add(chunks_from_file(SEAD3_README, "SEAD 3 Auditor", "sead3/README.md"))

    idx.build()
    return idx
