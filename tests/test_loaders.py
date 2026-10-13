
from pathlib import Path

import pytest
from langchain_core.documents import Document

from langchain_rag.ingestion.loaders import (
    load_directory,
    load_document,
)


def test_load_text_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("RAG retrieves relevant context.", encoding="utf-8")

    documents = load_document(path)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert "retrieves relevant context" in documents[0].page_content
    assert documents[0].metadata["source"] == str(path.resolve())


def test_reject_unsupported_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def test_load_directory(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("First document", encoding="utf-8")
    (tmp_path / "two.md").write_text("# Second document", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")

    documents = load_directory(tmp_path)

    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)

def test_load_text_file_strips_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_text(
        "\ufeffRAG document",
        encoding="utf-8",
    )

    documents = load_document(path)

    assert documents[0].page_content == "RAG document"