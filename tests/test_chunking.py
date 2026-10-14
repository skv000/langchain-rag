import pytest
from langchain_core.documents import Document

from langchain_rag.ingestion.chunking import (
    split_documents_into_chunks,
)


def test_short_document_preserves_content_and_metadata() -> None:
    document = Document(
        page_content="RAG retrieves relevant context.",
        metadata={"source": "sample.txt", "page": 2},
    )

    chunks = split_documents_into_chunks([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == document.page_content
    assert chunks[0].metadata["source"] == "sample.txt"
    assert chunks[0].metadata["page"] == 2
    assert chunks[0].metadata["start_index"] == 0


def test_long_document_is_split_with_overlap() -> None:
    text = "0123456789" * 30

    document = Document(
        page_content=text,
        metadata={"source": "long.txt"},
    )

    chunks = split_documents_into_chunks(
        [document],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert chunks[0].metadata["start_index"] == 0

    for previous, current in zip(chunks, chunks[1:]):
        assert previous.page_content[-20:] == current.page_content[:20]
        assert current.metadata["start_index"] == (
            previous.metadata["start_index"] + len(previous.page_content) - 20
        )

    assert all(chunk.metadata["source"] == "long.txt" for chunk in chunks)


def test_empty_document_list_returns_empty_list() -> None:
    assert split_documents_into_chunks([]) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_chunk_configuration(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        split_documents_into_chunks(
            [Document(page_content="Test document")],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
