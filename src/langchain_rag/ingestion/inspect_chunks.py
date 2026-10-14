from pathlib import Path

from langchain_rag.ingestion.chunking import (
    split_documents_into_chunks,
)
from langchain_rag.ingestion.loaders import load_directory


def main() -> None:
    source_directory = Path("data/raw")

    documents = load_directory(source_directory)
    chunks = split_documents_into_chunks(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Chunk {index} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Page: {chunk.metadata.get('page', 'N/A')}")
        print(f"Start index: {chunk.metadata.get('start_index')}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Content preview: {chunk.page_content[:200]!r}")


if __name__ == "__main__":
    main()
