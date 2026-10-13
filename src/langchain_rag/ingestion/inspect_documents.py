
from pathlib import Path

from langchain_rag.ingestion.loaders import load_directory


def main() -> None:
    directory = Path("data/raw")
    documents = load_directory(directory)

    print(f"Loaded {len(documents)} document(s).")

    for index, document in enumerate(documents, start=1):
        print(f"\n--- Document {index} ---")
        print(f"Source: {document.metadata['source']}")
        print(f"Characters: {len(document.page_content)}")
        print(f"Metadata: {document.metadata}")
        print(f"Preview: {document.page_content[:300]!r}")


if __name__ == "__main__":
    main()
