
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def load_document(file_path: Path) -> list[Document]:
    """Load one supported file into LangChain Document objects."""
    file_path = file_path.resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))
    else:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = str(file_path)

    return documents


def load_directory(directory: Path) -> list[Document]:
    """Load supported files from a directory, non-recursively."""
    directory = directory.resolve()

    if not directory.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")

    documents: list[Document] = []

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_document(file_path))

    return documents
