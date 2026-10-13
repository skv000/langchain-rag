def test_python_runtime() -> None:
    import sys

    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12


def test_application_package_imports() -> None:
    import langchain_rag

    assert langchain_rag is not None
