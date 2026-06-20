"""Tests for download endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents.output_format import EXCEL_DIR, PDF_DIR


@pytest.fixture
def sample_pdf():
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(PDF_DIR) / "test.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_excel():
    Path(EXCEL_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(EXCEL_DIR) / "test.xlsx"
    path.write_bytes(b"PK fake excel")
    yield path
    path.unlink(missing_ok=True)


def test_download_pdf(client: TestClient, sample_pdf):
    response = client.get("/api/v1/download/pdfs/test.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_download_excel(client: TestClient, sample_excel):
    response = client.get("/api/v1/download/excel/test.xlsx")
    assert response.status_code == 200


def test_download_pdf_not_found(client: TestClient):
    response = client.get("/api/v1/download/pdfs/missing.pdf")
    assert response.status_code == 404


def test_download_path_traversal_rejected(client: TestClient):
    from api.v1.downloads import _serve_file

    with pytest.raises(Exception) as exc_info:
        _serve_file("pdfs", "../secrets.txt", "application/pdf")
    assert exc_info.value.status_code == 400
