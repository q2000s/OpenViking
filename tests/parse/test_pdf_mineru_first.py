"""Tests for the ``mineru_first`` PDF strategy.

``mineru_first`` prefers the MinerU API but degrades to the local parser instead of
failing the import, so the contract worth protecting is the fallback: the parse must
succeed through the local path whenever MinerU raises, and the result metadata must
say which converter actually ran.
"""

from pathlib import Path

import pytest

from openviking.parse.parsers.pdf import PDFParser
from openviking_cli.utils.config.parser_config import PDFConfig

ENDPOINT = "http://127.0.0.1:8000"
MINERU_MD = "# From MinerU"
LOCAL_MD = "# From local pdfplumber"


@pytest.fixture
def pdf_path(tmp_path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 minimal")
    return p


def _parser(**overrides) -> PDFParser:
    defaults = {"strategy": "mineru_first", "mineru_endpoint": ENDPOINT}
    defaults.update(overrides)
    return PDFParser(PDFConfig(**defaults))


def _stub_converters(monkeypatch, parser, *, mineru):
    """Replace both conversion paths so the test exercises dispatch, not HTTP."""
    calls = {"mineru": 0, "local": 0}

    async def fake_mineru(self, pdf_path, storage=None, resource_name=None):
        calls["mineru"] += 1
        if isinstance(mineru, Exception):
            raise mineru
        return mineru

    async def fake_local(self, pdf_path, storage=None, resource_name=None):
        calls["local"] += 1
        return LOCAL_MD, {"strategy": "local"}

    monkeypatch.setattr(PDFParser, "_convert_mineru", fake_mineru)
    monkeypatch.setattr(PDFParser, "_convert_local", fake_local)
    return calls


@pytest.mark.asyncio
async def test_uses_mineru_when_available(monkeypatch, pdf_path):
    parser = _parser()
    calls = _stub_converters(monkeypatch, parser, mineru=(MINERU_MD, {"strategy": "mineru"}))

    md, meta = await parser._convert_to_markdown(pdf_path)

    assert md == MINERU_MD
    assert meta["strategy"] == "mineru"
    assert calls == {"mineru": 1, "local": 0}


@pytest.mark.asyncio
async def test_falls_back_to_local_when_mineru_raises(monkeypatch, pdf_path):
    """The behaviour the strategy exists for: a MinerU failure must not fail the parse."""
    parser = _parser()
    calls = _stub_converters(
        monkeypatch, parser, mineru=RuntimeError("MinerU endpoint unreachable")
    )

    md, meta = await parser._convert_to_markdown(pdf_path)

    assert md == LOCAL_MD
    assert meta["strategy"] == "local"
    assert calls == {"mineru": 1, "local": 1}


@pytest.mark.asyncio
async def test_falls_back_to_local_on_timeout(monkeypatch, pdf_path):
    parser = _parser()
    _stub_converters(monkeypatch, parser, mineru=TimeoutError("timed out"))

    md, meta = await parser._convert_to_markdown(pdf_path)

    assert md == LOCAL_MD
    assert meta["strategy"] == "local"


def test_strategy_is_accepted():
    PDFConfig(strategy="mineru_first", mineru_endpoint=ENDPOINT).validate()


def test_endpoint_is_required():
    with pytest.raises(ValueError, match="mineru_endpoint"):
        PDFConfig(strategy="mineru_first").validate()


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Invalid strategy"):
        PDFConfig(strategy="mineru-first", mineru_endpoint=ENDPOINT).validate()
