"""Tests for PDF strategy dispatch.

``auto`` uses whichever remote parser is configured and falls back to the local one,
so the contract worth protecting is:

- with a remote service configured, it is tried *before* local;
- if it fails, the document is still converted, by local;
- with nothing configured, local runs and no remote call is made.

``mineru`` stays strict: a failure is the result.
"""

import asyncio
from pathlib import Path

import httpx
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
    defaults = {"strategy": "auto"}
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


# ---------------------------------------------------------------------------
# auto: configured remote first, local as the fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_prefers_configured_mineru(monkeypatch, pdf_path):
    parser = _parser(mineru_endpoint=ENDPOINT)
    calls = _stub_converters(monkeypatch, parser, mineru=(MINERU_MD, {"strategy": "mineru"}))

    md, meta = await parser._convert_to_markdown(pdf_path)

    assert md == MINERU_MD
    assert meta["strategy"] == "mineru"
    # Remote first means local is not even called when the remote succeeds.
    assert calls == {"mineru": 1, "local": 0}


@pytest.mark.asyncio
async def test_auto_falls_back_to_local_when_mineru_raises(monkeypatch, pdf_path):
    """The behaviour that matters: a remote failure degrades, it does not fail."""
    parser = _parser(mineru_endpoint=ENDPOINT)
    calls = _stub_converters(
        monkeypatch, parser, mineru=RuntimeError("MinerU endpoint unreachable")
    )

    md, meta = await parser._convert_to_markdown(pdf_path)

    assert md == LOCAL_MD
    assert meta["strategy"] == "local"
    assert calls == {"mineru": 1, "local": 1}


@pytest.mark.asyncio
async def test_auto_falls_back_to_local_on_timeout(monkeypatch, pdf_path):
    parser = _parser(mineru_endpoint=ENDPOINT)
    _stub_converters(monkeypatch, parser, mineru=TimeoutError("timed out"))

    md, meta = await parser._convert_to_markdown(pdf_path)

    assert md == LOCAL_MD
    assert meta["strategy"] == "local"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [404, 405, 429, 500, 503],
)
def test_auto_falls_back_on_any_http_error(monkeypatch, pdf_path, status):
    """Every remote failure mode must degrade to local, not fail the import.

    404/405 matter in practice: the hosted MinerU service has no ``/file_parse``
    route at all (its contract is the async batch flow), so a user who points
    ``mineru_endpoint`` at it under ``auto`` hits 404 on every single document.
    """
    import httpx as _httpx

    parser = _parser(mineru_endpoint=ENDPOINT)

    async def boom(self, pdf_path, storage=None, resource_name=None):
        raise _httpx.HTTPStatusError(f"HTTP {status}", request=None, response=None)

    async def ok_local(self, pdf_path, storage=None, resource_name=None):
        return LOCAL_MD, {"strategy": "local"}

    monkeypatch.setattr(PDFParser, "_convert_mineru", boom)
    monkeypatch.setattr(PDFParser, "_convert_local", ok_local)

    md, meta = asyncio.run(parser._convert_to_markdown(pdf_path))

    assert md == LOCAL_MD
    assert meta["strategy"] == "local"


@pytest.mark.asyncio
async def test_auto_uses_local_when_nothing_configured(monkeypatch, pdf_path):
    """No endpoint means no remote service is in play; local must not be bypassed."""
    parser = _parser()
    calls = _stub_converters(monkeypatch, parser, mineru=(MINERU_MD, {"strategy": "mineru"}))

    md, meta = await parser._convert_to_markdown(pdf_path)

    assert md == LOCAL_MD
    assert calls == {"mineru": 0, "local": 1}


def test_remote_converters_only_lists_configured_services():
    assert _parser()._remote_converters() == []
    providers = _parser(mineru_endpoint=ENDPOINT)._remote_converters()
    assert [name for name, _ in providers] == ["MinerU"]


def test_mineru_enabled_is_the_single_source_of_truth():
    """Both the parser dispatch and the startup preflight read this one predicate."""
    assert PDFConfig(strategy="auto").mineru_enabled is False
    assert PDFConfig(strategy="auto", mineru_endpoint="").mineru_enabled is False
    assert PDFConfig(strategy="auto", mineru_endpoint=ENDPOINT).mineru_enabled is True
    # A token alone does not enable remote parsing: it is a credential, not an endpoint.
    assert PDFConfig(strategy="auto", mineru_token="t").mineru_enabled is False


@pytest.mark.asyncio
async def test_local_strategy_never_calls_the_remote(monkeypatch, pdf_path):
    parser = _parser(strategy="local", mineru_endpoint=ENDPOINT)
    calls = _stub_converters(monkeypatch, parser, mineru=(MINERU_MD, {"strategy": "mineru"}))

    md, _ = await parser._convert_to_markdown(pdf_path)

    assert md == LOCAL_MD
    assert calls == {"mineru": 0, "local": 1}


# ---------------------------------------------------------------------------
# mineru: strict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mineru_strategy_does_not_fall_back(monkeypatch, pdf_path):
    parser = _parser(strategy="mineru", mineru_endpoint=ENDPOINT)
    calls = _stub_converters(
        monkeypatch, parser, mineru=RuntimeError("MinerU endpoint unreachable")
    )

    with pytest.raises(RuntimeError):
        await parser._convert_to_markdown(pdf_path)

    assert calls == {"mineru": 1, "local": 0}


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_strategy_is_accepted():
    for strategy in ("local", "auto"):
        PDFConfig(strategy=strategy).validate()
    PDFConfig(strategy="mineru", mineru_endpoint=ENDPOINT).validate()


def test_endpoint_is_required_for_mineru():
    with pytest.raises(ValueError, match="mineru_endpoint"):
        PDFConfig(strategy="mineru").validate()


def test_endpoint_is_optional_for_auto():
    PDFConfig(strategy="auto").validate()


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Invalid strategy"):
        PDFConfig(strategy="mineru_first", mineru_endpoint=ENDPOINT).validate()


# ---------------------------------------------------------------------------
# hosted-service auth
# ---------------------------------------------------------------------------


def test_token_is_sent_as_bearer(monkeypatch, pdf_path):
    """The hosted service needs a bearer token; verify it reaches the request."""
    captured = {}

    def fake_post(self, url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        raise RuntimeError("stop here - only the request shape matters")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    parser = _parser(strategy="mineru", mineru_endpoint=ENDPOINT, mineru_token="secret-token")
    assert parser._mineru_headers() == {"Authorization": "Bearer secret-token"}

    # _convert_mineru logs and re-raises transport failures; we only care that the
    # call went out carrying the token.
    with pytest.raises(RuntimeError, match="only the request shape matters"):
        asyncio.run(parser._convert_mineru(pdf_path))

    assert captured["headers"] == {"Authorization": "Bearer secret-token"}
    assert captured["url"].endswith("/file_parse")


def test_no_token_means_no_auth_header():
    parser = _parser(mineru_endpoint=ENDPOINT)
    assert parser._mineru_headers() == {}


# ---------------------------------------------------------------------------
# protocol selection: a token means the hosted API, no token means self-hosted
# ---------------------------------------------------------------------------


def test_token_selects_the_hosted_batch_protocol(monkeypatch):
    """The hosted service has no /file_parse, so a token must pick the batch flow."""
    seen = []

    async def fake_batch(self, client, meta, pdf_path, storage=None, resource_name=None):
        seen.append("batch")
        return MINERU_MD, {"strategy": "mineru", "api_mode": "v4-batch"}

    async def fake_sync(self, client, meta, pdf_path, storage=None, resource_name=None):
        seen.append("sync")
        return MINERU_MD, {"strategy": "mineru", "api_mode": "file-parse"}

    monkeypatch.setattr(PDFParser, "_mineru_hosted_batch", fake_batch)
    monkeypatch.setattr(PDFParser, "_mineru_selfhosted", fake_sync)

    parser = _parser(strategy="mineru", mineru_endpoint=ENDPOINT, mineru_token="t")
    _, meta = asyncio.run(parser._convert_to_markdown(Path("x.pdf")))

    assert seen == ["batch"]
    assert meta["api_mode"] == "v4-batch"


def test_no_token_keeps_the_self_hosted_route(monkeypatch):
    """Backward compatibility: a tokenless endpoint keeps using /file_parse."""
    seen = []

    async def fake_batch(self, client, meta, pdf_path, storage=None, resource_name=None):
        seen.append("batch")
        return MINERU_MD, {"strategy": "mineru"}

    async def fake_sync(self, client, meta, pdf_path, storage=None, resource_name=None):
        seen.append("sync")
        return MINERU_MD, {"strategy": "mineru", "api_mode": "file-parse"}

    monkeypatch.setattr(PDFParser, "_mineru_hosted_batch", fake_batch)
    monkeypatch.setattr(PDFParser, "_mineru_selfhosted", fake_sync)

    parser = _parser(strategy="mineru", mineru_endpoint=ENDPOINT)
    _, meta = asyncio.run(parser._convert_to_markdown(Path("x.pdf")))

    assert seen == ["sync"]
    assert meta["api_mode"] == "file-parse"


def test_hosted_batch_failure_falls_back_to_self_hosted(monkeypatch):
    """A rejected token must not sink the parse if the endpoint still speaks v1."""
    seen = []

    async def refuse(self, client, base, pdf_path):
        raise RuntimeError("token rejected")

    async def fake_sync(self, client, meta, pdf_path, storage=None, resource_name=None):
        seen.append("sync")
        return MINERU_MD, {"strategy": "mineru", "api_mode": "file-parse"}

    monkeypatch.setattr(PDFParser, "_mineru_submit_file_batch", refuse)
    monkeypatch.setattr(PDFParser, "_mineru_selfhosted", fake_sync)

    parser = _parser(strategy="mineru", mineru_endpoint=ENDPOINT, mineru_token="t")
    _, meta = asyncio.run(parser._convert_to_markdown(Path("x.pdf")))

    assert seen == ["sync"]
    assert meta["api_mode"] == "file-parse"


# ---------------------------------------------------------------------------
# regressions raised in review: default-config startup, credential isolation,
# and image-reference rewriting
# ---------------------------------------------------------------------------


def test_default_pdfconfig_reports_no_remote_provider():
    """A deployment that never configured MinerU must still start.

    ``remote_enabled`` is read on the default ``auto`` path during service start-up,
    so it has to cope with the default ``None`` endpoint instead of raising.
    """
    from openviking_cli.utils.config.parser_config import PDFConfig

    config = PDFConfig()
    assert config.mineru_endpoint is None
    assert config.remote_enabled is False


def test_remote_enabled_true_only_for_a_configured_endpoint():
    from openviking_cli.utils.config.parser_config import PDFConfig

    config = PDFConfig()
    config.mineru_endpoint = "https://mineru.example"
    assert config.remote_enabled is True

    config.mineru_endpoint = ""
    assert config.remote_enabled is False


def test_presigned_zip_download_carries_no_mineru_token(monkeypatch):
    """The archive lives on the storage host, so the MinerU token must not go there."""
    from openviking_cli.utils.config.parser_config import PDFConfig

    parser = PDFParser(PDFConfig(mineru_endpoint=ENDPOINT, mineru_token="secret-token"))
    seen = {}

    class _Resp:
        content = b"zip-bytes"

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs.get("headers")
            seen["kwargs"] = kwargs
            return _Resp()

    zip_url = "https://oss.storage.example/batch/abc.zip?signature=deadbeef"
    asyncio.run(parser._mineru_download_zip(_Client(), zip_url))

    assert seen["url"] == zip_url
    assert "headers" not in seen["kwargs"], "a token-bearing header must not be sent"
    # The API headers still carry the token; they are simply not used for the
    # cross-host download.
    assert parser._mineru_headers()["Authorization"] == "Bearer secret-token"


def test_self_hosted_images_are_saved_and_references_rewritten(monkeypatch, pdf_path):
    """A saved image whose markdown still points at the provider path is broken."""
    from openviking_cli.utils.config.parser_config import PDFConfig

    monkeypatch.setattr(
        PDFParser,
        "_mineru_base_url",
        lambda self: ENDPOINT,
    )
    monkeypatch.setattr(
        PDFParser,
        "_mineru_file_parse_url",
        lambda self: ENDPOINT + "/file_parse",
    )

    markdown = "see ![fig](images/figure.png) for detail"
    # /file_parse answers with results keyed by file name
    payload = {
        "results": {
            pdf_path.name: {
                "md_content": markdown,
                "images": {"images/figure.png": "ZmFrZQ=="},
            }
        }
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class _Client:
        async def post(self, *a, **kw):
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    saved = {}

    class _Storage:
        media_dir = Path("/media")

        def save_image(self, resource_name, blob, filename, extension):
            target = Path("/media") / resource_name / (filename + extension)
            saved["path"] = target
            saved["blob"] = blob
            return target

    monkeypatch.setattr(
        "openviking.parse.parsers.pdf.lazy_import",
        lambda name: __import__("httpx") if name == "httpx" else None,
        raising=False,
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: _Client())

    parser = PDFParser(PDFConfig(strategy="mineru", mineru_endpoint=ENDPOINT))
    text, meta = asyncio.run(parser._mineru_selfhosted(_Client(), {}, pdf_path, _Storage(), "math"))

    assert meta["images_saved"] == 1
    assert saved["path"].name == "figure.png"
    # the reference must now resolve to where the image was actually written
    assert "images/figure.png" not in text
    assert "math/figure.png" in text
