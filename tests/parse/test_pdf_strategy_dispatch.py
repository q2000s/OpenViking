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


async def _no_sleep(_seconds):
    """Placeholder so retry backoff does not slow the suite down."""
    return None


# ---------------------------------------------------------------------------
# batch behaviour: partial remote failure, backoff, and circuit breaking
# ---------------------------------------------------------------------------


def _http_error(status):
    """An error shaped like the real thing, with a response carrying the status."""
    import httpx as _httpx

    request = _httpx.Request("POST", ENDPOINT)
    response = _httpx.Response(status, request=request)
    return _httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def test_transient_errors_are_retried_then_fall_back(monkeypatch, pdf_path):
    """A 429 is worth one backed-off retry; a 404 is not."""
    for status, expected_calls in ((429, 2), (500, 2), (404, 1)):
        calls = []

        async def flaky(self, p, storage=None, resource_name=None, _calls=calls, _status=status):
            _calls.append(1)
            raise _http_error(_status)

        async def ok_local(self, pdf_path, storage=None, resource_name=None):
            return LOCAL_MD, {"strategy": "local"}

        async def call_flaky(self, p, storage=None, resource_name=None):
            return await flaky(self, p, storage, resource_name)

        monkeypatch.setattr(PDFParser, "_convert_mineru", call_flaky)
        monkeypatch.setattr(PDFParser, "_convert_local", ok_local)
        monkeypatch.setattr(asyncio, "sleep", _no_sleep)

        parser = _parser(strategy="auto", mineru_endpoint=ENDPOINT)
        md, meta = asyncio.run(parser._convert_to_markdown(pdf_path))

        assert md == LOCAL_MD, status
        assert meta["strategy"] == "local", status
        assert len(calls) == expected_calls, (status, len(calls))


def test_circuit_opens_after_repeated_failures(monkeypatch, pdf_path):
    """Once the remote is clearly down, stop paying for a doomed call per document."""
    calls = []

    async def always_fail(self, pdf_path, storage=None, resource_name=None):
        calls.append(1)
        raise _http_error(404)

    async def ok_local(self, pdf_path, storage=None, resource_name=None):
        return LOCAL_MD, {"strategy": "local"}

    async def _stub_always_fail(self, p, storage=None, resource_name=None):
        return await always_fail(self, p, storage, resource_name)

    monkeypatch.setattr(PDFParser, "_convert_mineru", _stub_always_fail)
    monkeypatch.setattr(PDFParser, "_convert_local", ok_local)

    parser = _parser(strategy="auto", mineru_endpoint=ENDPOINT)
    parser._remote_circuit_threshold = 3
    parser._remote_circuit_cooldown = 1000  # stay open for this test's batch

    for _ in range(10):
        md, meta = asyncio.run(parser._convert_to_markdown(pdf_path))
        assert md == LOCAL_MD
        assert meta["strategy"] == "local"

    # 3 documents opened the circuit; the other 7 never called the remote again.
    assert len(calls) == 3, len(calls)


def test_circuit_recovers_after_a_success(monkeypatch, pdf_path):
    """A single success closes the circuit again."""
    state = {"fail": True}

    async def flaky(self, pdf_path, storage=None, resource_name=None):
        if state["fail"]:
            raise _http_error(404)
        return MINERU_MD, {"strategy": "mineru"}

    async def ok_local(self, pdf_path, storage=None, resource_name=None):
        return LOCAL_MD, {"strategy": "local"}

    async def _stub_flaky(self, p, storage=None, resource_name=None):
        return await flaky(self, p, storage, resource_name)

    monkeypatch.setattr(PDFParser, "_convert_mineru", _stub_flaky)
    monkeypatch.setattr(PDFParser, "_convert_local", ok_local)

    parser = _parser(strategy="auto", mineru_endpoint=ENDPOINT)
    parser._remote_circuit_threshold = 2
    parser._remote_circuit_cooldown = 1000

    asyncio.run(parser._convert_to_markdown(pdf_path))
    asyncio.run(parser._convert_to_markdown(pdf_path))
    assert "MinerU" in parser._remote_circuit_open

    state["fail"] = False
    parser._remote_circuit_cooldown = 1  # let the next document through as a probe
    md, meta = asyncio.run(parser._convert_to_markdown(pdf_path))
    assert md == MINERU_MD
    assert "MinerU" not in parser._remote_circuit_open


def test_batch_with_partial_remote_failures(monkeypatch, tmp_path):
    """10 documents, 3 fail remotely: the batch finishes and stays auditable.

    This is the shape a real import hits -- a handful of bad documents must not take
    down the run, and the operator needs to see that three of them degraded.
    """
    failing = {"doc2.pdf", "doc5.pdf", "doc9.pdf"}
    seen = []

    async def sometimes(self, pdf_path, storage=None, resource_name=None):
        seen.append(pdf_path.name)
        if pdf_path.name in failing:
            raise _http_error(400)
        return MINERU_MD, {"strategy": "mineru"}

    async def ok_local(self, pdf_path, storage=None, resource_name=None):
        return LOCAL_MD, {"strategy": "local"}

    async def _stub_sometimes(self, p, storage=None, resource_name=None):
        return await sometimes(self, p, storage, resource_name)

    monkeypatch.setattr(PDFParser, "_convert_mineru", _stub_sometimes)
    monkeypatch.setattr(PDFParser, "_convert_local", ok_local)

    parser = _parser(strategy="auto", mineru_endpoint=ENDPOINT)
    effective = []

    for i in range(10):
        doc = tmp_path / f"doc{i}.pdf"
        doc.write_bytes(b"%PDF-1.4 minimal")
        _, meta = asyncio.run(parser._convert_to_markdown(doc))
        effective.append(meta["strategy"])

    assert len(seen) == 10, seen
    # Every document produced a result; the failures degraded rather than aborting.
    assert effective.count("local") == len(failing)
    assert effective.count("mineru") == 10 - len(failing)
    # And the degradation is visible to operators.
    assert parser._degraded == len(failing)
