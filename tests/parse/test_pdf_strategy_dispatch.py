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


# ---------------------------------------------------------------------------
# hosted v4 batch protocol: upload handshake, polling, archive unpacking
# ---------------------------------------------------------------------------


async def _no_sleep(_seconds):
    """Poll without waiting, so the timeout test does not spend real seconds."""
    return None


def _zip_bytes(entries):
    """Build an in-memory zip from {name: text-bytes}."""
    import io as _io
    import zipfile

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, blob in entries.items():
            zf.writestr(name, blob)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _HostedClient:
    """Replays the hosted v4 conversation: reserve url, PUT, poll, download."""

    def __init__(self, polls, zip_url="https://oss.example/result.zip", zip_payload=b""):
        self.polls = list(polls)
        self.zip_url = zip_url
        self.zip_payload = zip_payload
        self.calls = []

    async def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return _FakeResponse(
            {
                "code": 0,
                "data": {
                    "batch_id": "batch-1",
                    "file_urls": ["https://oss.example/upload/doc.pdf"],
                },
            }
        )

    async def put(self, url, **kw):
        self.calls.append(("PUT", url, kw))
        return _FakeResponse()

    async def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        if "/extract-results/" in url:
            if not self.polls:
                raise AssertionError("polled more times than the test scripted")
            return _FakeResponse(self.polls.pop(0))
        return _FakeResponse(content=self.zip_payload)


def _hosted_parser(timeout=30.0):
    from openviking_cli.utils.config.parser_config import PDFConfig

    return PDFParser(
        PDFConfig(
            strategy="mineru",
            mineru_endpoint="https://mineru.example/api",
            mineru_token="tok",
            mineru_timeout=timeout,
        )
    )


def test_hosted_batch_reserves_uploads_polls_and_unpacks(monkeypatch, pdf_path):
    archive = _zip_bytes(
        {
            "full.md": b"# Title\n\n![fig](images/fig-1.png)",
            "images/fig-1.png": b"\x89PNG-bytes",
        }
    )
    client = _HostedClient(
        polls=[
            {"code": 0, "data": {"extract_result": [{"state": "waiting-file"}]}},
            {
                "code": 0,
                "data": {
                    "extract_result": [
                        {"state": "done", "full_zip_url": "https://oss.example/result.zip"}
                    ]
                },
            },
        ],
        zip_payload=archive,
    )

    saved = {}

    class _Storage:
        media_dir = Path("/media")

        def save_image(self, resource_name, blob, filename, extension):
            saved[(resource_name, filename, extension)] = blob
            return Path("/media") / resource_name / (filename + extension)

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: client)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    parser = _hosted_parser()
    meta = {}
    text, meta = asyncio.run(parser._mineru_hosted_batch(client, meta, pdf_path, _Storage(), "doc"))

    assert text.startswith("# Title")
    # the image reference must point at where the file was written
    assert "images/fig-1.png" not in text
    assert "doc/fig-1.png" in text
    assert meta["api_version"] == "task-zip"
    assert meta["images_saved"] == 1
    assert meta["batch_id"] == "batch-1"
    assert saved[("doc", "fig-1", ".png")] == b"\x89PNG-bytes"

    verbs = [c[0] for c in client.calls]
    assert verbs[0] == "POST" and "/file-urls/batch" in client.calls[0][1]
    assert verbs[1] == "PUT"
    # the pre-signed upload carries no Content-Type and no MinerU bearer
    assert client.calls[1][2]["headers"] == {"Content-Type": ""}
    # the archive download must not carry the token to the storage host
    zip_get = [c for c in client.calls if c[0] == "GET" and "result.zip" in c[1]][0]
    assert "headers" not in zip_get[2]


def test_hosted_batch_times_out_without_a_result(monkeypatch, pdf_path):
    client = _HostedClient(polls=[{"data": {"extract_result": [{"state": "running"}]}}] * 3)
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: client)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    parser = _hosted_parser(timeout=5.0)
    try:
        asyncio.run(parser._mineru_hosted_batch(client, {}, pdf_path, None, "doc"))
    except TimeoutError as e:
        assert "did not finish" in str(e)
    else:
        raise AssertionError("expected TimeoutError")


def test_hosted_batch_reports_a_failed_batch(monkeypatch, pdf_path):
    client = _HostedClient(
        polls=[{"data": {"extract_result": [{"state": "failed", "err_msg": "bad pdf"}]}}]
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: client)

    parser = _hosted_parser()
    try:
        asyncio.run(parser._mineru_hosted_batch(client, {}, pdf_path, None, "doc"))
    except ValueError as e:
        assert "err_msg=bad pdf" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_hosted_batch_done_without_zip_url_is_an_error(monkeypatch, pdf_path):
    client = _HostedClient(polls=[{"data": {"extract_result": [{"state": "done"}]}}])
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: client)

    parser = _hosted_parser()
    try:
        asyncio.run(parser._mineru_hosted_batch(client, {}, pdf_path, None, "doc"))
    except ValueError as e:
        assert "without a result zip" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_submit_file_batch_rejects_a_reply_without_urls(monkeypatch, pdf_path):
    class _NoUrls(_HostedClient):
        async def post(self, url, **kw):
            return _FakeResponse({"code": 0, "data": {"file_urls": []}})

    parser = _hosted_parser()
    try:
        asyncio.run(
            parser._mineru_submit_file_batch(_NoUrls([]), "https://mineru.example/api", pdf_path)
        )
    except ValueError as e:
        assert "no file_urls" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_post_json_raises_on_a_business_error_code():
    import httpx

    from openviking_cli.utils.config.parser_config import PDFConfig

    parser = PDFParser(PDFConfig(mineru_endpoint="https://x", mineru_token="t"))

    class _Client:
        async def post(self, *a, **kw):
            return _FakeResponse({"code": 4001, "msg": "quota exceeded"})

    try:
        asyncio.run(parser._mineru_post_json(_Client(), "https://x/y", {}))
    except RuntimeError as e:
        assert "code=4001" in str(e) and "quota exceeded" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert httpx is not None


def test_pick_markdown_prefers_full_md(tmp_path):
    (tmp_path / "full.md").write_text("full", encoding="utf-8")
    (tmp_path / "other.md").write_text("other", encoding="utf-8")
    nested = tmp_path / "deep" / "deeper"
    nested.mkdir(parents=True)
    (nested / "x.md").write_text("x", encoding="utf-8")

    assert PDFParser._mineru_pick_markdown(tmp_path).name == "full.md"


def test_pick_markdown_returns_the_only_candidate_and_none_for_empty(tmp_path):
    assert PDFParser._mineru_pick_markdown(tmp_path) is None
    only = tmp_path / "doc.md"
    only.write_text("x", encoding="utf-8")
    assert PDFParser._mineru_pick_markdown(tmp_path) == only


def test_pick_markdown_falls_back_to_the_shallowest_largest(tmp_path):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "big.md").write_text("x" * 50, encoding="utf-8")
    (tmp_path / "small.md").write_text("y", encoding="utf-8")
    assert PDFParser._mineru_pick_markdown(tmp_path).parent == tmp_path


def test_unpack_zip_without_markdown_is_an_error(pdf_path):
    parser = _hosted_parser()
    try:
        parser._mineru_unpack_zip(_zip_bytes({"images/a.png": b"x"}), pdf_path, {})
    except ValueError as e:
        assert "No markdown file found" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_unpack_zip_handles_empty_markdown(pdf_path):
    parser = _hosted_parser()
    text, meta = parser._mineru_unpack_zip(_zip_bytes({"full.md": b""}), pdf_path, {})
    assert text == ""
    assert meta["api_version"] == "task-zip"


def test_online_state_helpers():
    entry = lambda **kw: {"data": {"extract_result": [kw]}}  # noqa: E731
    assert PDFParser._mineru_online_done(entry(state="done"))
    assert not PDFParser._mineru_online_done(entry(state="running"))
    assert not PDFParser._mineru_online_done({})
    for busy in ("waiting", "waiting-file", "uploading", "pending", "running", "converting"):
        assert PDFParser._mineru_online_busy(entry(state=busy)), busy
    assert not PDFParser._mineru_online_busy(entry(state="done"))
    assert PDFParser._mineru_online_result_entry({"extract_result": []}) is None
    assert PDFParser._mineru_online_result_entry({"extract_result": [1]}) is None


def test_online_zip_url_and_failure_detail():
    good = {"extract_result": [{"full_zip_url": "https://a/b.zip"}]}
    assert PDFParser._mineru_online_zip_url(good) == "https://a/b.zip"
    assert PDFParser._mineru_online_zip_url({"extract_result": [{"zip_url": "ftp://x"}]}) is None
    assert PDFParser._mineru_online_zip_url({}) is None
    assert (
        PDFParser._mineru_online_failure_detail({"extract_result": [{"err_no": "E9"}]})
        == "err_no=E9"
    )
    # with nothing on the entry, the generic failure detail still yields text
    assert PDFParser._mineru_online_failure_detail({"msg": "nope"})


def test_extract_markdown_collects_only_image_files(tmp_path):
    (tmp_path / "full.md").write_text("# t", encoding="utf-8")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"p")
    (tmp_path / "images" / "b.txt").write_bytes(b"t")

    from openviking_cli.utils.config.parser_config import PDFConfig

    parser = PDFParser(PDFConfig(mineru_endpoint="https://x"))
    text, images = parser._mineru_extract_markdown(tmp_path / "full.md", tmp_path)
    assert text == "# t"
    assert list(images) == ["images/a.png"]


def test_decode_inline_images_skips_bad_payloads():
    decoded = PDFParser._mineru_decode_inline_images(
        {"ok.png": "data:image/png;base64,ZmFrZQ==", "bad.png": "!!!not base64!!!"}
    )
    assert decoded == {"ok.png": b"fake"}
    assert PDFParser._mineru_decode_inline_images(None) == {}


def test_save_images_returns_empty_and_survives_a_failing_store():
    from openviking_cli.utils.config.parser_config import PDFConfig

    parser = PDFParser(PDFConfig(mineru_endpoint="https://x"))
    assert asyncio.run(parser._mineru_save_images({}, None, None)) == {}

    class _Broken:
        media_dir = Path("/media")

        def save_image(self, *a, **kw):
            raise OSError("disk full")

    assert asyncio.run(parser._mineru_save_images({"a.png": b"x"}, _Broken(), "doc")) == {}


def test_convert_mineru_requires_an_endpoint():

    parser = _hosted_parser()
    parser.config.mineru_endpoint = ""
    try:
        asyncio.run(parser._convert_mineru(Path("x.pdf")))
    except ValueError as e:
        assert "endpoint not configured" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_payload_unwrapping_and_sync_entry_edges(tmp_path):
    assert PDFParser._mineru_payload_data({"data": {"k": 1}}) == {"k": 1}
    assert PDFParser._mineru_payload_data({"k": 1}) == {"k": 1}
    assert PDFParser._mineru_payload_data(["not", "a", "dict"]) == ["not", "a", "dict"]
    # a truthy non-dict results value must be rejected, not iterated
    assert PDFParser._mineru_sync_entry({"results": "oops"}, tmp_path / "a.pdf") is None
    assert PDFParser._mineru_sync_entry({"results": []}, tmp_path / "a.pdf") is None
    assert PDFParser._mineru_sync_entry({"results": {"a.pdf": "s"}}, tmp_path / "a.pdf") is None
    assert PDFParser._mineru_sync_entry(
        {"results": {"b.pdf": {"md_content": "x"}}}, tmp_path / "a.pdf"
    ) == {"md_content": "x"}


def test_http_status_reads_the_status_off_an_exception():
    import httpx

    req = httpx.Request("GET", "https://x")
    resp = httpx.Response(503, request=req)
    exc = httpx.HTTPStatusError("boom", request=req, response=resp)
    assert PDFParser._http_status(exc) == 503
    assert PDFParser._http_status(ValueError("no response")) is None


def test_failure_detail_falls_back_when_the_reply_says_nothing():
    assert PDFParser._mineru_online_failure_detail({}) == "no failure detail reported"
    # a reason on the batch envelope still surfaces when the entry is silent
    assert PDFParser._mineru_online_failure_detail({"data": {"msg": "quota"}}) == "msg=quota"


def test_unpack_zip_defaults_resource_name_and_skips_subdirectories(pdf_path):
    archive = _zip_bytes(
        {
            "full.md": b"![a](images/a.png)",
            "images/a.png": b"png",
            "images/nested-dir/x": b"junk",
        }
    )
    saved = {}

    class _Storage:
        media_dir = Path("/media")

        def save_image(self, resource_name, blob, filename, extension):
            saved["resource"] = resource_name
            return Path("/media") / resource_name / (filename + extension)

    parser = _hosted_parser()
    # resource_name=None must fall back to the pdf stem
    text, meta = parser._mineru_unpack_zip(archive, pdf_path, {}, _Storage(), None)
    assert saved["resource"] == pdf_path.stem
    assert meta["images_saved"] == 1
    assert "images/a.png" not in text


def test_unpack_zip_uses_the_default_storage_when_none_is_given(pdf_path, monkeypatch):
    archive = _zip_bytes({"full.md": b"![a](images/a.png)", "images/a.png": b"png"})

    class _Storage:
        media_dir = Path("/media")

        def save_image(self, resource_name, blob, filename, extension):
            return Path("/media") / resource_name / (filename + extension)

    monkeypatch.setattr(
        "openviking_cli.utils.storage.get_storage", lambda: _Storage(), raising=False
    )
    parser = _hosted_parser()
    text, meta = parser._mineru_unpack_zip(archive, pdf_path, {}, None, "doc")
    assert meta["images_saved"] == 1
    assert "doc/a.png" in text


def test_selfhosted_without_markdown_is_an_error(pdf_path):
    class _Client:
        async def post(self, *a, **kw):
            return _FakeResponse({"results": {}})

    parser = _hosted_parser()
    parser.config.mineru_token = ""
    try:
        asyncio.run(parser._mineru_selfhosted(_Client(), {}, pdf_path, None, "doc"))
    except ValueError as e:
        assert "returned no markdown" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_save_images_falls_back_to_the_default_storage(monkeypatch):
    class _Storage:
        media_dir = Path("/media")

        def save_image(self, resource_name, blob, filename, extension):
            return Path("/media") / resource_name / (filename + extension)

    monkeypatch.setattr(
        "openviking_cli.utils.storage.get_storage", lambda: _Storage(), raising=False
    )
    parser = _hosted_parser()
    repl = asyncio.run(parser._mineru_save_images({"images/a.png": b"png"}, None, "doc"))
    assert repl == {"images/a.png": "doc/a.png"}


def test_parse_records_the_effective_strategy(monkeypatch, pdf_path):
    """`auto` records which converter ran, not only what was requested."""
    from openviking.parse.base import NodeType, ResourceNode, create_parse_result

    class _MD:
        async def parse_content(self, content, **kw):
            return create_parse_result(
                root=ResourceNode(type=NodeType.ROOT),
                source_path=kw.get("source_path", ""),
                source_format="pdf",
                parser_name="MarkdownParser",
                parse_time=0.0,
            )

    class _Storage:
        media_dir = Path("/media")

    parser = _hosted_parser()
    parser.config.strategy = "auto"

    async def _remote(pdf, resource_name=None):
        return "remote markdown", {"strategy": "mineru"}

    monkeypatch.setattr(type(parser), "_remote_converters", lambda self: [("mineru", _remote)])
    monkeypatch.setattr(type(parser), "_get_markdown_parser", lambda self: _MD())
    monkeypatch.setattr(
        "openviking_cli.utils.storage.get_storage", lambda: _Storage(), raising=False
    )

    result = asyncio.run(parser.parse(pdf_path))
    assert result.meta["pdf_strategy"] == "auto"
    assert result.meta["pdf_strategy_effective"] == "mineru"


# ---------------------------------------------------------------------------
# remote circuit breaker
# ---------------------------------------------------------------------------


class _BreakerClient:
    """Remote converter whose success can be toggled per document."""

    def __init__(self):
        self.up = False
        self.calls = 0

    async def __call__(self, pdf_path, resource_name=None):
        self.calls += 1
        if not self.up:
            raise RuntimeError("remote down")
        return "remote markdown", {"strategy": "mineru"}


def _breaker_parser():
    from openviking_cli.utils.config.parser_config import PDFConfig

    return PDFParser(PDFConfig(strategy="auto", mineru_endpoint="https://mineru.example"))


async def _stub_local(self, pdf_path, storage=None, resource_name=None):
    return "local markdown", {"strategy": "local"}


def _run_batch(parser, remote, n, pdf_path):
    """Parse n documents and report the effective strategy of each."""
    parser._remote_converters = lambda: [("MinerU", remote)]
    parser._convert_local = _stub_local.__get__(parser, type(parser))
    out = []
    for _ in range(n):
        _, meta = asyncio.run(parser._convert_to_markdown(pdf_path))
        out.append(meta.get("strategy"))
    return out


def test_circuit_opens_and_the_rest_of_the_batch_skips_remote(pdf_path):
    parser = _breaker_parser()
    parser._remote_circuit_threshold = 3
    parser._remote_circuit_cooldown = 10_000.0  # stay open for this batch
    remote = _BreakerClient()

    effective = _run_batch(parser, remote, 10, pdf_path)

    assert effective.count("local") == 10
    # only the documents up to the threshold paid for a doomed call
    assert remote.calls == 3, remote.calls
    # all ten documents fell back: three on the failure, seven on the open circuit
    assert parser._remote_degraded == 10


def test_circuit_half_opens_once_the_cooldown_passes_and_closes_on_success(pdf_path):
    parser = _breaker_parser()
    parser._remote_circuit_threshold = 2
    parser._remote_circuit_cooldown = 10_000.0
    remote = _BreakerClient()

    _run_batch(parser, remote, 4, pdf_path)
    assert "MinerU" in parser._remote_circuit_open
    calls_while_open = remote.calls

    # let the cooldown elapse, then bring the service back
    parser._remote_circuit_cooldown = 0.0
    remote.up = True

    effective = _run_batch(parser, remote, 2, pdf_path)

    assert "MinerU" not in parser._remote_circuit_open
    assert effective == ["mineru", "mineru"]
    assert remote.calls == calls_while_open + 2


def test_a_failing_probe_rearms_the_cooldown_instead_of_retrying_every_document(pdf_path):
    """Regression: a half-open probe that fails must re-arm, not hammer.

    With a threshold of 1 and a cooldown of 3 documents, a probe failure that did
    not reset the breaker produced [1, 0, 0, 1, 1, 1] -- a remote call per
    document again, which is exactly what the breaker exists to prevent.
    """
    parser = _breaker_parser()
    parser._remote_circuit_threshold = 1
    remote = _BreakerClient()

    calls = []
    real_admit = type(parser)._remote_admit.__get__(parser, type(parser))

    def _admit(name):
        allowed = real_admit(name)
        calls.append(1 if allowed else 0)
        return allowed

    # document 1 fails and opens the circuit
    parser._remote_admit = _admit
    remote_ = _BreakerClient()
    _run_batch(parser, remote_, 1, pdf_path)
    assert calls == [1]
    assert "MinerU" in parser._remote_circuit_open

    # cooldown expires, so one probe gets through; it fails again
    parser._remote_circuit_cooldown = 0.0
    calls.clear()
    _run_batch(parser, remote_, 1, pdf_path)
    parser._remote_admit = real_admit
    assert calls == [1], "the probe should have been let through"
    assert "MinerU" in parser._remote_circuit_open

    # the next document must be skipped: the probe re-armed the cooldown
    parser._remote_circuit_cooldown = 10_000.0
    before = remote_.calls
    _run_batch(parser, remote_, 4, pdf_path)
    assert remote_.calls == before, "no remote call may happen while the circuit is armed"


def test_breaker_state_is_cleared_after_a_success(pdf_path):
    parser = _breaker_parser()
    parser._remote_circuit_threshold = 1
    parser._remote_circuit_cooldown = 0.0
    remote = _BreakerClient()

    _run_batch(parser, remote, 1, pdf_path)
    assert "MinerU" in parser._remote_circuit_open

    remote.up = True
    _run_batch(parser, remote, 1, pdf_path)
    assert not parser._remote_circuit_open
    assert not parser._remote_failures


def test_degraded_count_is_exposed_on_the_parse_result(monkeypatch, pdf_path):
    from openviking.parse.base import NodeType, ResourceNode, create_parse_result

    class _MD:
        async def parse_content(self, content, **kw):
            return create_parse_result(
                root=ResourceNode(type=NodeType.ROOT),
                source_path=kw.get("source_path", ""),
                source_format="pdf",
                parser_name="MarkdownParser",
                parse_time=0.0,
            )

    class _Storage:
        media_dir = Path("/media")

    parser = _breaker_parser()
    parser._remote_circuit_threshold = 1
    parser._remote_circuit_cooldown = 10_000.0
    remote = _BreakerClient()
    monkeypatch.setattr(type(parser), "_remote_converters", lambda self: [("MinerU", remote)])
    monkeypatch.setattr(type(parser), "_convert_local", _stub_local)
    monkeypatch.setattr(type(parser), "_get_markdown_parser", lambda self: _MD())
    monkeypatch.setattr(
        "openviking_cli.utils.storage.get_storage", lambda: _Storage(), raising=False
    )

    result = asyncio.run(parser.parse(pdf_path))
    assert result.meta["pdf_strategy"] == "auto"
    assert result.meta["pdf_strategy_effective"] == "local"
    assert result.meta["pdf_remote_degraded"] == 1
