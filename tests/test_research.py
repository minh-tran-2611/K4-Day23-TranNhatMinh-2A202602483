import json

import pytest
from langchain_core.messages import AIMessage

import research
from research import REPORT_PATH, SOURCES_PATH, save_outputs, slugify, summarize


def test_slugify():
    assert slugify("Survey about World Model") == "survey-about-world-model"
    assert slugify("../../x") == "x"
    assert slugify("") == "topic" and slugify("!!!") == "topic"
    assert len(slugify("a" * 200)) == 60


def test_summarize_counts_lead_calls_and_tokens():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "1"}, {"name": "task", "args": {}, "id": "2"},
                                          {"name": "execute", "args": {}, "id": "3"}],
                  usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        AIMessage(content="done", usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
    ]
    meta = summarize(messages, 12.345, "m")
    assert meta == {"model": "m", "elapsed_s": 12.3, "subagent_calls": 2, "tool_calls": {"task": 2, "execute": 1},
                    "tokens": {"input": 13, "output": 7}}


def fake_download(files):
    return lambda backend, paths: {p: files.get(p) for p in paths}


def test_save_outputs_writes_three_files(tmp_path, monkeypatch):
    sources = [{"n": 1, "url": "https://arxiv.org/abs/1", "source": "arxiv"},
               {"n": 2, "url": "https://example.org", "source": "web"}]
    monkeypatch.setattr(research, "download",
                        fake_download({REPORT_PATH: b"# R\n", SOURCES_PATH: json.dumps(sources).encode()}))
    path = save_outputs(None, "My Topic", [], 1.0, "m", reports_dir=tmp_path)
    assert path == tmp_path / "my-topic.md"
    meta = json.loads((tmp_path / "my-topic.meta.json").read_text())
    assert meta["topic"] == "My Topic" and meta["n_sources"] == 2 and meta["source_families"] == ["arxiv", "web"]
    assert (tmp_path / "my-topic.sources.json").exists()


@pytest.mark.parametrize("files", [{}, {REPORT_PATH: b"  "}, {REPORT_PATH: b"# R", SOURCES_PATH: b"{broken"},
                                   {REPORT_PATH: b"# R", SOURCES_PATH: b"[]"}])
def test_failed_run_writes_nothing(tmp_path, monkeypatch, files):
    monkeypatch.setattr(research, "download", fake_download(files))
    with pytest.raises(RuntimeError):
        save_outputs(None, "t", [], 1.0, "m", reports_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_main_without_topic():
    assert research.main("   ") == 2
