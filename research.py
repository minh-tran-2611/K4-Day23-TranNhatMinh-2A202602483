"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
RECURSION_LIMIT = 1000  # LangGraph steps of the lead graph (~2 per model->tool turn); subagents use middleware limits


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(topic).lower())[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\n"
            f"Produce the cited survey report at {REPORT_PATH} and the merged sources at {SOURCES_PATH}, following "
            "every step of your instructions: plan, delegate at least 3 researchers in parallel, check and merge "
            f"their notes, write the report body, run the finalizer and the validator ({VALIDATOR_PATH}) until it "
            "prints OK, then spot-check citations.")


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Lead messages only: subagent tokens are not included, so this undercounts the real cost.
    """
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    A failed run (missing/empty report, missing/invalid sources.json) raises RuntimeError and writes nothing.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report, raw_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report or not report.strip():
        raise RuntimeError(f"the agent produced no report at {REPORT_PATH}")
    try:
        sources = json.loads(raw_sources or b"")
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is missing or not valid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON list")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")})}
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    (reports_dir / f"{slug}.sources.json").write_bytes(raw_sources)
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report)
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = topic.strip()
    if not topic:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    model = make_model()
    start = time.monotonic()
    with open_sandbox() as backend:  # the sandbox is always stopped and removed, even on errors
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": RECURSION_LIMIT})
            report_path = save_outputs(backend, topic, result["messages"], time.monotonic() - start,
                                       _model_name(model))
        except Exception as exc:  # GraphRecursionError, model/API errors, missing outputs: a failed run
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
