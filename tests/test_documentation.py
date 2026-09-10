"""Keep the public entry documentation runnable and its local links valid."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def test_local_markdown_links_resolve():
    documents = [ROOT / "README.md"]
    for name in ("docs", "supplementary", "experiments"):
        documents.extend((ROOT / name).rglob("*.md"))
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            destination = (document.parent / target.split("#")[0]).resolve()
            assert destination.is_relative_to(ROOT), (document, target)
            assert destination.exists(), (document, target)


def test_readme_python_examples_have_the_stated_results():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    examples = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    assert len(examples) == 2
    scope = {}
    for code, expected in zip(examples, (2 / 3, 1 / 6)):
        exec(compile(code, "README.md", "exec"), scope)
        assert scope["result"].gei == pytest.approx(expected)
