from check_citations import check

SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-02",
     "source": "arxiv"},
    {"n": 2, "id": "2502.00002", "url": "https://huggingface.co/papers/2502.00002", "title": "B", "date": "2025-02-02",
     "source": "hf-search"},
]
REFS = ("## References\n"
        "[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n"
        "[2] B. hf-search. https://huggingface.co/papers/2502.00002 (2025-02-02)\n")


def test_ok():
    assert check("# T\n\nClaim [1]. Other [2].\n\n" + REFS, SOURCES) == []


def test_grouped_citations_are_expanded():
    assert check("Claims [1, 2].\n\n" + REFS, SOURCES) == []
    assert check("Claims [1-2].\n\n" + REFS, SOURCES) == []


def test_empty_sources():
    assert check("x", []) == ["no sources in sources.json"]


def test_missing_heading():
    assert any("References" in p for p in check("Claim [1] [2].", SOURCES))


def test_uncited_and_unknown():
    problems = check("Claim [1] [7].\n\n" + REFS, SOURCES)
    assert "[7] cited but missing from sources.json" in problems
    assert "source [2] never cited" in problems


def test_reference_list_numbers_are_not_citations():
    problems = check("Claim [1].\n\n" + REFS, SOURCES)
    assert "source [2] never cited" in problems


def test_code_and_links_ignored():
    problems = check("Claim [1]. `[2]` and [2](http://x)\n\n" + REFS, SOURCES)
    assert "source [2] never cited" in problems


def test_bundled_reference_line():
    refs = ("## References\n"
            "[1] A; B. https://arxiv.org/abs/2501.00001 https://huggingface.co/papers/2502.00002\n")
    problems = check("Claim [1] [2].\n\n" + refs, SOURCES)
    assert any("exactly one URL" in p for p in problems)
    assert "source [2] has no line in References" in problems


def test_wrong_url_and_duplicates():
    sources = SOURCES + [{"n": 3, "url": SOURCES[0]["url"]}, {"n": "4", "url": "ftp://x"}]
    refs = REFS.replace("2502.00002 (", "9999.99999 (") + "[2] dup. https://huggingface.co/papers/2502.00002\n"
    problems = check("Claim [1] [2] [3].\n\n" + refs, sources)
    assert any("duplicates the url" in p for p in problems)
    assert any("not an integer" in p for p in problems)
    assert any("does not match" in p for p in problems)
    assert "reference [2] appears more than once in References" in problems
