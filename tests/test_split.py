import pandas as pd

from psx_fintext_lab.data.split import fingerprint_test_set, make_splits


def _data(n: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"id{i:04d}" for i in range(n)]
    labels = (["positive", "negative", "neutral"] * n)[:n]
    dataset = pd.DataFrame({"id": ids, "text": [f"announcement number {i}" for i in range(n)]})
    llm = pd.DataFrame({"id": ids, "sentiment": labels, "confidence": 0.9})
    return dataset, llm


def test_splits_are_disjoint_and_test_uses_human_labels() -> None:
    dataset, llm = _data()
    human = pd.DataFrame({"id": dataset["id"][:40], "sentiment": ["neutral"] * 38 + ["skip"] * 2})
    train, val, test, _ = make_splits(dataset, llm, human, set(human["id"]), 0.2, 0)

    assert len(test) == 38
    assert set(test["label_source"]) == {"human"}
    tr, va, te = set(train["id"]), set(val["id"]), set(test["id"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    skipped = set(dataset["id"][38:40])
    assert not skipped & (tr | va | te)  # skipped sample items are used nowhere


def test_last_human_answer_wins() -> None:
    dataset, llm = _data()
    human = pd.DataFrame({"id": ["id0000", "id0000"], "sentiment": ["positive", "negative"]})
    _, _, test, _ = make_splits(dataset, llm, human, {"id0000"}, 0.2, 0)
    assert test.loc[0, "sentiment"] == "negative"


def test_duplicate_of_test_text_is_removed_from_training() -> None:
    dataset, llm = _data()
    dataset.loc[100, "text"] = dataset.loc[0, "text"].upper() + "!"
    human = pd.DataFrame({"id": ["id0000"], "sentiment": ["neutral"]})
    train, val, _, n_leaked = make_splits(dataset, llm, human, {"id0000"}, 0.2, 0)
    assert n_leaked == 1
    assert "id0100" not in set(train["id"]) | set(val["id"])


def test_fingerprint_ignores_order_but_detects_label_changes() -> None:
    a = pd.DataFrame({"id": ["x", "y"], "sentiment": ["positive", "neutral"]})
    b = a.assign(sentiment=["negative", "neutral"])
    assert fingerprint_test_set(a) == fingerprint_test_set(a.iloc[::-1])
    assert fingerprint_test_set(a) != fingerprint_test_set(b)
