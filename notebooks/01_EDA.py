# %% 1. Class balance per split
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / "configs").exists() else Path.cwd().parent
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

splits = {name: pd.read_parquet(DATA / f"{name}.parquet") for name in ["train", "val", "test"]}
df = pd.concat([d.assign(split=name) for name, d in splits.items()], ignore_index=True)
llm = pd.read_json(ROOT / "data/interim/llm_labels.jsonl", lines=True, dtype={"id": str})
df = df.merge(llm[["id", "event_type"]], on="id", how="left")
print(df.shape)
print(df["split"].value_counts().to_string())


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.show()


counts = pd.crosstab(df["sentiment"], df["split"], normalize="columns")[["train", "val", "test"]]
print(counts.round(3).to_string())
counts.plot.bar(rot=0, title="Label distribution by split")
plt.ylabel("share of rows")
save("class_balance")

# %% 2. Text length in words: decides max sequence length for neural models later
df["n_words"] = df["text"].str.split().str.len()
print(df.groupby("text_source")["n_words"].describe().round(0).to_string())
print("\n95th percentile:", int(df["n_words"].quantile(0.95)), "words")
df["n_words"].plot.hist(bins=50, title="Words per text")
plt.xlabel("words")
save("text_lengths")

# %% 3. Text length by label: are some labels systematically longer?
df.boxplot(column="n_words", by="sentiment")
plt.suptitle("")
plt.title("Words per text by label")
save("length_by_label")

# %% 4. Where did the text come from?
print(pd.crosstab(df["text_source"], df["sentiment"], margins=True).to_string())
pd.crosstab(df["text_source"], df["sentiment"], normalize="index").plot.barh(
    stacked=True, title="Label mix by text source"
)
save("label_by_text_source")

# %% 5. Event type vs label: which kinds of announcements carry good or bad news?
table = pd.crosstab(df["event_type"], df["sentiment"])
table = table.loc[table.sum(axis=1).sort_values().index]
print(table.to_string())
table.plot.barh(stacked=True, title="Labels by event type", figsize=(8, 5))
save("label_by_event_type")

# %% 6. Over time
df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time
weekly = pd.crosstab(df["week"], df["sentiment"])
weekly.plot(title="Items per week by label", figsize=(9, 4))
save("over_time")

# %% 7. Most frequent companies
print(df.loc[df["source"] == "psx", "symbol"].value_counts().head(15).to_string())

# %% 8. Data quality checks
garbled = df["text"].str.contains(r"/MT\d", regex=True)
print("Garbled font-encoded texts (/MT..):", int(garbled.sum()))
print("Duplicate texts within train:", int(splits["train"]["text"].duplicated().sum()))
print("Label sources:", df["label_source"].value_counts().to_dict())
# %%
