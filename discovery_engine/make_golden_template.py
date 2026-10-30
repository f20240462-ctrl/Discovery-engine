"""Sample 20 items for hand labelling. Use a different sample from the 30 items you tuned the prompt on."""
import pandas as pd

df = pd.read_csv("data/extracted.csv")
if "item_id" not in df:
    df["item_id"] = df["source"] + "::" + df["id"].astype(str)
rel = df[df["relevant"].astype(str).str.lower() == "true"]
non = df[df["relevant"].astype(str).str.lower() == "false"]
parts = [
    rel[rel["source"] == "youtube_comment"].sample(7, random_state=7),
    rel[rel["source"].isin(["play_store", "app_store"])].sample(5, random_state=7),
    rel[rel["source"].str.startswith("reddit")].sample(3, random_state=7),
    non.sample(5, random_state=7),
]
pick = pd.concat(parts)
out = pick[["item_id", "source", "text", "url"]].copy()
for f in ["relevant", "sip_specific", "reason_category", "outcome", "journey_stage"]:
    out["gold_" + f] = ""
out.sample(frac=1, random_state=1).to_csv("golden_template.csv", index=False, encoding="utf-8-sig")
print("Wrote golden_template.csv. Fill the gold_ columns by reading each text yourself, then save as data/golden_labeled.csv")
