"""
Step 2 of the discovery engine: LLM extraction with the Gemini API.

Setup:
    pip install google-genai pandas python-dotenv
    .env file:
        GEMINI_API_KEY=your_key_from_aistudio.google.com
        GEMINI_MODEL=gemini-3.5-flash-lite      # just the model name, nothing else

Run:
    python extract_insights.py                 # full run on filtered_reviews.csv
    python extract_insights.py --limit 30      # quick test on 30 items

Input : filtered_reviews.csv (from collect_reviews.py)
Output: extracted.jsonl (one JSON record per item, saved as it goes)
        extracted.csv   (flat table for analysis)

The script resumes automatically: items already in extracted.jsonl are skipped.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()

# Avoid UnicodeEncodeError when printing emojis / Hindi on Windows terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
BATCH_SIZE = 10
SLEEP_BETWEEN_CALLS = 6      # seconds; raise this if you hit rate limits
MAX_RETRIES = 5
RETRYABLE = {429, 500, 502, 503, 504}

SCHEMA = """
Each record must have exactly these keys:
- item_id: the id given in the input (string)
- relevant: true/false. true only if the text is about starting, continuing, pausing, reducing or stopping a SIP, or about first-time investing
- first_timer_signal: "yes" | "likely" | "no" | "unclear"
- journey_stage: "pre_start" | "month_1_3" | "month_4_12" | "after_12" | "paused_or_stopped" | "unclear"
- months_since_start: number or null (only if stated or clearly implied)
- outcome: "continuing" | "reduced" | "paused" | "stopped" | "switched" | "unclear"
- why_started: short text or null
- expectation: short text or null
- behavior: "continue" | "reduce" | "pause" | "stop" | "switch" | "check_often" | "restart" | "none" | "unclear"
- stated_reason: short text or null
- underlying_reason: short text or null (only if the text explains why the stated reason mattered)
- reason_category: "mandate_failure" | "cash_flow" | "expectation_mismatch" | "uncertainty" | "purpose_gap" | "social_influence" | "fatigue" | "app_friction" | "other" | "unclear"
- what_they_check: list of strings or null (NAV, returns, XIRR, balance, news, etc.)
- check_frequency: "daily" | "weekly" | "monthly" | "rarely" | null
- confusion_point: short text or null
- in_app_or_outside: "in_app" | "outside" | "both" | null
- outside_workaround: short text or null (YouTube, friend, Reddit, spreadsheet, another app, etc.)
- would_have_helped: short text or null (what the person says they wanted or needed)
- segment_income: "steady" | "irregular" | "student" | "unclear"
- segment_age_band: "under_21" | "21_24" | "25_28" | "29_35" | "over_35" | "unclear"
- segment_money_source: "salary" | "stipend_or_pocket_money" | "freelance_irregular" | "family" | "unclear"
- segment_influencer: "self" | "friend_family" | "youtube_social" | "workplace" | "unclear"
- quote: exact words copied from the text, under 25 words, or null
- confidence: number between 0 and 1
- needs_human_review: true/false
- sip_specific: true/false. true only if the text is about a SIP specifically, not lump sum, trading, IPOs or app problems in general
"""

PROMPT = """You are a research analyst extracting structured insights from public
conversations about SIPs and first-time mutual fund investing in India.
Texts may be in English, Hindi, or Hinglish.

You will receive a JSON list of items. Return ONLY a JSON list with one record
per item, in the same order. No commentary, no markdown.

Rules:
1. Use only what the text says. If a field is not supported, return null or "unclear". Never guess.
2. Keep stated_reason and underlying_reason separate. Fill underlying_reason only if the text
   explains why the stated reason mattered to the person. Do not infer deeper causes yourself.
3. Do not treat "the market fell" as a full explanation. Capture what the person felt, feared or
   did because of it, if stated.
4. Segment fields must come from explicit or very clear cues in the text.
5. quote must be copied exactly from the text and be under 25 words.
6. If the text is sarcastic, mixed-language in a way you are unsure of, or your confidence is
   below 0.6, set needs_human_review to true.
7. If the text is not about SIPs or first-time investing, set relevant to false, set other
   fields to null or "unclear", and keep confidence as your confidence in that judgement.
8. Do not recommend funds, stocks or actions.
9. journey_stage and months_since_start only if the text gives explicit timing. Otherwise "unclear".
10. reason_category "uncertainty" only if the person says they were confused, unsure or unable to judge. "Market fluctuations" alone is "other".
11. outcome "paused" or "stopped" only if it refers to investing or a SIP, not uninstalling the app.

{schema}

Items:
{items}
"""


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["item_id"])
                except Exception:
                    pass
    return done


def call_model(client, batch):
    items = [
        {"item_id": r["item_id"], "source": r["source"], "text": str(r["text"])[:1500]}
        for r in batch
    ]
    prompt = PROMPT.format(schema=SCHEMA, items=json.dumps(items, ensure_ascii=False))
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 0},
            )
            data = json.loads(resp.text)
            if isinstance(data, list):
                return data
            print("  model returned non-list JSON, retrying")
        except genai_errors.APIError as e:
            if e.code not in RETRYABLE:
                raise  # 400/403/404 etc. will never succeed, so fail fast
            print(f"  retry {attempt}/{MAX_RETRIES} after error {e.code}")
        except (json.JSONDecodeError, TypeError):
            print("  bad or empty JSON from model, retrying")
        if attempt < MAX_RETRIES:
            wait = min(15 * 2 ** (attempt - 1), 120)
            print(f"  waiting {wait}s")
            time.sleep(wait)
    return []


def quote_in_text(quote, text):
    """True if the quote appears in the source text (whitespace-insensitive)."""
    return " ".join(str(quote).split()) in " ".join(str(text).split())

def apply_flags(rec, quote_ok=None):
    """Rule-based review flags, so quality checks don't rely on the model's own confidence."""
    if rec.get("relevant") is False:
        rec["review_reasons"] = ""
        return

    why = []
    if rec.get("reason_category") not in (None, "unclear", "other") and not rec.get("stated_reason"):
        why.append("reason category without a stated reason")
    if rec.get("journey_stage") in ("month_1_3", "month_4_12", "after_12") and rec.get("months_since_start") is None:
        why.append("stage given without timing")
    if rec.get("sip_specific") is False and rec.get("outcome") in ("paused", "stopped"):
        why.append("pause or stop but not about a SIP")
    if rec.get("reason_category") == "uncertainty" and not rec.get("confusion_point") and not rec.get("underlying_reason"):
        why.append("uncertainty with no confusion described")
    if quote_ok is False:
        why.append("quote not found in source text")
    if rec.get("needs_human_review") and not why:
        why.append("flagged by model")

    if why:
        rec["needs_human_review"] = True
    rec["review_reasons"] = "; ".join(why)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--input", default="filtered_reviews.csv")
    ap.add_argument("--output", default="extracted.jsonl")
    args = ap.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    df = pd.read_csv(args.input, encoding="utf-8")
    df["item_id"] = df["source"] + "::" + df["id"].astype(str)
    if args.limit:
        df = df.head(args.limit)

    done = load_done(args.output)
    todo = df[~df["item_id"].isin(done)].to_dict("records")
    print(f"{len(done)} already done, {len(todo)} to process, model={MODEL}")

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        results = call_model(client, batch)
        by_id = {r.get("item_id"): r for r in results if isinstance(r, dict)}

        saved = 0
        with open(args.output, "a", encoding="utf-8") as f:
            for row in batch:
                rec = by_id.get(row["item_id"])
                if rec is None:
                    continue    # failed items are retried on the next run
                rec["source"] = row["source"]
                rec["date"] = row["date"]
                rec["rating"] = row["rating"]
                rec["url"] = row["url"]
                rec["text"] = row["text"]
                rec["model_used"] = MODEL

                q = rec.get("quote")
                if q:
                    ok = quote_in_text(q, row["text"])
                    rec["quote_verified"] = ok
                else:
                    ok = None
                    rec["quote_verified"] = None

                apply_flags(rec, quote_ok=ok)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                saved += 1

        print(f"batch {i // BATCH_SIZE + 1}: saved {saved}/{len(batch)}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    with open(args.output, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    pd.DataFrame(recs).to_csv("extracted.csv", index=False, encoding="utf-8-sig")
    print(f"Done. {len(recs)} records in extracted.csv")


if __name__ == "__main__":
    main()