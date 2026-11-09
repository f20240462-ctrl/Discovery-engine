"""
SIP Continuation Discovery Engine (Groww) - Streamlit app.
Tabs: Try it (live extraction) | Findings (themes, cuts, ranking) | Evidence (funnel, golden set, corrections)
Needs in the same folder: extract_insights.py (same prompt as the batch run), data/extracted.csv
"""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SIP Discovery Engine", layout="wide")

try:
    from extract_insights import MODEL, PROMPT, SCHEMA, apply_flags, quote_in_text
except Exception as e:
    st.error(f"Could not load extract_insights.py: {e}")
    st.stop()

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_TRIES = 8
DROP = ["paused", "reduced", "stopped"]
BRANCH = {
    "mandate_failure": "Involuntary lapse", "cash_flow": "Voluntary exit: cash flow",
    "expectation_mismatch": "Voluntary exit: expectation mismatch", "uncertainty": "Voluntary exit: uncertainty",
    "purpose_gap": "Voluntary exit: purpose gap", "social_influence": "Voluntary exit: social influence",
    "fatigue": "Voluntary exit: fatigue", "app_friction": "Product friction", "other": "Other",
}
# Judgment assumptions (0 to 1): how solvable is this within the brief's constraints. Editable in the app.
ADDR = {"mandate_failure": 0.9, "cash_flow": 0.6, "expectation_mismatch": 0.8, "uncertainty": 0.9,
        "purpose_gap": 0.8, "social_influence": 0.5, "fatigue": 0.7, "app_friction": 0.7, "other": 0.3}
EXAMPLES = {
    "Pause question": "I started a SIP six months ago and my returns are still red. I'm not sure if this is normal, so I'm thinking of pausing it until I understand more.",
    "Autopay failure": "My SIP instalment failed twice because the autopay mandate expired and nobody told me. I only found out a month later.",
    "Not about SIPs": "Great app for trading, intraday charts are smooth and stop loss orders work well.",
}


def tobool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


@st.cache_data
def load(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for c in ["relevant", "sip_specific", "needs_human_review", "quote_verified"]:
        if c in df:
            df[c] = tobool(df[c])
    if "item_id" not in df and {"source", "id"} <= set(df):
        df["item_id"] = df["source"] + "::" + df["id"].astype(str)
    return df


def get_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


@st.cache_data(show_spinner=False)
def analyze(text, _key):
    from google import genai
    client = genai.Client(api_key=_key)
    items = [{"item_id": "paste", "source": "user_paste", "text": text[:1500]}]
    prompt = PROMPT.format(schema=SCHEMA, items=json.dumps(items, ensure_ascii=False))
    resp = client.models.generate_content(
        model=MODEL, contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0})
    rec = json.loads(resp.text)[0]
    q = rec.get("quote")
    ok = quote_in_text(q, text) if q else None
    rec["quote_verified"] = ok
    apply_flags(rec, quote_ok=ok)
    return rec


st.title("SIP Continuation Discovery Engine")
st.caption("Research prototype built on public Play Store reviews, YouTube comments and Reddit threads about SIPs. "
           "Not investment advice. Please do not paste personal or account details.")
t1, t2, t3 = st.tabs(["Try it", "Findings", "Evidence"])

# ---------------------------------------------------------------- Try it
with t1:
    st.write("Paste a review or comment about SIPs. The engine extracts structured fields, then applies quality checks. "
             "If it is unsure, it says so and asks for human review instead of guessing.")
    pick = st.selectbox("Or start from an example", ["(none)"] + list(EXAMPLES))
    text = st.text_area("Text to analyse", value=EXAMPLES.get(pick, ""), height=140, max_chars=1500)
    st.session_state.setdefault("tries", 0)
    if st.button("Analyse", type="primary") and text.strip():
        key = get_key()
        if not key:
            st.error("No API key configured on the server.")
        elif st.session_state.tries >= MAX_TRIES:
            st.warning("Demo limit reached for this session. Please refresh the page to continue.")
        else:
            try:
                st.session_state.tries += 1
                with st.spinner("Reading..."):
                    rec = analyze(text.strip(), key)
                conf = rec.get("confidence") or 0
                if rec.get("relevant") is False:
                    st.info("Not about SIP continuation or first-time investing, so no themes were extracted.")
                elif rec.get("needs_human_review") or conf < 0.6:
                    st.warning("Low confidence or quality flags raised: needs human review. Theme is left unassigned. "
                               + (f"Reason: {rec.get('review_reasons')}" if rec.get("review_reasons") else ""))
                else:
                    st.success(f"Passed quality checks (confidence {conf:.2f}, quote verified: {rec.get('quote_verified')})")
                show = ["sip_specific", "first_timer_signal", "journey_stage", "months_since_start", "outcome", "behavior",
                        "stated_reason", "underlying_reason", "reason_category", "what_they_check", "confusion_point",
                        "outside_workaround", "would_have_helped", "segment_income", "segment_money_source",
                        "segment_influencer", "quote", "confidence"]
                st.dataframe(pd.DataFrame({"field": show, "value": [str(rec.get(k)) for k in show]}),
                             hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"The model is busy or returned something unusable. Try again in a minute. ({str(e)[:100]})")
    st.caption(f"Model: {MODEL}. Same prompt and quality checks as the batch run behind the Findings tab.")

# -------------------------------------------------------------- Findings
with t2:
    df = load(f"{DATA}/extracted.csv")
    if df is None:
        st.warning("data/extracted.csv not found.")
        st.stop()
    rel = df[df["relevant"] == True]
    sip = rel[rel["sip_specific"] == True] if "sip_specific" in rel else rel
    c = st.columns(4)
    c[0].metric("Items analysed", len(df))
    c[1].metric("Relevant", len(rel))
    c[2].metric("SIP specific", len(sip))
    c[3].metric("Flagged for review", f"{(df['needs_human_review'] == True).mean():.0%}")
    view = st.radio("Analyse", ["Already started a SIP", "Already started, first-timers only", "All SIP specific", "All relevant"], horizontal=True)
    started = rel["journey_stage"].isin(["month_1_3", "month_4_12", "after_12", "paused_or_stopped"]) | rel["outcome"].isin(["continuing", "reduced", "paused", "stopped", "switched"])
    if view == "All relevant":
        d = rel
    elif view == "SIP specific":
        d = sip
    elif view == "Already started a SIP":
        d = sip[started.loc[sip.index]]
    else:
        d = sip[started.loc[sip.index] & sip["first_timer_signal"].isin(["yes", "likely"])]
    srcs = st.multiselect("Sources", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
    d = d[d["source"].isin(srcs)]
    st.write("Source mix:", d["source"].value_counts().to_dict())

    themed = d[d["reason_category"].notna() & (d["reason_category"] != "unclear")]
    if themed.empty:
        st.info("No themed items in this view yet.")
        st.stop()

    st.subheader("Themes and opportunity ranking")
    g = themed.assign(drop=themed["outcome"].isin(DROP), clean=~themed["needs_human_review"].fillna(False)).groupby(
        "reason_category").agg(items=("item_id", "count"), drop_rate=("drop", "mean"), sources=("source", "nunique"),
                               clean=("clean", "mean"), confidence=("confidence", "mean")).reset_index()
    g["share"] = g["items"] / g["items"].sum()
    with st.expander("Addressability assumptions (judgment, adjust to test sensitivity)"):
        addr = {k: st.slider(k, 0.0, 1.0, ADDR.get(k, 0.3), 0.05) for k in g["reason_category"]}
    g["addressability"] = g["reason_category"].map(addr)
    g["score"] = g["share"] * (g["drop_rate"] + 0.05) * g["clean"] * (g["sources"].clip(upper=3) / 3) * g["addressability"]
    g["score"] = (100 * g["score"] / g["score"].max()).round(0)
    g["metric_branch"] = g["reason_category"].map(BRANCH).fillna("Other")
    g["note"] = g["items"].map(lambda n: "low n, treat as a hypothesis" if n < 5 else "")
    out = g.sort_values("score", ascending=False)[["reason_category", "metric_branch", "items", "share", "drop_rate",
                                                   "sources", "clean", "addressability", "score", "note"]]
    st.dataframe(out.style.format({"share": "{:.0%}", "drop_rate": "{:.0%}", "clean": "{:.0%}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Score = share of themed items x drop rate (paused, reduced or stopped) x share passing quality checks "
               "x source breadth (up to 3) x addressability. Directional only, not a forecast.")

    st.subheader("Example quotes (verified, unflagged)")
    good = themed[(themed["quote_verified"] == True) & (~themed["needs_human_review"].fillna(False))]
    for theme in out["reason_category"]:
        rows = good[good["reason_category"] == theme].head(3)
        with st.expander(f"{theme} ({int(out.set_index('reason_category').loc[theme, 'items'])} items)"):
            for _, r in rows.iterrows():
                st.markdown(f"> {r['quote']}  \n*{r['source']}* [link]({r['url']})")
            if rows.empty:
                st.write("No verified quotes yet.")

    st.subheader("Cuts")
    cut = st.selectbox("Split themes by", [x for x in ["journey_stage", "outcome", "segment_income",
                       "segment_money_source", "segment_influencer", "segment_age_band", "source"] if x in themed])
    st.dataframe(pd.crosstab(themed[cut].fillna("unclear"), themed["reason_category"]), use_container_width=True)
    st.caption("Segment fields stay 'unclear' unless the text says so. Large 'unclear' counts are a real limit of public data.")

    st.subheader("Continuers versus droppers")
    grp = d.assign(group=d["outcome"].map(lambda o: "dropped (paused, reduced, stopped)" if o in DROP
                                          else ("continuing" if o == "continuing" else "unclear")))
    st.dataframe(pd.crosstab(grp["reason_category"].fillna("none"), grp["group"]), use_container_width=True)
    for col, label in [("confusion_point", "What confuses or worries people"), ("would_have_helped", "What people say would have helped")]:
        sub = d[d[col].notna()][[col, "outcome", "source"]].head(25)
        st.markdown(f"**{label}**")
        st.dataframe(sub, hide_index=True, use_container_width=True)

# -------------------------------------------------------------- Evidence
with t3:
    ev = f"{DATA}/evidence.json"
    if os.path.exists(ev):
        e = json.load(open(ev, encoding="utf-8"))
        st.subheader("Sources and volume")
        st.dataframe(pd.DataFrame(e.get("funnel", [])), hide_index=True, use_container_width=True)
        for n in e.get("notes", []):
            st.write("- " + n)
    st.subheader("Golden set agreement")
    gold = load(f"{DATA}/golden_labeled.csv")
    base = load(f"{DATA}/extracted.csv")
    if gold is None or base is None:
        st.info("Golden set not loaded yet (data/golden_labeled.csv).")
    else:
        fields = [c[5:] for c in gold.columns if c.startswith("gold_") and c[5:] in base.columns]
        m = gold[["item_id"] + ["gold_" + f for f in fields]].merge(base[["item_id"] + fields], on="item_id")
        res = []
        for f in fields:
            sub = m[m["gold_" + f].notna()]
            agree = (sub["gold_" + f].astype(str).str.lower().str.strip() == sub[f].astype(str).str.lower().str.strip())
            res.append({"field": f, "labelled": len(sub), "agreement": agree.mean() if len(sub) else None})
        st.dataframe(pd.DataFrame(res).style.format({"agreement": "{:.0%}"}), hide_index=True, use_container_width=True)
        st.caption(f"{len(m)} hand-labelled items compared with the engine's output.")
    st.subheader("Where I corrected or rejected the AI")
    p = f"{DATA}/corrections.md"
    st.markdown(open(p, encoding="utf-8").read() if os.path.exists(p) else "Add data/corrections.md.")
