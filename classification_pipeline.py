# -*- coding: utf-8 -*-
"""
Mental Health Tweet Classification Pipeline
============================================
Stage 1 : Per-tweet classification — LLM sees bio + tweet together in one call.
           Each model annotates the same DataFrame in sequence.
           One label column per model, side by side.
Stage 2 : User-level aggregation — deterministic Python logic, per model.

Bio override rule: if ANY tweet for a user has BIO_PATIENT_SELF_IDENTIFICATION
in its reason_tags, the user is immediately classified as possible_patient.

Usage (rerun mode — re-parses existing LLM outputs, no new API calls):
    python classification_pipeline.py \
        --results results_tweets.csv \
        --output  results_rerun.csv

Usage (fresh annotation — runs both GPT and Qwen):
    python classification_pipeline.py \
        --mode fresh \
        --input tweets_preprocessed.csv \
        --output results_rerun.csv

Environment variables required:
    OPENAI_API_KEY      OpenAI API key (required for GPT-4.1)
    DASHSCOPE_API_KEY   Alibaba DashScope API key (required for Qwen3-235B)

Python >= 3.8 required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import pandas as pd
from openai import OpenAI

# =============================================================================
# SECTION 1: Configuration (overridable via CLI — see parse_args())
# =============================================================================

# Default paths — override with CLI arguments in production.
INPUT_PATH         = "tweets_preprocessed.csv"
RESULTS_INPUT_PATH = "results_tweets.csv"
OUTPUT_PATH        = "results_rerun.csv"
CHECKPOINT_PATH    = "checkpoint.csv"

# Checkpoint cadence
CHECKPOINT_EVERY = 10

# Inference settings
MAX_RETRIES    = 3
RETRY_DELAY    = 60       # base delay in seconds (exponential backoff)
MAX_NEW_TOKENS = 250

# Model identifiers
# These strings must match the API exactly.
# GPT-4.1:  https://platform.openai.com/docs/models
# Qwen3:    https://dashscope.aliyuncs.com — verify the identifier remains current
#           before each run, as DashScope model names are subject to revision.
MODEL_GPT  = "gpt-4.1"
MODEL_QWEN = "qwen3-235b-a22b-instruct-2507"

ALL_MODEL_COLS = [MODEL_GPT, MODEL_QWEN]

# Model registry
MODELS = {
    MODEL_GPT: {
        "model":      MODEL_GPT,
        "extra_body": {},
    },
    MODEL_QWEN: {
        "model":      MODEL_QWEN,
        "base_url":   "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "extra_body": {"enable_thinking": False},
    },
}

# =============================================================================
# SECTION 2: System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are an expert classifier working with Arabic-language social media data collected from focused mental health communities on X (Twitter). These communities focus on ADHD, Bipolar Disorder, and BPD (Borderline Personality Disorder).

Your task is to classify a single tweet as either:
- "positive" — the tweet contains patient-level signals: the author appears to be personally living with or experiencing a mental health condition
- "negative" — the tweet contains no patient-level signals: the content is educational, professional, neutral, or irrelevant

You are classifying the tweet only. User-level decisions are handled separately.

## INPUT FORMAT
{
  "community": "ADHD | BPD | bipolar",
  "user_bio": "...",
  "tweet_text": "..."
}

Use BOTH the bio and the tweet together to make your decision.

BIO OVERRIDE RULE:
If the bio clearly identifies the user as a patient — explicit diagnosis, living-with language, or condition as personal identity (e.g., "تم تشخيصي بـ ADHD", "أعيش مع ثنائي القطب", "#BPD", "bipolar girl") — then:
  * Set tweet_label to "positive"
  * Add BIO_PATIENT_SELF_IDENTIFICATION to reason_tags
  * This applies even if the tweet itself contains no patient signal

If the bio indicates a professional or institutional account, this supports "negative" — but does NOT override a clearly personal tweet.
If the bio is empty, rely on the tweet alone and add EDGE_EMPTY_BIO to reason_tags.

## CLASSIFICATION RULES

### Classify as "positive" if ANY of the following are true:
- First-person account of experiencing symptoms (e.g., "أعاني من تشتت", "نفسيتي مدمرة", "ما أقدر أنام")
- Disclosing a personal diagnosis (e.g., "تم تشخيصي بـ ADHD", "عندي ثنائي القطب")
- Sharing personal emotional distress or struggles related to a mental health condition
- Seeking peer support or venting about daily life with a condition
- Asking whether a personally experienced symptom belongs to a condition (first-person question)
- Expressing experience from the inside — not explaining or educating others about a condition

### Classify as "negative" if ANY of the following are true:
- Educational or psychoeducational content explaining symptoms or treatments in third-person
- Written as a professional advising or answering someone else's question
- Promoting a therapy session, app, webinar, course, book, or product
- Discussing research findings, clinical definitions, or diagnostic criteria
- Addressing patients as a separate audience (e.g., "هؤلاء الأشخاص يحتاجون...")
- Spam, off-topic, or completely irrelevant content

DEFAULT RULE: When in doubt, label "negative".

## OUTPUT FORMAT
Return ONLY a valid JSON object with no extra text, no markdown fences:
{
  "tweet_label": "positive | negative",
  "confidence": "high | medium | low",
  "reason_tags": ["TAG_1", "TAG_2"]
}

## REASON TAG TAXONOMY

Patient signal tags (support "positive"):
- TWEET_SYMPTOM_DISCLOSURE            : Tweet describes experiencing symptoms in first person
- TWEET_PERSONAL_DIAGNOSIS_DISCLOSURE : Tweet explicitly states the user was diagnosed
- TWEET_PEER_SUPPORT_SEEKING          : Tweet seeks support or validation from others with the condition
- TWEET_EMOTIONAL_VENTING             : Tweet expresses raw personal emotion or distress without educational intent
- TWEET_FIRST_PERSON_SYMPTOM_QUESTION : Tweet asks whether a personally experienced symptom belongs to a condition

Non-patient signal tags (support "negative"):
- TWEET_EDUCATIONAL_CONTENT           : Tweet explains symptoms, conditions, or treatment in informational third-person style
- TWEET_PROFESSIONAL_ADVICE           : Tweet offers clinical guidance or answers someone else's question professionally
- TWEET_SERVICE_PROMOTION             : Tweet promotes a therapy session, app, webinar, course, or mental health product
- TWEET_RESEARCH_OR_CLINICAL          : Tweet discusses diagnostic criteria, research findings, or clinical definitions
- TWEET_THIRD_PERSON_FRAMING          : Tweet addresses patients as a separate audience
- TWEET_SPAM_OR_IRRELEVANT            : Tweet is off-topic, spam, or unrelated to mental health

Bio signal tags (always added when detected, regardless of tweet label):
- BIO_PATIENT_SELF_IDENTIFICATION     : Bio clearly identifies the user as personally living with or diagnosed with a condition

Edge case tags:
- EDGE_EMPTY_BIO                      : Bio is absent; classification relies entirely on tweet content
- EDGE_AMBIGUOUS_FIRST_PERSON         : Tweet could be personal or professional; classified based on best available signal
- EDGE_PROFESSIONAL_BIO_PERSONAL_TWEET: Bio suggests a professional but tweet content is clearly personal/experiential

## CONFIDENCE LEVELS
- high   : Strong unambiguous signal (e.g., explicit diagnosis disclosure, clear third-person educational content)
- medium : Signal present but indirect or requires inference
- low    : Very weak or contradictory signals; classification is a best guess

## EXAMPLES (fully synthetic)

Input:  {"community": "BPD", "user_bio": "إنسانة تتعلم كيف تعيش مع اضطراب الشخصية الحدية يومًا بيوم 🖤 | #BPD", "tweet_text": "أصعب شي في الحدية إنك تحب بشكل كامل وفجأة تحس إن كل شي انهار بدون سبب واضح"}
Output: {"tweet_label": "positive", "confidence": "high", "reason_tags": ["TWEET_EMOTIONAL_VENTING", "BIO_PATIENT_SELF_IDENTIFICATION"]}

Input:  {"community": "ADHD", "user_bio": "أخصائي نفسي إكلينيكي | ماجستير إرشاد نفسي | مرخص من هيئة التخصصات الصحية", "tweet_text": "الفرق بين فرط الحركة عند الأطفال والبالغين: الأطفال يُظهرون أعراضًا حركية واضحة، بينما يعاني البالغون من أعراض داخلية كالقلق الذهني وصعوبة التنظيم."}
Output: {"tweet_label": "negative", "confidence": "high", "reason_tags": ["TWEET_EDUCATIONAL_CONTENT", "TWEET_THIRD_PERSON_FRAMING"]}

Input:  {"community": "bipolar", "user_bio": "", "tweet_text": "أحس هالأيام بطاقة زايدة عن اللزوم، ما أنام، وعندي رغبة أشتري أشياء ما أحتاجها — هل هذا طبيعي ولا ممكن يكون هوس؟"}
Output: {"tweet_label": "positive", "confidence": "high", "reason_tags": ["TWEET_SYMPTOM_DISCLOSURE", "TWEET_FIRST_PERSON_SYMPTOM_QUESTION", "EDGE_EMPTY_BIO"]}

Input:  {"community": "BPD", "user_bio": "معالج نفسي معتمد | متخصص في اضطرابات الشخصية | باحث في العلاج الجدلي السلوكي DBT", "tweet_text": "أحيانًا الشفاء لا يبدو كالشفاء — بل يبدو كلحظة هدوء صغيرة وسط العاصفة. أتمنى لكم تلك اللحظة 💙"}
Output: {"tweet_label": "negative", "confidence": "medium", "reason_tags": ["TWEET_THIRD_PERSON_FRAMING", "EDGE_AMBIGUOUS_FIRST_PERSON"]}

Input:  {"community": "ADHD", "user_bio": "مبرمج ومهتم بالتقنية | تم تشخيصي بـ ADHD منذ سنتين", "tweet_text": "الفرق بين فرط الحركة عند الأطفال والبالغين من وجهة نظر علمية"}
Output: {"tweet_label": "positive", "confidence": "high", "reason_tags": ["TWEET_EDUCATIONAL_CONTENT", "BIO_PATIENT_SELF_IDENTIFICATION"]}

Input:  {"community": "bipolar", "user_bio": "طالبة دكتوراه علم نفس 📚 | أعيش مع ثنائي القطب وأحاول أفهمه من الداخل والخارج", "tweet_text": "لما تكون في نوبة اكتئاب وتعرف نظريًا كل الأدوات العلاجية بس ما تقدر تطبق ولو واحدة — هذا تناقض ما يفهمه غير اللي عاشه"}
Output: {"tweet_label": "positive", "confidence": "medium", "reason_tags": ["TWEET_EMOTIONAL_VENTING", "TWEET_SYMPTOM_DISCLOSURE", "BIO_PATIENT_SELF_IDENTIFICATION", "EDGE_PROFESSIONAL_BIO_PERSONAL_TWEET"]}

## IMPORTANT NOTES
- This dataset is primarily in Arabic (Modern Standard and Gulf/Saudi dialect). Be sensitive to dialectal expressions of distress (e.g., "مو زين", "نفسيتي مدمرة", "قرفانة من كل شي").
- Do NOT base the classification solely on the community tag — professionals, researchers, and caregivers are present in all three communities.
- If the bio clearly identifies the user as a patient, always return tweet_label "positive" and add BIO_PATIENT_SELF_IDENTIFICATION — even if the tweet itself is educational or neutral.
- A professional bio does NOT override a clearly personal tweet — classify such cases as "positive" and add EDGE_PROFESSIONAL_BIO_PERSONAL_TWEET.
- This classification is for research purposes. Handle all data with care and do not make clinical inferences beyond the binary label requested.
"""

# =============================================================================
# SECTION 3: API Clients
# =============================================================================

def _build_clients() -> tuple:
    """
    Initialise API clients from environment variables.
    Raises RuntimeError if a required key is missing.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")

    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Export it before running: export OPENAI_API_KEY=<your_key>"
        )
    if not dashscope_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY environment variable is not set. "
            "Export it before running: export DASHSCOPE_API_KEY=<your_key>"
        )

    gpt_client = OpenAI(api_key=openai_key)
    qwen_client = OpenAI(
        api_key  = dashscope_key,
        base_url = MODELS[MODEL_QWEN]["base_url"],
    )
    return gpt_client, qwen_client


def _get_client(model_name: str, gpt_client: OpenAI, qwen_client: OpenAI):
    """Return the correct client, model string, and extra_body for a given model."""
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )
    config = MODELS[model_name]
    client = gpt_client if model_name == MODEL_GPT else qwen_client
    return client, config["model"], config.get("extra_body", {})

# =============================================================================
# SECTION 4: Parser
# =============================================================================

VALID_LABELS = {"positive", "negative"}
VALID_REASON_TAGS = {
    "TWEET_SYMPTOM_DISCLOSURE",
    "TWEET_PERSONAL_DIAGNOSIS_DISCLOSURE",
    "TWEET_PEER_SUPPORT_SEEKING",
    "TWEET_EMOTIONAL_VENTING",
    "TWEET_FIRST_PERSON_SYMPTOM_QUESTION",
    "TWEET_EDUCATIONAL_CONTENT",
    "TWEET_PROFESSIONAL_ADVICE",
    "TWEET_SERVICE_PROMOTION",
    "TWEET_RESEARCH_OR_CLINICAL",
    "TWEET_THIRD_PERSON_FRAMING",
    "TWEET_SPAM_OR_IRRELEVANT",
    "BIO_PATIENT_SELF_IDENTIFICATION",
    "EDGE_EMPTY_BIO",
    "EDGE_AMBIGUOUS_FIRST_PERSON",
    "EDGE_PROFESSIONAL_BIO_PERSONAL_TWEET",
}


def parse_output(raw_text: str) -> dict:
    """
    Parse raw LLM output into structured fields.

    Returns a dict with:
      tweet_label   : "positive" | "negative" | "PARSE_FAIL"
      confidence    : "high" | "medium" | "low" | ""
      reason_tags   : pipe-separated valid tags, or "PARSE_FAIL"
      parse_status  : "ok" | "partial" | "fail"
        - ok      : both label and reason_tags parsed successfully
        - partial : label parsed but reason_tags missing or all invalid
        - fail    : label could not be parsed
    """
    if not raw_text or not raw_text.strip() or str(raw_text).startswith("API_ERROR"):
        return {
            "tweet_label":  "PARSE_FAIL",
            "confidence":   "",
            "reason_tags":  "PARSE_FAIL",
            "parse_status": "fail",
        }

    raw = str(raw_text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Initialise data so references outside the try block are always safe.
    data = {}

    # --- Parse label ---
    label = None
    try:
        data  = json.loads(raw)
        label = str(data.get("tweet_label", "")).strip().lower()
        if label not in VALID_LABELS:
            label = None
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: scan raw text for label keywords.
    if label is None:
        if "positive" in raw.lower():
            label = "positive"
        elif "negative" in raw.lower():
            label = "negative"

    # --- Parse confidence ---
    confidence = ""
    try:
        confidence = str(data.get("confidence", "")).strip().lower()
    except Exception:
        pass

    # --- Parse reason_tags ---
    reasons = []
    try:
        raw_tags = data.get("reason_tags", [])
        if isinstance(raw_tags, list):
            reasons = [t.strip() for t in raw_tags if t.strip() in VALID_REASON_TAGS]
    except Exception:
        pass

    # --- Determine parse_status ---
    if label in VALID_LABELS and len(reasons) > 0:
        status = "ok"
    elif label in VALID_LABELS:
        status  = "partial"
        reasons = reasons if reasons else ["PARSE_FAIL"]
    else:
        status     = "fail"
        label      = "PARSE_FAIL"
        confidence = ""
        reasons    = ["PARSE_FAIL"]

    return {
        "tweet_label":  label,
        "confidence":   confidence,
        "reason_tags":  "|".join(reasons),
        "parse_status": status,
    }


def apply_parser(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """
    Parse raw LLM output column and write structured columns into df in-place.

    Columns added:
      {model_col}__label, {model_col}__confidence,
      {model_col}__reason_tags, {model_col}__parse_status
    """
    parsed = df[model_col].apply(lambda x: parse_output(str(x)))
    df[f"{model_col}__label"]        = parsed.apply(lambda x: x["tweet_label"])
    df[f"{model_col}__confidence"]   = parsed.apply(lambda x: x["confidence"])
    df[f"{model_col}__reason_tags"]  = parsed.apply(lambda x: x["reason_tags"])
    df[f"{model_col}__parse_status"] = parsed.apply(lambda x: x["parse_status"])
    return df

# =============================================================================
# SECTION 5: Classifier
# =============================================================================

def classify(
    client:     OpenAI,
    model:      str,
    community:  str,
    user_bio:   str,
    tweet_text: str,
    extra_body: dict = None,
) -> str:
    """
    Classify a single tweet using the tweet text and user bio as joint context.

    Returns the raw LLM response string.
    On unrecoverable failure returns "API_ERROR: <reason>".
    API_ERROR rows are retried automatically on the next pipeline resume.
    """
    user_message = json.dumps(
        {
            "community":  community,
            "user_bio":   user_bio or "",
            "tweet_text": tweet_text,
        },
        ensure_ascii=False,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model       = model,
                temperature = 0,
                max_tokens  = MAX_NEW_TOKENS,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                **({"extra_body": extra_body} if extra_body else {}),
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))   # 60s, 120s, 240s
                print(f"  ⚠️  Attempt {attempt} failed: {exc}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"API_ERROR: {exc}"

# =============================================================================
# SECTION 6: Checkpoint Helpers
# =============================================================================

def save_checkpoint(df: pd.DataFrame) -> None:
    """Save the full DataFrame to the checkpoint path atomically."""
    tmp = CHECKPOINT_PATH + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, CHECKPOINT_PATH)
    print(f"  💾 Checkpoint saved ({len(df)} rows) → {CHECKPOINT_PATH}")


def get_todo_indices(df: pd.DataFrame, model_col: str) -> list:
    """
    Return row indices that still require annotation for this model.

    Rows that are non-empty and not API_ERROR are considered done.
    API_ERROR rows are included so they are retried automatically.
    """
    if model_col not in df.columns:
        return list(df.index)
    done = (
        df[model_col].notna()
        & (df[model_col].astype(str).str.strip() != "")
        & (~df[model_col].astype(str).str.startswith("API_ERROR"))
    )
    todo = df[~done].index.tolist()
    print(f"  {model_col}: {done.sum()} done, {len(todo)} remaining")
    return todo

# =============================================================================
# SECTION 7: Dataset & Save Helpers
# =============================================================================

def load_dataset() -> pd.DataFrame:
    """Load the input CSV and normalise required columns."""
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df["user_bio"] = df["user_bio"].fillna("")
    df["tweet_id"] = df["tweet_id"].astype(str)
    print(f"  ✅ Loaded {len(df)} rows from {INPUT_PATH}")
    return df


def save_results(tweets_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    """Save tweet-level and user-level results to Excel and CSV."""
    base        = os.path.splitext(OUTPUT_PATH)[0]
    tweets_xlsx = f"{base}_tweets.xlsx"
    users_xlsx  = f"{base}_users.xlsx"
    tweets_csv  = f"{base}_tweets.csv"
    users_csv   = f"{base}_users.csv"

    tweets_df.to_excel(tweets_xlsx, index=False)
    users_df.to_excel(users_xlsx,   index=False)
    tweets_df.to_csv(tweets_csv, index=False, encoding="utf-8-sig")
    users_df.to_csv(users_csv,   index=False, encoding="utf-8-sig")

    print(f"\n  ✅ Saved tweet-level → {tweets_xlsx}  |  {tweets_csv}")
    print(f"  ✅ Saved user-level  → {users_xlsx}  |  {users_csv}")


def save_patient_tweets(
    tweets_df:   pd.DataFrame,
    users_df:    pd.DataFrame,
    model_cols:  list,
    output_base: str,
) -> None:
    """
    Filter tweets to those belonging to users classified as 'possible_patient'
    and save several subset CSVs for downstream analysis.

    Subsets produced:
      * consensus — users where ALL active models agree on possible_patient
      * either    — users where AT LEAST ONE model says possible_patient
      * per-model — one file per model
    """
    print("\n[pipeline] Filtering patient tweets...")
    tweets_df            = tweets_df.copy()
    tweets_df["user_id"] = tweets_df["user_id"].astype(str)
    users_df             = users_df.copy()
    users_df["user_id"]  = users_df["user_id"].astype(str)
    n_total_tweets       = len(tweets_df)
    n_total_users        = len(users_df)

    def _save_subset(user_ids: set, suffix: str, label: str) -> None:
        if not user_ids:
            print(f"  ⚠️  {label}: 0 users matched — skipping.")
            return
        subset   = tweets_df[tweets_df["user_id"].isin(user_ids)].copy()
        path     = f"{output_base}_patient_tweets_{suffix}.csv"
        subset.to_csv(path, index=False, encoding="utf-8-sig")
        n_users  = len(user_ids)
        n_tweets = len(subset)
        print(
            f"  ✅ {label:<22} {n_users:>4} users ({n_users / n_total_users * 100:.1f}%)  "
            f"{n_tweets:>5} tweets ({n_tweets / n_total_tweets * 100:.1f}%)  "
            f"→ {path}"
        )

    # 1. Consensus — every model voted possible_patient
    label_cols = [
        f"{c.replace('.', '_').replace('-', '_')}__user_label"
        for c in model_cols
    ]
    label_cols = [c for c in label_cols if c in users_df.columns]

    if label_cols:
        consensus_mask = pd.Series(True, index=users_df.index)
        for col in label_cols:
            consensus_mask &= (users_df[col] == "possible_patient")
        consensus_users = set(users_df.loc[consensus_mask, "user_id"])
        _save_subset(consensus_users, "consensus", "Consensus (all models)")

        # 2. Either — at least one model voted possible_patient
        either_mask = pd.Series(False, index=users_df.index)
        for col in label_cols:
            either_mask |= (users_df[col] == "possible_patient")
        either_users = set(users_df.loc[either_mask, "user_id"])
        _save_subset(either_users, "either", "Either model")

    # 3. Per-model subsets
    for col in model_cols:
        slug      = col.replace(".", "_").replace("-", "_")
        label_col = f"{slug}__user_label"
        if label_col not in users_df.columns:
            continue
        model_users = set(
            users_df.loc[users_df[label_col] == "possible_patient", "user_id"]
        )
        _save_subset(model_users, slug, f"{col} only")

# =============================================================================
# SECTION 8: Model Runners
# =============================================================================

def run_gpt(df: pd.DataFrame, gpt_client: OpenAI, qwen_client: OpenAI) -> pd.DataFrame:
    """Annotate df with GPT-4.1. Resumes from checkpoint. Retries API_ERROR rows."""
    if MODEL_GPT not in df.columns:
        df[MODEL_GPT] = None

    client, model, extra_body = _get_client(MODEL_GPT, gpt_client, qwen_client)
    todo = get_todo_indices(df, MODEL_GPT)

    for i, idx in enumerate(todo):
        row = df.loc[idx]
        df.at[idx, MODEL_GPT] = classify(
            client     = client,
            model      = model,
            community  = str(row.get("community", "")),
            user_bio   = str(row.get("user_bio", "")),
            tweet_text = str(row.get("text", "")),
            extra_body = extra_body,
        )
        if (i + 1) % CHECKPOINT_EVERY == 0:
            df = apply_parser(df, MODEL_GPT)
            save_checkpoint(df)

    df = apply_parser(df, MODEL_GPT)
    save_checkpoint(df)
    return df


def run_qwen(df: pd.DataFrame, gpt_client: OpenAI, qwen_client: OpenAI) -> pd.DataFrame:
    """Annotate df with Qwen3-235B. Resumes from checkpoint. Retries API_ERROR rows."""
    if MODEL_QWEN not in df.columns:
        df[MODEL_QWEN] = None

    client, model, extra_body = _get_client(MODEL_QWEN, gpt_client, qwen_client)
    todo = get_todo_indices(df, MODEL_QWEN)

    for i, idx in enumerate(todo):
        row = df.loc[idx]
        df.at[idx, MODEL_QWEN] = classify(
            client     = client,
            model      = model,
            community  = str(row.get("community", "")),
            user_bio   = str(row.get("user_bio", "")),
            tweet_text = str(row.get("text", "")),
            extra_body = extra_body,
        )
        if (i + 1) % CHECKPOINT_EVERY == 0:
            df = apply_parser(df, MODEL_QWEN)
            save_checkpoint(df)

    df = apply_parser(df, MODEL_QWEN)
    save_checkpoint(df)
    return df

# =============================================================================
# SECTION 9: User-level Aggregation
# =============================================================================

def aggregate_user_for_model(user_id: str, tweet_results: list) -> dict:
    """
    Aggregate per-tweet labels for one model into a single user-level label.

    Rules applied in order (any positive evidence wins):
      1. Any tweet has BIO_PATIENT_SELF_IDENTIFICATION in reason_tags
             → possible_patient  (bio override)
      2. Any tweet_label == "positive"
             → possible_patient
      3. All tweets negative and no bio signal
             → other
    """
    # Rule 1: bio override
    bio_override = any(
        "BIO_PATIENT_SELF_IDENTIFICATION" in t.get("reason_tags", [])
        for t in tweet_results
    )
    if bio_override:
        return {
            "user_label":           "possible_patient",
            "aggregation_reason":   "AGG_BIO_PATIENT_OVERRIDE",
            "triggering_tweet_ids": [],
        }

    # Rule 2: any positive tweet wins
    positive_ids = [
        t["tweet_id"] for t in tweet_results if t.get("tweet_label") == "positive"
    ]
    if positive_ids:
        reason = (
            "AGG_ANY_POSITIVE_TWEET"
            if len(positive_ids) == len(tweet_results)
            else "AGG_CONFLICT_POSITIVE_WINS"
        )
        return {
            "user_label":           "possible_patient",
            "aggregation_reason":   reason,
            "triggering_tweet_ids": positive_ids,
        }

    # Rule 3: all negative, no bio signal
    return {
        "user_label":           "other",
        "aggregation_reason":   "AGG_ALL_NEGATIVE_NO_BIO_SIGNAL",
        "triggering_tweet_ids": [],
    }


def build_users_df(tweets_df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """
    Build the user-level DataFrame.

    For each model in model_cols, adds:
      {model}__user_label, {model}__aggregation_reason,
      {model}__triggering_tweet_ids, {model}__positive_tweets,
      {model}__negative_tweets, {model}__error_tweets
    """
    user_rows = []

    for user_id, group in tweets_df.groupby("user_id"):
        uid = str(user_id)
        row = {
            "user_id":      user_id,
            "user_name":    group["user_name"].iloc[0],
            "community":    group["community"].iloc[0],
            "user_bio":     next((b for b in group["user_bio"] if str(b).strip()), ""),
            "total_tweets": len(group),
        }

        for model_col in model_cols:
            label_col = f"{model_col}__label"
            tags_col  = f"{model_col}__reason_tags"
            if label_col not in group.columns:
                continue

            tweet_results = [
                {
                    "tweet_id":    str(r["tweet_id"]),
                    "tweet_label": r[label_col],
                    "reason_tags": (
                        str(r[tags_col]).split("|")
                        if pd.notna(r.get(tags_col)) else []
                    ),
                }
                for _, r in group.iterrows()
            ]

            agg  = aggregate_user_for_model(uid, tweet_results)
            slug = model_col.replace(".", "_").replace("-", "_")

            row[f"{slug}__user_label"]           = agg["user_label"]
            row[f"{slug}__aggregation_reason"]   = agg["aggregation_reason"]
            row[f"{slug}__triggering_tweet_ids"] = "|".join(agg["triggering_tweet_ids"])
            row[f"{slug}__positive_tweets"]      = sum(
                1 for t in tweet_results if t["tweet_label"] == "positive"
            )
            row[f"{slug}__negative_tweets"]      = sum(
                1 for t in tweet_results if t["tweet_label"] == "negative"
            )
            row[f"{slug}__error_tweets"]         = sum(
                1 for t in tweet_results
                if t["tweet_label"] in ("error", "PARSE_FAIL")
            )

        user_rows.append(row)

    return pd.DataFrame(user_rows)

# =============================================================================
# SECTION 10: Agreement Analysis
# =============================================================================

def compute_kappa(df: pd.DataFrame, model_cols: list):
    """
    Compute inter-rater agreement across model label columns.

    Uses Cohen's kappa for 2 models, Fleiss' kappa for 3 or more.
    Rows with PARSE_FAIL in any column are excluded.
    Returns the kappa value, or None if too few valid rows are available.
    """
    label_cols = [f"{c}__label" for c in model_cols]
    sub        = df[[c for c in label_cols if c in df.columns]].copy()
    valid      = sub[~sub.isin(["PARSE_FAIL"]).any(axis=1)]
    n_dropped  = len(sub) - len(valid)
    print(
        f"  Rows used for kappa: {len(valid)} / {len(sub)} "
        f"({n_dropped} dropped — PARSE_FAIL)"
    )

    if len(valid) < 10:
        print("  ⚠️  Too few valid rows for reliable kappa.")
        return None

    if len(label_cols) == 2:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(valid.iloc[:, 0], valid.iloc[:, 1])
        print("  (Cohen's kappa — 2 raters)")
    else:
        from statsmodels.stats.inter_rater import fleiss_kappa as fk
        LABELS = ["positive", "negative"]
        M      = valid.apply(
            lambda row: [row.tolist().count(lbl) for lbl in LABELS], axis=1
        ).tolist()
        kappa  = fk(M)
        print(f"  (Fleiss' kappa — {len(label_cols)} raters)")

    return kappa


def agreement_summary(df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """
    Add majority_label, strong_consensus, and needs_human_review columns to df.
    """
    label_cols = [f"{c}__label" for c in model_cols if f"{c}__label" in df.columns]
    n_models   = len(label_cols)
    threshold  = max(n_models - 1, (n_models // 2) + 1)

    df["vote_positive"]    = df[label_cols].apply(lambda r: (r == "positive").sum(), axis=1)
    df["vote_negative"]    = df[label_cols].apply(lambda r: (r == "negative").sum(), axis=1)
    df["majority_label"]   = df[["vote_positive", "vote_negative"]].apply(
        lambda r: "positive" if r["vote_positive"] > r["vote_negative"] else "negative",
        axis=1,
    )
    df["strong_consensus"]   = df[["vote_positive", "vote_negative"]].apply(
        lambda r: r.max() >= threshold, axis=1
    )
    df["needs_human_review"] = ~df["strong_consensus"]
    return df


def print_analysis(df: pd.DataFrame, model_cols: list):
    """Print a full tweet-level annotation summary to stdout."""
    print("\n" + "=" * 60)
    print("ANNOTATION SUMMARY")
    print("=" * 60)

    print("\n── Parse Status ──")
    for col in model_cols:
        status_col = f"{col}__parse_status"
        if status_col in df.columns:
            ok      = (df[status_col] == "ok").mean()      * 100
            partial = (df[status_col] == "partial").mean() * 100
            fail    = (df[status_col] == "fail").mean()    * 100
            print(f"  {col:<35} ok={ok:.1f}%  partial={partial:.1f}%  fail={fail:.1f}%")

    print("\n── Label Distribution ──")
    for col in model_cols:
        label_col = f"{col}__label"
        if label_col in df.columns:
            counts   = df[label_col].value_counts()
            pos_pct  = counts.get("positive",   0) / len(df) * 100
            neg_pct  = counts.get("negative",   0) / len(df) * 100
            fail_pct = counts.get("PARSE_FAIL", 0) / len(df) * 100
            print(
                f"  {col:<35} "
                f"positive={pos_pct:.1f}%  negative={neg_pct:.1f}%  fail={fail_pct:.1f}%"
            )

    if "strong_consensus" in df.columns:
        print("\n── Agreement Stats ──")
        print(f"  Strong consensus:   {df['strong_consensus'].mean() * 100:.1f}%")
        print(f"  Needs human review: {df['needs_human_review'].mean() * 100:.1f}%")

    print("\n── Inter-rater Agreement ──")
    kappa = compute_kappa(df, model_cols)
    if kappa is not None:
        metric = "Cohen's kappa" if len(model_cols) == 2 else "Fleiss' kappa"
        print(f"  {metric} = {kappa:.4f}")

    print("\n── Reason Tags (positive tweets only) ──")
    for col in model_cols:
        label_col = f"{col}__label"
        tags_col  = f"{col}__reason_tags"
        if label_col in df.columns and tags_col in df.columns:
            top_tags = (
                df[df[label_col] == "positive"][tags_col]
                .dropna()
                .str.split("|")
                .explode()
                .value_counts()
                .head(10)
            )
            print(f"  {col}: {dict(top_tags)}")

    if "community" in df.columns:
        print("\n── Positive Rate per Community ──")
        for col in model_cols:
            label_col = f"{col}__label"
            if label_col in df.columns:
                rates = (
                    df.groupby("community")[label_col]
                    .apply(lambda x: (x == "positive").mean() * 100)
                    .round(1)
                    .rename("positive_%")
                )
                print(f"\n  {col}:")
                print(rates.to_string())

    print("\n" + "=" * 60)

# =============================================================================
# SECTION 10b: User-level Agreement Analysis
# =============================================================================

def compute_user_kappa(users_df: pd.DataFrame, model_cols: list):
    """
    Compute inter-rater agreement on user-level labels.

    Uses Cohen's kappa for 2 models, Fleiss' kappa for 3 or more.
    Rows with missing or empty labels are excluded.
    Returns the kappa value, or None if too few valid users are available.
    """
    label_cols = [
        f"{c.replace('.', '_').replace('-', '_')}__user_label"
        for c in model_cols
    ]
    label_cols = [c for c in label_cols if c in users_df.columns]

    if len(label_cols) < 2:
        print("  ⚠️  Need at least 2 models with user labels for kappa.")
        return None

    sub       = users_df[label_cols].copy()
    valid     = sub.dropna()
    valid     = valid[~valid.isin([""]).any(axis=1)]
    n_dropped = len(sub) - len(valid)
    print(f"  Users used for kappa: {len(valid)} / {len(sub)} ({n_dropped} dropped)")

    if len(valid) < 10:
        print("  ⚠️  Too few valid users for reliable kappa.")
        return None

    if len(label_cols) == 2:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(valid.iloc[:, 0], valid.iloc[:, 1])
        print("  (Cohen's kappa — 2 raters, user-level)")
    else:
        from statsmodels.stats.inter_rater import fleiss_kappa as fk
        LABELS = ["possible_patient", "other"]
        M = valid.apply(
            lambda row: [row.tolist().count(lbl) for lbl in LABELS], axis=1
        ).tolist()
        kappa = fk(M)
        print(f"  (Fleiss' kappa — {len(label_cols)} raters, user-level)")

    return kappa


def user_agreement_summary(users_df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    """
    Add user-level vote counts, majority label, consensus, and review flags.
    """
    label_cols = [
        f"{c.replace('.', '_').replace('-', '_')}__user_label"
        for c in model_cols
    ]
    label_cols = [c for c in label_cols if c in users_df.columns]

    if len(label_cols) < 2:
        return users_df

    n_models  = len(label_cols)
    threshold = max(n_models - 1, (n_models // 2) + 1)

    users_df["user_vote_patient"] = users_df[label_cols].apply(
        lambda r: (r == "possible_patient").sum(), axis=1
    )
    users_df["user_vote_other"] = users_df[label_cols].apply(
        lambda r: (r == "other").sum(), axis=1
    )
    users_df["user_majority_label"] = users_df[
        ["user_vote_patient", "user_vote_other"]
    ].apply(
        lambda r: "possible_patient"
        if r["user_vote_patient"] > r["user_vote_other"]
        else "other",
        axis=1,
    )
    users_df["user_strong_consensus"] = users_df[
        ["user_vote_patient", "user_vote_other"]
    ].apply(lambda r: r.max() >= threshold, axis=1)
    users_df["user_needs_human_review"] = ~users_df["user_strong_consensus"]
    return users_df


def print_user_analysis(users_df: pd.DataFrame, model_cols: list):
    """Print user-level annotation and agreement summary to stdout."""
    print("\n" + "=" * 60)
    print("USER-LEVEL ANNOTATION SUMMARY")
    print("=" * 60)

    print(f"\n  Total users: {len(users_df)}")

    print("\n── User Label Distribution ──")
    for col in model_cols:
        slug      = col.replace(".", "_").replace("-", "_")
        label_col = f"{slug}__user_label"
        if label_col in users_df.columns:
            counts = users_df[label_col].value_counts()
            pat    = counts.get("possible_patient", 0) / len(users_df) * 100
            oth    = counts.get("other",            0) / len(users_df) * 100
            print(f"  {col:<35} possible_patient={pat:.1f}%  other={oth:.1f}%")

    print("\n── Aggregation Reason Breakdown ──")
    for col in model_cols:
        slug       = col.replace(".", "_").replace("-", "_")
        reason_col = f"{slug}__aggregation_reason"
        if reason_col in users_df.columns:
            counts = users_df[reason_col].value_counts()
            print(f"\n  {col}:")
            for reason, n in counts.items():
                pct = n / len(users_df) * 100
                print(f"    {reason:<35} {n:>5}  ({pct:.1f}%)")

    if "user_strong_consensus" in users_df.columns:
        print("\n── User-Level Agreement Stats ──")
        print(f"  Strong consensus:   {users_df['user_strong_consensus'].mean() * 100:.1f}%")
        print(f"  Needs human review: {users_df['user_needs_human_review'].mean() * 100:.1f}%")

    print("\n── User-Level Inter-rater Agreement ──")
    kappa = compute_user_kappa(users_df, model_cols)
    if kappa is not None:
        metric = "Cohen's kappa" if len(model_cols) == 2 else "Fleiss' kappa"
        print(f"  {metric} (user-level) = {kappa:.4f}")

    # Side-by-side confusion matrix (only meaningful for exactly 2 models)
    if len(model_cols) == 2:
        slugs    = [c.replace(".", "_").replace("-", "_") for c in model_cols]
        lbl_cols = [f"{s}__user_label" for s in slugs]
        if all(c in users_df.columns for c in lbl_cols):
            print("\n── User-Level Confusion Matrix ──")
            ct = pd.crosstab(
                users_df[lbl_cols[0]],
                users_df[lbl_cols[1]],
                rownames=[model_cols[0]],
                colnames=[model_cols[1]],
            )
            print(ct.to_string())

    if "community" in users_df.columns:
        print("\n── possible_patient Rate per Community (user-level) ──")
        for col in model_cols:
            slug      = col.replace(".", "_").replace("-", "_")
            label_col = f"{slug}__user_label"
            if label_col in users_df.columns:
                rates = (
                    users_df.groupby("community")[label_col]
                    .apply(lambda x: (x == "possible_patient").mean() * 100)
                    .round(1)
                    .rename("possible_patient_%")
                )
                print(f"\n  {col}:")
                print(rates.to_string())

    print("\n" + "=" * 60)

# =============================================================================
# SECTION 11: Pipeline Orchestrator
# =============================================================================

def run_pipeline(mode: str = "rerun") -> None:
    """
    Execute the classification pipeline.

    mode="rerun"  (default) — re-parses existing LLM outputs; no API calls made.
    mode="fresh"            — runs both GPT and Qwen annotation from scratch,
                              resuming from checkpoint if one exists.

    Steps:
      1. Load tweet data
      2. (fresh only) Annotate with GPT-4.1 and Qwen3-235B
      3. Re-parse raw LLM outputs
      4. Tweet-level agreement analysis
      5. User-level aggregation per model
      6. User-level agreement analysis
      7. Save all results and patient-tweet subsets
    """
    print(f"\n[pipeline] Starting ({mode} mode)...")

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    if mode == "fresh":
        df = load_dataset()
    else:
        df = pd.read_csv(RESULTS_INPUT_PATH, encoding="utf-8-sig")
        df["user_bio"] = df["user_bio"].fillna("")
        df["tweet_id"] = df["tweet_id"].astype(str)
        print(f"  ✅ Loaded {len(df)} rows from {RESULTS_INPUT_PATH}")

    # ── Step 2: Annotation (fresh mode only) ──────────────────────────────────
    if mode == "fresh":
        gpt_client, qwen_client = _build_clients()
        df = run_gpt(df,  gpt_client, qwen_client)
        df = run_qwen(df, gpt_client, qwen_client)

    # ── Step 3: Parse raw LLM outputs ─────────────────────────────────────────
    for model_col in ALL_MODEL_COLS:
        if model_col in df.columns:
            df = apply_parser(df, model_col)
            print(f"  ✅ Parsed outputs for {model_col}")
        else:
            print(f"  ⚠️  Column '{model_col}' not found — skipping")

    # ── Step 4: Tweet-level agreement analysis ────────────────────────────────
    annotated_cols = [c for c in ALL_MODEL_COLS if f"{c}__label" in df.columns]
    if len(annotated_cols) >= 2:
        df = agreement_summary(df, annotated_cols)
    print_analysis(df, annotated_cols)

    # ── Step 5: User-level aggregation ────────────────────────────────────────
    users_df = build_users_df(df, annotated_cols)

    # ── Step 6: User-level agreement analysis ─────────────────────────────────
    if len(annotated_cols) >= 2:
        users_df = user_agreement_summary(users_df, annotated_cols)
    print_user_analysis(users_df, annotated_cols)

    # ── Step 7: Save results ───────────────────────────────────────────────────
    save_results(df, users_df)
    output_base = os.path.splitext(OUTPUT_PATH)[0]
    save_patient_tweets(df, users_df, annotated_cols, output_base)

    print("\n[pipeline] Done.")

# =============================================================================
# SECTION 12: CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arabic mental-health tweet classification pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["rerun", "fresh"],
        default="rerun",
        help=(
            "rerun: re-parse existing LLM outputs without new API calls. "
            "fresh: annotate from scratch with GPT-4.1 and Qwen3-235B."
        ),
    )
    parser.add_argument(
        "--input",
        default=INPUT_PATH,
        help="Path to the preprocessed tweet CSV (required for fresh mode).",
    )
    parser.add_argument(
        "--results",
        default=RESULTS_INPUT_PATH,
        help="Path to the existing annotated CSV (required for rerun mode).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Base path for output files (suffixes added automatically).",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
        help="Path for the checkpoint CSV used during fresh annotation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply CLI overrides to module-level path constants.
    INPUT_PATH         = args.input
    RESULTS_INPUT_PATH = args.results
    OUTPUT_PATH        = args.output
    CHECKPOINT_PATH    = args.checkpoint

    run_pipeline(mode=args.mode)
