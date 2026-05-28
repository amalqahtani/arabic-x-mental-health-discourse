# Arabic Mental-Health Tweet Classification Pipeline

This repository contains the LLM-assisted personal-disclosure classification pipeline described in:

> **Understanding the Sociocultural Dimensions of Mental Health Discourse in Arabic-Language X Communities**
> Amal Alqahtani, Rana Salama, and Mona Diab
> *Proceedings of the 11th Workshop on Social Media Mining for Health Applications and Health Real-World Data (#SMM4H-HeaRD 2026), co-located with ACL 2026*

The pipeline classifies Arabic tweets from condition-specific X Communities (ADHD, Bipolar Disorder, BPD) as either containing personal-disclosure signals (`positive`) or not (`negative`), then aggregates tweet-level labels into user-level verdicts (`possible_patient` / `other`). GPT-4.1 serves as the primary annotator; Qwen3-235B-A22B-Instruct-2507 runs in parallel as a conservative screening model.

---

## Repository

```
https://github.com/amalqahtani/arabic-x-mental-health-discourse
```

---

## Repository Contents

| File | Description |
|---|---|
| `classification_pipeline.py` | LLM-assisted personal-disclosure classification pipeline |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `LICENSE` | MIT License |
| `.gitignore` | Excludes data files, credentials, and build artefacts |
| `.env.example` | Template for API key environment variables |
| `post_ids.csv` | Post IDs of the 8,147 tweets in the self-disclosure-filtered corpus |

> **Note on data release:** Only post IDs are released, in accordance with the X Developer Agreement. Tweet content must be rehydrated using the X API. User identifiers and text are not included.

---

## Requirements

- Python >= 3.8
- An [OpenAI API key](https://platform.openai.com/account/api-keys) with access to `gpt-4.1`
- An [Alibaba DashScope API key](https://dashscope.aliyuncs.com/) with access to `qwen3-235b-a22b-instruct-2507`

> **Note on model identifiers:** DashScope model names are subject to change. Verify `MODEL_QWEN` in `classification_pipeline.py` against the current [DashScope model list](https://dashscope.aliyuncs.com/) before running.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## API Key Setup

**Never hardcode API keys in source files.** Export them as environment variables before running:

```bash
export OPENAI_API_KEY="your-openai-key"
export DASHSCOPE_API_KEY="your-dashscope-key"
```

On Windows:

```cmd
set OPENAI_API_KEY=your-openai-key
set DASHSCOPE_API_KEY=your-dashscope-key
```

A template is provided in `.env.example`.

---

## Input Data Format

The pipeline expects a UTF-8 CSV with the following columns:

| Column      | Type   | Description                                    |
|-------------|--------|------------------------------------------------|
| `tweet_id`  | string | Unique tweet identifier                        |
| `user_id`   | string | Unique user identifier                         |
| `user_name` | string | Username (anonymised for release)              |
| `user_bio`  | string | Profile biography text (may be empty)          |
| `community` | string | Community label: `ADHD`, `BPD`, or `bipolar`   |
| `text`      | string | Tweet text (Arabic)                            |

---

## Usage

### Rerun mode (re-parse existing LLM outputs — no API calls)

Use this to reproduce agreement statistics or regenerate output files from a prior annotation run without incurring new API costs:

```bash
python classification_pipeline.py \
    --mode rerun \
    --results results_tweets.csv \
    --output  results_rerun.csv
```

### Fresh mode (full annotation with GPT-4.1 and Qwen3-235B-A22B)

```bash
python classification_pipeline.py \
    --mode fresh \
    --input tweets_preprocessed.csv \
    --output results.csv
```

Annotation resumes automatically from `--checkpoint` if the process is interrupted.

### All CLI options

```
--mode        rerun | fresh          (default: rerun)
--input       path to input CSV      (required for fresh mode)
--results     path to existing CSV   (required for rerun mode)
--output      base path for outputs  (default: results_rerun.csv)
--checkpoint  path for checkpoint    (default: checkpoint.csv)
```

---

## Output Files

| File                                       | Description                                                      |
|--------------------------------------------|------------------------------------------------------------------|
| `{output}_tweets.csv` / `.xlsx`            | Tweet-level labels, confidence, reason tags, vote counts         |
| `{output}_users.csv` / `.xlsx`             | User-level verdicts, aggregation reasons, tweet counts           |
| `{output}_patient_tweets_consensus.csv`    | Tweets from users both models agree are `possible_patient`       |
| `{output}_patient_tweets_either.csv`       | Tweets from users at least one model classifies as `possible_patient` |
| `{output}_patient_tweets_{model}.csv`      | Per-model patient-tweet subsets                                  |

---

## Classification Schema

### Tweet-level labels
- `positive` — tweet contains personal-disclosure signals
- `negative` — no personal-disclosure signals detected

### User-level labels
- `possible_patient` — at least one tweet is positive, or bio identifies user as a patient
- `other` — all tweets negative and no bio signal detected

### Aggregation rules (applied in priority order)

| Priority | Condition | User label | Reason code |
|----------|-----------|------------|-------------|
| 1 | Any tweet has `BIO_PATIENT_SELF_IDENTIFICATION` | `possible_patient` | `AGG_BIO_PATIENT_OVERRIDE` |
| 2a | All tweets positive | `possible_patient` | `AGG_ANY_POSITIVE_TWEET` |
| 2b | Mixed positive/negative | `possible_patient` | `AGG_CONFLICT_POSITIVE_WINS` |
| 3 | All negative, no bio signal | `other` | `AGG_ALL_NEGATIVE_NO_BIO_SIGNAL` |

---

## Pipeline Performance (from paper)

| Comparison | κ | Agreement | F₁ (positive) | Band |
|---|---|---|---|---|
| Inter-human ceiling | 0.905 | 96.0% | — | Almost perfect |
| GPT-4.1 vs. human gold | 0.631 | 83.9% | 0.88 | Substantial |
| Qwen3 vs. human gold | 0.329 | 66.3% | 0.72 | Fair |
| GPT-4.1 vs. Qwen3 (inter-model) | 0.840 | 90.8% | — | Almost perfect |

Validation was performed on a stratified sample of 200 tweets across five difficulty tiers. Full results are reported in Appendix D of the paper.

---

## Agreement Analysis

The pipeline automatically computes:
- **Cohen's kappa** (2 models) or **Fleiss' kappa** (3+ models) at both tweet and user level
- Strong consensus rate and human-review flag per tweet and per user
- Confusion matrix (2-model case)
- Per-community positive rates

---

## Hardware and Cost Notes

- The pipeline makes one API call per tweet per model.
- At 9,582 tweets and 2 models, expect approximately 19,164 API calls total.
- Runtime depends on API latency; allow several hours for a full fresh run.
- `temperature=0` is used throughout for deterministic output.
- Checkpoint files are saved every 10 tweets; the pipeline resumes safely after interruption.

---

## Ethics and Data Statement

This pipeline was developed for academic research on Arabic mental-health discourse in condition-specific X Communities (BPD, Bipolar, ADHD). All user identifiers in released data are anonymised before analysis. The `possible_patient` label is an operational descriptor based on the presence of self-reported experiential language; it does not constitute a clinical diagnosis. No clinical inferences should be drawn from the pipeline outputs.

The classification prompt was developed on a Saudi-centric dataset. Dialectal adaptation is needed before applying the pipeline to Egyptian, Levantine, or North African Arabic (see Table 8 in the paper).

---

## Citation

If you use this pipeline or the associated corpus, please cite:

```bibtex
@inproceedings{alqahtani2026arabic,
  title     = {Understanding the Sociocultural Dimensions of Mental Health Discourse in {A}rabic-Language {X} Communities},
  author    = {Alqahtani, Amal and Salama, Rana and Diab, Mona},
  booktitle = {Proceedings of the 11th Workshop on Social Media Mining for Health Applications and Health Real-World Data ({SMM4H-HeaRD} 2026)},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {https://github.com/amalqahtani/arabic-x-mental-health-discourse},
}
```

---

## License

MIT License. See `LICENSE` for details.
