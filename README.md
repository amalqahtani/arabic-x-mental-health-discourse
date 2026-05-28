# Arabic Mental-Health Tweet Classification Pipeline

This repository contains the LLM-assisted personal-disclosure classification pipeline described in:

> **Understanding the Sociocultural Dimensions of Mental Health Discourse in Arabic-Language X Communities**
> Amal Alqahtani, Rana Salama, and Mona Diab
> *Proceedings of the 11th Workshop on Social Media Mining for Health Applications and Health Real-World Data (#SMM4H-HeaRD 2026), co-located with ACL 2026*

The pipeline classifies Arabic tweets from condition-specific X Communities (ADHD, Bipolar Disorder, BPD) as either containing personal-disclosure signals (`positive`) or not (`negative`), then aggregates tweet-level labels into user-level verdicts (`likely_disclosure` / `other`). GPT-4.1 serves as the primary annotator; Qwen3-235B-A22B-Instruct-2507 runs in parallel as a conservative screening model.

---

## Repository Contents

| File | Description |
|---|---|
| `requirements.txt` | Python dependencies |
| `classification_pipeline.py` | LLM-assisted personal-disclosure classification pipeline |
| `post_ids.csv` | Post IDs for all 9,582 preprocessed tweets |
| `DATA.md` | Dataset description and re-hydration instructions |
| `LICENSE` | MIT License |

> **Note on data release:** Only post IDs are released, in accordance with the X Developer Agreement. Tweet content must be re-hydrated using the X API. User identifiers and text are not included.

---

## Requirements

- Python >= 3.8
- An [OpenAI API key](https://platform.openai.com/account/api-keys) with access to `gpt-4.1`
- An [Alibaba DashScope API key](https://dashscope.aliyuncs.com/) with access to `qwen3-235b-a22b-instruct-2507`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## API Key Setup

```bash
export OPENAI_API_KEY="your-openai-key"
export DASHSCOPE_API_KEY="your-dashscope-key"
```

On Windows:

```cmd
set OPENAI_API_KEY=your-openai-key
set DASHSCOPE_API_KEY=your-dashscope-key
```
---

## Input Data Format

The pipeline expects a UTF-8 CSV with the following columns:

| Column | Type | Description |
|---|---|---|
| `tweet_id` | string | Unique tweet identifier |
| `user_id` | string | Unique user identifier |
| `user_name` | string | Username (anonymised) |
| `user_bio` | string | Profile biography text (may be empty) |
| `community` | string | Community label: `ADHD`, `BPD`, or `bipolar` |
| `text` | string | Tweet text (Arabic) |

---

## Usage

```bash
python classification_pipeline.py --mode fresh --input tweets_preprocessed.csv --output results.csv
```

---

## Output Files

| File | Description |
|---|---|
| `{output}_tweets.csv` / `.xlsx` | Tweet-level labels, confidence, reason tags |
| `{output}_users.csv` / `.xlsx` | User-level verdicts and aggregation reasons |
| `{output}_disclosure_tweets_consensus.csv` | Tweets from users both models agree are `likely_disclosure` |
| `{output}_disclosure_tweets_either.csv` | Tweets from users at least one model classifies as `likely_disclosure` |
| `{output}_disclosure_tweets_{model}.csv` | Per-model disclosure-tweet subsets |

---

## Classification Schema

### Tweet-level labels
- `positive` — tweet contains personal-disclosure signals
- `negative` — no personal-disclosure signals detected

### User-level labels
- `likely_disclosure` — at least one tweet is positive, or bio contains self-identification language
- `other` — all tweets negative and no self-identification signal detected

### Aggregation rules (applied in priority order)

| Priority | Condition | User label | Reason code |
|---|---|---|---|
| 1 | Bio contains self-identification language | `likely_disclosure` | `AGG_BIO_DISCLOSURE_OVERRIDE` |
| 2a | All tweets positive | `likely_disclosure` | `AGG_ANY_POSITIVE_TWEET` |
| 2b | Mixed positive/negative | `likely_disclosure` | `AGG_CONFLICT_POSITIVE_WINS` |
| 3 | All negative, no bio signal | `other` | `AGG_ALL_NEGATIVE_NO_BIO_SIGNAL` |

---


## Ethics and Data Statement

This pipeline was developed for academic research on Arabic mental-health discourse in condition-specific X Communities (BPD, Bipolar, ADHD). All user identifiers in released data are anonymised before analysis. The `likely_disclosure` label is an operational descriptor based on the presence of self-reported experiential language and does not constitute a clinical diagnosis. No clinical inferences should be drawn from the pipeline outputs.

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
