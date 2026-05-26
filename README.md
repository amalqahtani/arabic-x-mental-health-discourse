# Arabic X Mental Health Discourse Annotations

This repository contains **post IDs** and **GPT-4.1 annotation labels** for the paper:

**Understanding the Sociocultural Dimensions of Mental Health Discourse in Arabic-Language X Communities**

Accepted at **#SMM4H-HeaRD 2026**, the Social Media Mining for Health Applications and Health Real-World Data Workshop, co-located with **ACL 2026**.

## Dataset

The released file contains **8,147 post-level annotation rows** from Arabic-language X Communities related to:

- Borderline personality disorder (BPD)
- Bipolar disorder
- ADHD

To protect user privacy and comply with platform policies, this repository releases **post IDs only**. It does not include tweet text, usernames, user bios, or profile information.

## File

| File | Description |
|---|---|
| `post_ids_labels_only.csv` | X post IDs with GPT-4.1 personal-disclosure annotations |

## Data Format

| Column | Description |
|---|---|
| `post_id` | X/Twitter post ID |
| `gpt-4.1` | JSON annotation output from GPT-4.1 |

Each `gpt-4.1` entry contains:

| Field | Description |
|---|---|
| `tweet_label` | `positive` or `negative` personal-disclosure label |
| `confidence` | Annotation confidence: `high`, `medium`, or `low` |
| `reason_tags` | Tags explaining the annotation decision |

## Annotation

Annotations identify whether a post contains **personal-disclosure signals** suggesting that the author may be personally discussing lived experience with a mental health condition.

These labels are for research use only and **must not be interpreted as clinical diagnoses**.

## Data Use

Posts may be rehydrated using the provided post IDs, subject to X’s terms of service and post availability.

## Citation

```bibtex
@inproceedings{qahtani2026sociocultural,
  title = {Understanding the Sociocultural Dimensions of Mental Health Discourse in Arabic-Language X Communities},
  author = {"Alqahtani, Amal Abdullah and Salama, Rana and Diab, Mona T.",
  booktitle = "Proceedings of the 11th Social Media Mining for Health Research and Applications (SMM4H 2026) Workshop and Shared Tasks", 
  month = jul,
  year = "2026",
  address = "San Diego, California",
  publisher = "Association for Computational Linguistics",
}
