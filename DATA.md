# Dataset

The full tweet text cannot be redistributed under the
X (Twitter) Developer Policy.

This repository provides `post_ids.csv`, containing the
tweet IDs and community labels for all 9,582 preprocessed
posts described in the paper.

## Re-hydration

To recover the full tweet content from the IDs, use
[twarc2](https://twarc-project.readthedocs.io/):

    pip install twarc
    twarc2 hydrate post_ids.csv hydrated.jsonl

A valid X Developer account and bearer token are required.
Note that tweets deleted or made private after our
collection window (March 2022 – February 2026) will
not be recoverable.

## File format

| Column      | Description                          |
|-------------|--------------------------------------|
| `post_id`   | Unique X post identifier (string)    |
| `community` | Source community: BPD, bipolar, ADHD |

## Statistics

| Community | Posts |
|-----------|-------|
| BPD       | 6,324 |
| bipolar   | 2,668 |
| ADHD      |   590 |
| **Total** | **9,582** |
