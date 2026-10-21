# Labeling guidelines (Component 2)

Do not use model outputs as gold. Two bilingual annotators + one clinician arbitrator.

## Intent

One primary intent from the schema in `lexicon.yaml`. Multi-label allowed for `pregnancy_breastfeeding` + `eligibility` or `contraindications`.

- **eligibility**: can this person/method be used?
- **contraindications**: must not / is forbidden / ተቃራኒ
- These two are never merged.

## Entities

Span-level BIO in `training/ner_schema.json`. Only mark what the user said. Never add implied pregnancy or age.

## Retrieval

Each query lists `relevant_ids` from the KB. Include at least one **hard negative** that shares tokens but opposite polarity (breastfeeding allowed vs contraindicated).

## Splits

Group paraphrases with a `cluster` id. Put an entire cluster in train **or** test, never both.

## Status

`developer_draft` in this repo is not independently reviewed gold. Replace before publishing metrics as clinical.
