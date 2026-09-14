# Team Gashaka - Intelligent Complaint Classification

This TRI AI Cohort 10 project classifies consumer-support complaints into ten categories: `account_access`, `billing`, `customer_service`, `delivery_shipping`, `fraud_unauthorized`, `general_inquiry`, `product_defect`, `refund_return`, `subscription_cancel`, and `warranty_repair`.

## Dataset

The ComplaintSense competition data is stored in `data/`. `train_complaints.csv` has 380 labelled rows with complaint text, category, and a `FamilyId`; `test_complaints.csv` has 160 unlabelled rows. `FamilyId` is used only to construct grouped validation folds - it is not a model feature. The supplied data is small and template-like, so results should not be assumed to generalise to production support traffic.

## Training pipeline

The final implementation is in `scripts/late submission.ipynb`. It normalises text into a `core_text` representation, groups repeated complaint families, and uses five-fold grouped cross-validation. It trains two balanced logistic-regression classifiers: one over word/character TF-IDF features and one over `sentence-transformers/all-mpnet-base-v2` embeddings. It then selects a 50/50 probability blend and fits the final models on all available training rows. Random seed: 42.

`scripts/first-submission.ipynb` contains the earlier DistilBERT baseline. Its 10-fold run produced an overall out-of-fold macro F1 of 0.4942. The later notebook achieved a best grouped out-of-fold macro F1 of 0.7556 with the MPNet model; this is development validation, not the unseen competition score.

## Evaluation

The competition metric is macro F1, which weights all classes equally. The pipeline uses grouped folds to avoid placing complaints from the same family in both train and validation splits. Submission checks assert the required `ComplaintId,Category` schema, all 160 test IDs, valid labels, and no missing predictions.

## Reproduction

1. Install Python 3.10+ and the packages used in `scripts/late submission.ipynb`, including `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `sentencepiece`, and `torch`.
2. Open `scripts/late submission.ipynb` in Kaggle or Jupyter.
3. Ensure `data/train_complaints.csv` and `data/test_complaints.csv` are available at the paths configured in the notebook, then run all cells in order.
4. The notebook writes `submission.csv`; the repository retains submitted CSV variants in `data/`.

## Project documents

The required cohort challenge documents are in `docs/`: `problem_statement.pdf`, `data_card.pdf`, `impact_statement_card.pdf`, and `stakeholder_engagement.pdf`.

## Contributors and mentors

Team Gashaka. This repository does not record a verified mentor contribution; none is claimed here.
