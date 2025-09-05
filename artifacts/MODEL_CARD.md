# SoMe Score Model Card

**Version:** some-score-20250902T112818Z

**Features:** engagement_rate, follower_count_c, posting_frequency_c

**Splits:** train 60% / val 20% / test 20% (stratified)

## Validation metrics
- ROC AUC: 0.7634
- PR AUC: 0.5021
- Brier: 0.1626

## Test metrics
- ROC AUC: 0.8224
- PR AUC: 0.6005
- Brier: 0.1292

## Thresholds
{
  "t_review": 0.09090909090909091,
  "t_launch": 0.9059585435440444
}

## Category success rate (VAL)
- Ignore: 0.000 (n=13)
- Launch: 0.700 (n=10)
- Review: 0.221 (n=77)
