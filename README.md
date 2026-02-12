# F1 Race Winner Prediction (FastF1)

This project trains a machine-learning model to estimate each driver's probability of winning a Formula 1 race using historical race data from the [FastF1](https://theoehrly.github.io/Fast-F1/) API.

## What this does

- Downloads race results for one or more seasons with FastF1.
- Builds a **driver-race** training table (one row per driver per race).
- Uses only features available **before** each race (rolling form and team form).
- Trains a classifier to predict whether a driver wins that race.
- Exports a reusable model artifact for future inference.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_winner_model.py --start-year 2018 --end-year 2024
```

Model artifacts are saved in `artifacts/`.

## Output

After training, you'll get:

- `artifacts/winner_model.joblib`: trained sklearn pipeline
- `artifacts/feature_columns.json`: feature list used for training

The script also prints validation metrics (ROC-AUC, log-loss, Brier score).

## Notes

- FastF1 can take time on first run because sessions are downloaded/cached.
- The model predicts **driver win probability**. The race winner prediction is the driver with highest probability for that event.
- This is a baseline. You can improve it by adding qualifying pace, weather, circuit characteristics, and bookmaker odds.
