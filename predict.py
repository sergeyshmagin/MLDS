# -*- coding: utf-8 -*-
"""
CTR inference script — Advandex
Usage:
    python predict.py --input sample.csv --output predictions.csv
"""
import argparse
import pandas as pd
import joblib


def extract_time_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    h = X['hour'].astype(str)
    X['hour_of_day'] = h.str[-2:].astype(int)
    X['day_of_week'] = pd.to_datetime(h.str[:6], format='%y%m%d').dt.dayofweek
    return X.drop(columns=['hour'])


def main():
    parser = argparse.ArgumentParser(description='CTR prediction')
    parser.add_argument('--input',    required=True,  help='Path to input CSV')
    parser.add_argument('--output',   required=True,  help='Path to output CSV')
    parser.add_argument('--pipeline', default='artifacts/pipeline_lr_calibrated.joblib',
                        help='Path to pipeline .joblib')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Decision threshold (default 0.3)')
    args = parser.parse_args()

    pipeline = joblib.load(args.pipeline)

    df = pd.read_csv(args.input)
    drop_cols = [c for c in ['id', 'click'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    X_fe = extract_time_features(X)

    proba = pipeline.predict_proba(X_fe)[:, 1]
    pred  = (proba >= args.threshold).astype(int)

    out = df[['id']].copy() if 'id' in df.columns else pd.DataFrame(index=df.index)
    out['ctr_proba'] = proba.round(6)
    out['click_pred'] = pred

    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} predictions -> {args.output}")


if __name__ == '__main__':
    main()
