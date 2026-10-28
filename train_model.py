"""
train_model.py  -  Chess Move Predictor: Model Training module

What this module does
---------------------
Reads   data/processed/hikaru_move_features.csv   (made by feature_extractor.py)
Each row is ONE legal candidate move in ONE position:
    label = 1  ->  the move Hikaru actually played
    label = 0  ->  another legal move he did not play

Steps:
  1. Load and validate the data.
  2. Choose the input features (everything except metadata and the label).
  3. Split into training and test sets BY GAME (never by row).
  4. Train a RandomForestClassifier.
  5. Evaluate it, including per-position ranking metrics
     (top-1, top-3, mean reciprocal rank).
  6. Save the model and its feature-column list in the  models/  folder.

The model learns to recognise moves that RESEMBLE Hikaru's historical choices.
It does not use Stockfish and does not try to find the objectively best move.

Install:   pip install pandas scikit-learn joblib
Run:       python train_model.py      (from the project root)

Why we split by game and not by row (data leakage)
--------------------------------------------------
All moves in one game are strongly related: the same opening, the same pawn
structure, similar pieces on similar squares. If rows were shuffled randomly,
candidate moves from the SAME game (even from neighbouring positions) would end
up in both the training and the test set. The model could then "recognise" a
game it has already seen instead of learning Hikaru's general style, and the
test scores would look far better than they really are. Keeping whole games
together makes the test set a fair simulation of unseen games.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import GroupShuffleSplit
except ImportError as import_error:  # friendly message for beginners
    print("A required package is missing:", import_error)
    print("Install everything with:  pip install pandas scikit-learn joblib")
    sys.exit(1)


# ---------------------------------------------------------------------------
# SETTINGS  (change these in one place)
# ---------------------------------------------------------------------------

# Paths are relative to THIS file, not to the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "processed" / "hikaru_move_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "hikaru_move_model.joblib"
INFO_PATH = MODEL_DIR / "hikaru_move_model_info.json"   # feature names + results
TEMP_MODEL_PATH = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")
TEMP_INFO_PATH = INFO_PATH.with_name(INFO_PATH.name + ".tmp")

TARGET_COLUMN = "label"

# Columns used only for identification/debugging. NEVER used as model input.
# (actual_move_uci and candidate_move_uci would reveal the answer.)
METADATA_COLUMNS = ["position_id", "game_id", "ply", "fen",
                    "actual_move_uci", "candidate_move_uci"]

# Columns this script cannot work without.
REQUIRED_COLUMNS = ["position_id", "game_id", TARGET_COLUMN]

RANDOM_SEED = 42      # makes the split and the model reproducible
TEST_SIZE = 0.2       # 20% of the GAMES go to the test set

# Random forest settings. "balanced_subsample" gives the rare label-1 rows
# more weight, so the model does not just answer "0" every time.
# Limited depth and leaf size keep training fast and reduce overfitting.
RF_SETTINGS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,                 # use all CPU cores
    "random_state": RANDOM_SEED,
}


def fail(message):
    """Print a clear error and stop the program safely."""
    print(f"\nERROR: {message}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# STEP 1: LOAD AND VALIDATE
# ---------------------------------------------------------------------------

def load_data(path):
    """Read the features CSV and check that the data is usable."""
    if not path.exists():
        fail(f"features file not found:\n  {path}\n"
             "Run feature_extractor.py first to create it.")

    try:
        # Read IDs as text so they are never turned into numbers
        df = pd.read_csv(path, dtype={"position_id": str, "game_id": str})
    except pd.errors.EmptyDataError:
        fail("the features file is empty.")
    except Exception as error:
        fail(f"could not read the features file: {error}")

    if df.empty:
        fail("the features file has a header but no rows.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        fail(f"required column(s) missing: {missing}\n"
             f"Columns found: {list(df.columns)}")

    for col in METADATA_COLUMNS:
        if col not in df.columns:
            print(f"WARNING: metadata column '{col}' is not in the file "
                  "(it is not needed for training, so this is fine).")

    # --- IDs must be present and usable ---
    for col in ("position_id", "game_id"):
        blank = df[col].isna() | (df[col].str.strip() == "")
        if blank.any():
            if col == "game_id":
                fail(f"{int(blank.sum())} row(s) have a missing game_id. "
                     "Splitting by game is impossible, so training was stopped "
                     "(a random row split would leak information).")
            fail(f"{int(blank.sum())} row(s) have a missing {col}.")

    # --- label must be only 0 or 1 ---
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if df[TARGET_COLUMN].isna().any() or not df[TARGET_COLUMN].isin([0, 1]).all():
        fail("the 'label' column must contain only 0 and 1.")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # --- every position: exactly one positive, and it belongs to one game ---
    positives_per_position = df.groupby("position_id")[TARGET_COLUMN].sum()
    bad_positions = positives_per_position[positives_per_position != 1]
    if len(bad_positions) > 0:
        fail(f"{len(bad_positions)} position(s) do not have exactly one positive "
             f"candidate (example: {bad_positions.index[0]}). "
             "Re-run feature_extractor.py.")

    games_per_position = df.groupby("position_id")["game_id"].nunique()
    if (games_per_position > 1).any():
        fail("some position_id values appear under more than one game_id.")

    return df.reset_index(drop=True)


def choose_feature_columns(df):
    """
    Features = every column that is NOT metadata and NOT the label.
    Choosing by exclusion means answer-revealing columns can never slip in.
    """
    feature_columns = [col for col in df.columns
                       if col not in METADATA_COLUMNS and col != TARGET_COLUMN]
    if not feature_columns:
        fail("no feature columns were found.")

    non_numeric = [col for col in feature_columns
                   if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        fail(f"these feature columns are not numeric: {non_numeric}")

    values = df[feature_columns]
    columns_with_missing = values.columns[values.isna().any()].tolist()
    if columns_with_missing:
        fail(f"these feature columns contain missing values: {columns_with_missing}")
    if np.isinf(values.to_numpy(dtype=float)).any():
        fail("some feature values are infinite.")

    return feature_columns


# ---------------------------------------------------------------------------
# STEP 2: SPLIT BY GAME
# ---------------------------------------------------------------------------

def split_by_game(df):
    """Split rows into train/test so that every game stays in ONE set."""
    number_of_games = df["game_id"].nunique()
    if number_of_games < 2:
        fail("at least 2 different games are needed to split by game, "
             f"but only {number_of_games} was found.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=RANDOM_SEED)
    train_index, test_index = next(splitter.split(df, groups=df["game_id"]))
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    # Safety checks
    overlap = set(train_df["game_id"]) & set(test_df["game_id"])
    if overlap:
        fail(f"{len(overlap)} game(s) appear in both training and test sets.")
    for name, part in (("training", train_df), ("test", test_df)):
        if part[TARGET_COLUMN].nunique() < 2:
            fail(f"the {name} set does not contain both positive and negative "
                 "labels. Use more games or change TEST_SIZE / RANDOM_SEED.")

    return train_df, test_df


# ---------------------------------------------------------------------------
# STEP 3: EVALUATION
# ---------------------------------------------------------------------------

def ranking_metrics(part_df, scores):
    """
    Ranking quality, calculated separately for each POSITION.

    For every position, all its legal candidates are sorted by model score and
    we look at where the move Hikaru actually played ended up (rank 1 = best).

    Ties are counted against the model (the real move is placed after all
    candidates with an equal score) so the numbers are never inflated.

    Also returns the expected results of a random guess for comparison.
    """
    ranked = pd.DataFrame({
        "position_id": part_df["position_id"].to_numpy(),
        "label": part_df[TARGET_COLUMN].to_numpy(),
        "score": scores,
    })
    ranked["rank"] = ranked.groupby("position_id")["score"].rank(
        method="max", ascending=False)

    candidates_per_position = ranked.groupby("position_id")["label"].size()
    actual = ranked[ranked["label"] == 1].set_index("position_id")
    ranks = actual["rank"]                      # one value per position
    n = candidates_per_position.loc[ranks.index].to_numpy()

    # Expected random-guess results: top-1 = 1/n, top-3 = min(3,n)/n,
    # MRR = (1 + 1/2 + ... + 1/n) / n
    harmonic = np.cumsum(1.0 / np.arange(1, n.max() + 1))

    return {
        "positions": int(len(ranks)),
        "top1": float((ranks <= 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "random_top1": float((1.0 / n).mean()),
        "random_top3": float((np.minimum(3, n) / n).mean()),
        "random_mrr": float((harmonic[n - 1] / n).mean()),
    }


def print_ranking(title, m):
    print(f"{title} ({m['positions']} positions)")
    print(f"  Top-1 accuracy : {m['top1'] * 100:6.2f}%   (random guess: {m['random_top1'] * 100:.2f}%)")
    print(f"  Top-3 accuracy : {m['top3'] * 100:6.2f}%   (random guess: {m['random_top3'] * 100:.2f}%)")
    print(f"  MRR            : {m['mrr']:6.4f}   (random guess: {m['random_mrr']:.4f})")


def evaluate_model(model, train_df, test_df, feature_columns):
    """Print and return all evaluation results."""
    y_test = test_df[TARGET_COLUMN]
    test_scores = model.predict_proba(test_df[feature_columns])[:, 1]
    test_predictions = (test_scores >= 0.5).astype(int)

    print("\n=== Row-level classification (test set, threshold 0.5) ===")
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_scores)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("  (Precision is low by design: only ~1 in 30 candidates is a positive and")
    print("   class weighting makes the model flag more moves. Ranking metrics below")
    print("   are the more meaningful measure for this project.)")

    print("\n=== Ranking quality per position ===")
    train_scores = model.predict_proba(train_df[feature_columns])[:, 1]
    train_ranking = ranking_metrics(train_df, train_scores)
    test_ranking = ranking_metrics(test_df, test_scores)
    print_ranking("Training set (for overfitting check only)", train_ranking)
    print_ranking("TEST set (unseen games)", test_ranking)

    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "roc_auc": float(roc_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_ranking": test_ranking,
        "train_ranking": train_ranking,
    }


# ---------------------------------------------------------------------------
# STEP 4: SAVE
# ---------------------------------------------------------------------------

def save_model(model, info):
    """
    Save the model and the info file. Both are written to temporary files first
    and only renamed once everything was written successfully.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, TEMP_MODEL_PATH)
        with open(TEMP_INFO_PATH, "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
        TEMP_MODEL_PATH.replace(MODEL_PATH)
        TEMP_INFO_PATH.replace(INFO_PATH)
    except Exception:
        TEMP_MODEL_PATH.unlink(missing_ok=True)
        TEMP_INFO_PATH.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    print("=== Train Model ===")
    print(f"Data file: {DATA_CSV}")

    # 1. Load and check the data
    df = load_data(DATA_CSV)
    feature_columns = choose_feature_columns(df)
    n_positions = df["position_id"].nunique()
    n_games = df["game_id"].nunique()
    n_positive = int(df[TARGET_COLUMN].sum())
    n_negative = len(df) - n_positive
    print(f"\nLoaded {len(df)} candidate rows from {n_positions} positions "
          f"in {n_games} games.")
    print(f"Positive rows: {n_positive}   Negative rows: {n_negative}   "
          f"(about 1 positive per {n_negative / n_positive:.0f} negatives)")
    print(f"Using {len(feature_columns)} feature columns. "
          f"Excluded metadata: {METADATA_COLUMNS}")

    # 2. Split by game
    train_df, test_df = split_by_game(df)
    print(f"\nSplit by game (seed {RANDOM_SEED}):")
    print(f"  Training: {train_df['game_id'].nunique()} games, "
          f"{train_df['position_id'].nunique()} positions, {len(train_df)} rows")
    print(f"  Test    : {test_df['game_id'].nunique()} games, "
          f"{test_df['position_id'].nunique()} positions, {len(test_df)} rows")
    print("  No game appears in both sets.")

    # 3. Train
    print("\nTraining RandomForestClassifier ... (this can take a minute)")
    model = RandomForestClassifier(**RF_SETTINGS)
    model.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
    print("Training finished.")

    # 4. Evaluate
    results = evaluate_model(model, train_df, test_df, feature_columns)

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    print("\n=== Ten most important features ===")
    for name, value in importances.sort_values(ascending=False).head(10).items():
        print(f"  {name:<36} {value:.4f}")

    # 5. Save (only reached if everything above succeeded)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model_type": "RandomForestClassifier",
        "model_settings": RF_SETTINGS,
        "sklearn_version": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "feature_columns": feature_columns,    # exact order used for training
        "number_of_features": len(feature_columns),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_games": int(train_df["game_id"].nunique()),
        "test_games": int(test_df["game_id"].nunique()),
        "training_positions": int(train_df["position_id"].nunique()),
        "test_positions": int(test_df["position_id"].nunique()),
        "test_results": results,
        "note": "Predicts resemblance to Hikaru's historical move choices, "
                "not the objectively best move.",
    }
    try:
        if MODEL_PATH.exists():
            print("\nNOTE: an older model exists; it is replaced now that "
                  "training succeeded.")
        save_model(model, info)
    except PermissionError:
        fail("could not write the model files. Are they open in another program? "
             "Any previous model was kept.")
    except Exception as error:
        fail(f"could not save the model ({error}). Any previous model was kept.")

    # 6. Summary
    test_rank = results["test_ranking"]
    print("\n=== Summary ===")
    print(f"Test-set Top-1 accuracy : {test_rank['top1'] * 100:.2f}%  "
          f"(random guess {test_rank['random_top1'] * 100:.2f}%)")
    print(f"Test-set Top-3 accuracy : {test_rank['top3'] * 100:.2f}%  "
          f"(random guess {test_rank['random_top3'] * 100:.2f}%)")
    print(f"Test-set MRR            : {test_rank['mrr']:.4f}  "
          f"(random guess {test_rank['random_mrr']:.4f})")
    print(f"Model saved to          : {MODEL_PATH}")
    print(f"Feature list saved to   : {INFO_PATH}")
    print(f"Time taken              : {time.time() - start_time:.1f} seconds")
    print("\nReminder: these scores measure how closely the model's ranking matches")
    print("Hikaru's historical choices on unseen games. They do not measure chess")
    print("strength or how good the moves are.")


if __name__ == "__main__":
    main()