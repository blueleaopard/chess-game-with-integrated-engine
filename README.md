# ChessLab backend update

This bundle connects the existing PGN loader, feature extractor, and Random Forest training script to the ChessLab frontend.

## Install

Open a terminal in the folder where you extract this bundle and install the required packages:

```powershell
py -m pip install flask flask-cors python-chess pandas numpy scikit-learn joblib
```

If `py` does not work on your computer, use `python` instead.

## Run

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Do not open `index.html` directly from File Explorer.

## What it does

- Uploads a `.pgn` file and accepts a target player's name and a model label.
- Uses your supplied PGN data loader and its game validation / player matching.
- Requires at least 100 valid games and at least 100 successfully processed games for the target player.
- Extracts candidate-move features, then trains and evaluates the Random Forest.
- Shows progress and console logs in the page.
- Stores each model and its data in its own folder under `models/player_workspaces/`; training a new player does not overwrite an earlier player's model.
- `/api/status` checks the backend, `/api/models` lists saved models, `/api/progress/<task_id>` reports training progress.

## Important notes

- Large PGNs can take a long time. Random Forest `.fit()` does not expose reliable percentage progress, so the interface reports the training stage instead of inventing a precise percentage.
- Top-1 / Top-3 / MRR measure whether the model ranks a player's historical move highly on held-out games. They are not the chance the player will make that move, and they do not measure chess strength or Stockfish agreement.
- This update only connects the **model-building page**. The interactive FEN/chessboard analysis page and Stockfish comparison are still a later step.
- The supplied feature extractor is adjusted to find the latest `*_positions.csv`, so it works with players other than Hikaru too.
- Keep the `project_files` folder beside `app.py`; it contains the scripts that the backend copies into each isolated model workspace.


## Chess Analysis page (first working version)

After starting `python app.py`, open `http://127.0.0.1:5000/analysis` or click **Chess analysis** in the sidebar. Select a saved player model, paste a valid FEN, and click **Predict top moves**. This version scores every legal move with the saved Random Forest and displays the top three. Scores are ranking scores, not calibrated probabilities or evaluations of move quality.

The first analysis page accepts FEN text rather than interactive board moves. Stockfish comparison and an interactive board are not included yet; they are the next implementation step.
