"""
data_loader.py  -  Chess Move Predictor: Data Loader module

What this module does
---------------------
1. Shows a small Tkinter window where you choose a .pgn file and type the
   name of the player whose moves you want to model.
2. Validates every game in the file with python-chess (headers alone are
   NOT enough: the moves must be legal and readable).
3. Copies the original PGN into  data/raw/  (never overwriting anything).
4. Extracts, for every move made by the target player:
       the position BEFORE the move (FEN)  +  the move actually played (UCI)
   and saves everything as a CSV in  data/processed/.

It does NOT train any model. It only collects, validates, stores and
processes data.

Install:   pip install python-chess pandas
Run:       python data_loader.py
"""

import difflib
import logging
import queue
import re
import shutil
import sys
import threading
import tkinter as tk
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    import chess
    import chess.pgn
    import pandas as pd
except ImportError as import_error:  # friendly message for beginners
    print("A required package is missing:", import_error)
    print("Install everything with:  pip install python-chess pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# SETTINGS  (change these in one place)
# ---------------------------------------------------------------------------

# Minimum number of valid games required, and minimum number of valid games
# that must contain the target player.
MIN_GAMES = 100

# All paths are relative to THIS file, not to the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOG_FILE = BASE_DIR / "data_loader.log"

# Column order of the output CSV.
CSV_COLUMNS = [
    "game_id", "ply", "move_number", "fen", "side_to_move", "player_color",
    "target_player", "opponent", "move_uci", "move_san",
    "white", "black", "result", "date", "event", "site",
    "white_elo", "black_elo", "player_elo", "opponent_elo",
    "time_control", "termination", "eco", "opening",
]


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
logger = logging.getLogger("data_loader")


# ---------------------------------------------------------------------------
# SMALL HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def create_folders():
    """Create data/raw and data/processed if they do not exist yet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_header(value):
    """PGN uses '?' for unknown values. Turn those into an empty string."""
    if value is None:
        return ""
    value = str(value).strip()
    if value in ("?", "????.??.??", "-"):
        return ""
    return value


def normalize_name(name):
    """
    Make player names comparable.
      - ignores upper/lower case and extra spaces
      - 'Carlsen, Magnus' and 'Magnus Carlsen' are treated as the same name
    """
    text = clean_header(name).lower()
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        text = f"{first} {last}"
    return " ".join(text.split())


def safe_name(text):
    """Turn any text into something safe to use inside a Windows filename."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return cleaned if cleaned else "file"


def make_unique_path(folder, base_name, extension):
    """Return a path inside `folder` that does not exist yet."""
    path = folder / f"{base_name}{extension}"
    counter = 1
    while path.exists():
        path = folder / f"{base_name}_{counter}{extension}"
        counter += 1
    return path


def check_file_readable(path):
    """Return (True, '') if the file looks usable, otherwise (False, reason)."""
    try:
        if not path.exists():
            return False, "The file does not exist."
        if not path.is_file():
            return False, "The selected path is not a file."
        if path.stat().st_size == 0:
            return False, "The file is empty."
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            handle.read(1024)  # try reading a little bit
    except PermissionError:
        return False, "Permission denied: the file cannot be read."
    except OSError as error:
        return False, f"The file could not be read: {error}"
    return True, ""


# ---------------------------------------------------------------------------
# GAME VALIDATION AND PLAYER MATCHING
# ---------------------------------------------------------------------------

def check_game(game):
    """
    Decide whether a parsed game is valid.
    Returns (True, '') or (False, 'reason').

    python-chess replays every move while parsing. If a move is illegal or
    unreadable it stores a message in game.errors, which we check here.
    """
    variant = game.headers.get("Variant", "Standard").strip().lower()
    if variant not in ("standard", "chess"):
        return False, f"Unsupported chess variant ({variant})"

    if game.errors:
        return False, "Illegal or unreadable move text"

    try:
        game.board()  # builds the starting position (handles custom FEN)
    except ValueError:
        return False, "Invalid starting position (FEN)"

    if next(iter(game.mainline_moves()), None) is None:
        return False, "Game contains no moves"

    return True, ""


def find_player_color(white_name, black_name, target_norm):
    """
    Work out which side the target player played.
    Returns 'white', 'black', 'ambiguous' (name matches both sides) or None.
    Only exact matches (after normalizing) are accepted, so 'Magnus' will not
    accidentally match 'Magnus Carlsen'.
    """
    is_white = normalize_name(white_name) == target_norm
    is_black = normalize_name(black_name) == target_norm
    if is_white and is_black:
        return "ambiguous"
    if is_white:
        return "white"
    if is_black:
        return "black"
    return None


def suggest_player_names(target_norm, name_counts):
    """Suggest names from the file that look similar to what the user typed."""
    norm_to_raw = {}
    for raw_name in name_counts:
        norm = normalize_name(raw_name)
        if norm:
            norm_to_raw[norm] = raw_name

    suggestions = []
    for norm, raw in norm_to_raw.items():  # substring matches
        if len(norm) >= 3 and (target_norm in norm or norm in target_norm):
            suggestions.append(raw)
    for norm in difflib.get_close_matches(target_norm, list(norm_to_raw),
                                          n=5, cutoff=0.6):  # spelling-like
        if norm_to_raw[norm] not in suggestions:
            suggestions.append(norm_to_raw[norm])
    return suggestions[:5]


def extract_examples(game, game_id, color, target_name):
    """
    Replay one game and collect one row for each move made by the target
    player: the FEN BEFORE the move, plus the move actually played.
    Raises ValueError if an illegal move is found (extra safety check).
    """
    h = game.headers
    white, black = clean_header(h.get("White")), clean_header(h.get("Black"))
    white_elo, black_elo = clean_header(h.get("WhiteElo")), clean_header(h.get("BlackElo"))

    if color == "white":
        opponent, player_elo, opponent_elo = black, white_elo, black_elo
    else:
        opponent, player_elo, opponent_elo = white, black_elo, white_elo

    shared = {
        "game_id": game_id,
        "player_color": color,
        "target_player": target_name,
        "opponent": opponent,
        "white": white,
        "black": black,
        "result": clean_header(h.get("Result")),
        "date": clean_header(h.get("Date")),
        "event": clean_header(h.get("Event")),
        "site": clean_header(h.get("Site")),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "player_elo": player_elo,
        "opponent_elo": opponent_elo,
        "time_control": clean_header(h.get("TimeControl")),
        "termination": clean_header(h.get("Termination")),
        "eco": clean_header(h.get("ECO")),
        "opening": clean_header(h.get("Opening")),
    }

    rows = []
    board = game.board()
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        if not board.is_legal(move):
            raise ValueError(f"Illegal move {move.uci()} at ply {ply}")

        side = "white" if board.turn == chess.WHITE else "black"
        if side == color:  # only the target player's moves
            row = dict(shared)
            row["ply"] = ply
            row["move_number"] = board.fullmove_number
            row["fen"] = board.fen()          # position BEFORE the move
            row["side_to_move"] = side
            row["move_uci"] = move.uci()      # e.g. e2e4
            row["move_san"] = board.san(move)  # e.g. e4 (for humans)
            rows.append(row)
        board.push(move)
    return rows


# ---------------------------------------------------------------------------
# MAIN PROCESSING (runs in a background thread)
# ---------------------------------------------------------------------------

def process_pgn_file(pgn_path, target_player, min_games, progress_callback):
    """
    Read the whole PGN file, validate games, and collect training rows.
    Nothing is written to disk here. Returns a dictionary with the results.
    """
    stats = {"games_found": 0, "valid_games": 0, "invalid_games": 0,
             "player_games": 0, "examples": 0}
    skip_reasons = Counter()
    name_counts = Counter()      # how often each player name appears
    ambiguous_games = 0
    missing_name_games = 0
    rows = []
    notes = []                   # warnings shown to the user

    target_norm = normalize_name(target_player)
    file_size = max(pgn_path.stat().st_size, 1)

    with open(pgn_path, "r", encoding="utf-8-sig", errors="replace") as handle:
        while True:
            try:
                game = chess.pgn.read_game(handle)
            except Exception as error:  # very rare: severe parsing problem
                logger.exception("Parsing stopped early")
                notes.append(f"Reading stopped early because of a parsing error: {error}")
                break

            if game is None:         # end of file
                break

            stats["games_found"] += 1
            game_number = stats["games_found"]
            game_id = f"game_{game_number:06d}"

            is_valid, reason = check_game(game)
            if not is_valid:
                stats["invalid_games"] += 1
                skip_reasons[reason] += 1
                logger.info("Skipped %s: %s", game_id, reason)
            else:
                white_raw = game.headers.get("White", "")
                black_raw = game.headers.get("Black", "")
                for raw in (white_raw, black_raw):
                    if clean_header(raw):
                        name_counts[clean_header(raw)] += 1
                if not clean_header(white_raw) or not clean_header(black_raw):
                    missing_name_games += 1

                color = find_player_color(white_raw, black_raw, target_norm)
                if color == "ambiguous":
                    ambiguous_games += 1
                    stats["valid_games"] += 1
                    logger.info("%s: target name matches BOTH players, ignored", game_id)
                elif color is None:
                    stats["valid_games"] += 1       # valid, but target not in it
                else:
                    target_name_in_pgn = clean_header(white_raw if color == "white" else black_raw)
                    try:
                        game_rows = extract_examples(game, game_id, color, target_name_in_pgn)
                    except ValueError as error:
                        stats["invalid_games"] += 1
                        skip_reasons["Illegal move found while replaying"] += 1
                        logger.info("Skipped %s: %s", game_id, error)
                    else:
                        stats["valid_games"] += 1
                        stats["player_games"] += 1
                        stats["examples"] += len(game_rows)
                        rows.extend(game_rows)

            # Update the GUI every 25 games (checking the file position
            # on every game would slow things down).
            if game_number % 25 == 0:
                try:
                    percent = min(99.0, handle.tell() * 100.0 / file_size)
                except (OSError, ValueError):
                    percent = 0.0
                progress_callback(dict(stats), percent)

    progress_callback(dict(stats), 100.0)

    # ---- Decide whether the data is good enough ----------------------------
    problems = []
    if stats["games_found"] == 0:
        problems.append("No chess games were found. The file may be empty or not valid PGN.")
    elif stats["valid_games"] < min_games:
        problems.append(
            f"Only {stats['valid_games']} valid game(s) were found, "
            f"but at least {min_games} are required.")

    if stats["valid_games"] > 0 and stats["player_games"] < min_games:
        problems.append(
            f"Player '{target_player}' appears in only {stats['player_games']} valid game(s); "
            f"at least {min_games} are required.")
        similar = suggest_player_names(target_norm, name_counts)
        if similar:
            notes.append("Similar names found in the file: " + ", ".join(similar))
        if name_counts:
            common = ", ".join(f"{name} ({count})" for name, count in name_counts.most_common(5))
            notes.append("Most common player names in this file: " + common)
        notes.append("Names must match the PGN headers exactly (case and spacing are ignored; "
                     "'Last, First' and 'First Last' are treated as the same).")

    if ambiguous_games:
        notes.append(f"{ambiguous_games} game(s) were ignored because the name matched both "
                     f"White and Black.")
    if missing_name_games:
        notes.append(f"{missing_name_games} valid game(s) have a missing player name ('?').")
    if skip_reasons:
        details = "; ".join(f"{count} x {reason}" for reason, count in skip_reasons.items())
        notes.append(f"Skipped invalid games: {details}")

    return {"stats": stats, "rows": rows, "problems": problems, "notes": notes}


def save_outputs(pgn_path, target_player, rows):
    """Copy the raw PGN to data/raw and write the CSV to data/processed."""
    create_folders()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_base = f"{safe_name(pgn_path.stem)}_{stamp}"
    raw_path = make_unique_path(RAW_DIR, raw_base, ".pgn")
    shutil.copy2(pgn_path, raw_path)          # original file is left untouched

    csv_base = f"{safe_name(pgn_path.stem)}_{safe_name(target_player)}_{stamp}_positions"
    csv_path = make_unique_path(PROCESSED_DIR, csv_base, ".csv")
    data_frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    data_frame.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info("Saved raw PGN to %s and CSV to %s", raw_path, csv_path)
    return raw_path, csv_path


def worker(pgn_path, target_player, messages):
    """Background thread: does the heavy work and reports through a queue."""
    try:
        def report(stats, percent):
            messages.put(("progress", stats, percent))

        result = process_pgn_file(pgn_path, target_player, MIN_GAMES, report)

        if result["problems"]:
            result["success"] = False
        else:
            raw_path, csv_path = save_outputs(pgn_path, target_player, result["rows"])
            result["success"] = True
            result["raw_path"] = raw_path
            result["csv_path"] = csv_path

        result["rows"] = None   # free memory; the GUI does not need the rows
        messages.put(("done", result))
    except Exception as error:
        logger.exception("Unexpected error during processing")
        messages.put(("error", f"Unexpected error: {error}"))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DataLoaderApp:
    """The Tkinter window."""

    STAT_LABELS = [
        ("file", "Selected file"),
        ("player", "Target player"),
        ("games_found", "Games discovered"),
        ("valid_games", "Valid games processed"),
        ("invalid_games", "Invalid games skipped"),
        ("player_games", "Valid games containing target player"),
        ("examples", "Training examples extracted"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Chess Move Predictor — Data Loader")
        self.root.geometry("780x720")

        self.selected_path = None
        self.messages = queue.Queue()   # worker thread -> GUI
        self.processing = False
        self.stat_vars = {key: tk.StringVar(value="-") for key, _ in self.STAT_LABELS}
        self.player_var = tk.StringVar()

        self.build_widgets()

    # ----- building the window ------------------------------------------
    def build_widgets(self):
        pad = {"padx": 12, "pady": 4}

        tk.Label(self.root, text="Chess Move Predictor — Data Loader",
                 font=("Segoe UI", 16, "bold")).pack(**pad)
        tk.Label(self.root,
                 text=f"Please upload a PGN file containing the selected player's "
                      f"chess games (minimum {MIN_GAMES} games).",
                 wraplength=720).pack(**pad)

        # file selection
        file_frame = tk.Frame(self.root)
        file_frame.pack(fill="x", **pad)
        tk.Button(file_frame, text="Choose PGN File",
                  command=self.choose_file).grid(row=0, column=0, sticky="w")
        self.file_label = tk.Label(file_frame, text="No file selected",
                                   anchor="w", justify="left", wraplength=720)
        self.file_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        # player name
        player_frame = tk.Frame(self.root)
        player_frame.pack(fill="x", **pad)
        tk.Label(player_frame, text="Target player's name / username:").grid(row=0, column=0, sticky="w")
        tk.Entry(player_frame, textvariable=self.player_var, width=40).grid(row=0, column=1, padx=8)

        # process button
        self.process_button = tk.Button(self.root, text="Process Games",
                                        command=self.start_processing)
        self.process_button.pack(**pad)

        # progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal",
                                        length=600, mode="determinate", maximum=100)
        self.progress.pack(**pad)

        # statistics
        stats_frame = tk.LabelFrame(self.root, text="Progress")
        stats_frame.pack(fill="x", **pad)
        for row, (key, label) in enumerate(self.STAT_LABELS):
            tk.Label(stats_frame, text=label + ":", anchor="w").grid(
                row=row, column=0, sticky="w", padx=8, pady=1)
            tk.Label(stats_frame, textvariable=self.stat_vars[key], anchor="w",
                     wraplength=450, justify="left").grid(
                row=row, column=1, sticky="w", padx=8, pady=1)

        # messages
        tk.Label(self.root, text="Messages:", anchor="w").pack(fill="x", padx=12)
        self.message_box = scrolledtext.ScrolledText(self.root, height=12, wrap="word",
                                                     state="disabled")
        self.message_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    # ----- small GUI helpers ----------------------------------------------
    def log_message(self, text):
        self.message_box.configure(state="normal")
        self.message_box.insert("end", text + "\n")
        self.message_box.see("end")
        self.message_box.configure(state="disabled")

    def update_stats(self, stats):
        for key in ("games_found", "valid_games", "invalid_games", "player_games", "examples"):
            self.stat_vars[key].set(str(stats[key]))

    def reset_stats(self):
        for key in ("games_found", "valid_games", "invalid_games", "player_games", "examples"):
            self.stat_vars[key].set("0")
        self.progress["value"] = 0

    # ----- button actions ---------------------------------------------------
    def choose_file(self):
        if self.processing:
            return
        filename = filedialog.askopenfilename(
            title="Choose a PGN file",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")])
        if not filename:
            return
        self.selected_path = Path(filename)
        self.file_label.config(
            text=f"File: {self.selected_path.name}\nLocation: {self.selected_path.parent}")
        self.stat_vars["file"].set(self.selected_path.name)
        self.log_message(f"Selected: {self.selected_path}")

    def start_processing(self):
        if self.processing:
            return
        if self.selected_path is None:
            messagebox.showwarning("No file", "Please choose a PGN file first.")
            return
        player = self.player_var.get().strip()
        if not player:
            messagebox.showwarning("No player", "Please enter the target player's name.")
            return

        ok, reason = check_file_readable(self.selected_path)
        if not ok:
            self.log_message(f"ERROR: {reason}")
            messagebox.showerror("Cannot read file", reason)
            return

        self.processing = True
        self.process_button.config(state="disabled")
        self.stat_vars["player"].set(player)
        self.reset_stats()
        self.log_message(f"Processing started for player '{player}'...")

        thread = threading.Thread(
            target=worker, args=(self.selected_path, player, self.messages), daemon=True)
        thread.start()
        self.root.after(100, self.check_messages)

    def check_messages(self):
        """Runs in the GUI thread: reads messages sent by the worker thread."""
        finished = False
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self.update_stats(message[1])
                    self.progress["value"] = message[2]
                elif kind == "done":
                    self.show_result(message[1])
                    finished = True
                elif kind == "error":
                    self.log_message("ERROR: " + message[1])
                    messagebox.showerror("Error", message[1])
                    finished = True
        except queue.Empty:
            pass

        if finished:
            self.processing = False
            self.process_button.config(state="normal")
        else:
            self.root.after(100, self.check_messages)

    def show_result(self, result):
        self.update_stats(result["stats"])
        for note in result["notes"]:
            self.log_message("NOTE: " + note)

        if result["success"]:
            self.progress["value"] = 100
            self.log_message("SUCCESS: processing finished.")
            self.log_message(f"Raw PGN saved to:   {result['raw_path']}")
            self.log_message(f"Processed CSV saved to: {result['csv_path']}")
            messagebox.showinfo(
                "Success",
                f"Processing finished.\n\nTraining examples: {result['stats']['examples']}\n\n"
                f"PGN copy:\n{result['raw_path']}\n\nCSV file:\n{result['csv_path']}")
        else:
            for problem in result["problems"]:
                self.log_message("ERROR: " + problem)
            self.log_message("Nothing was saved because the data did not meet the requirements.")
            messagebox.showerror("Not enough data", "\n\n".join(result["problems"]))


def main():
    create_folders()
    logger.info("Starting Data Loader. Base folder: %s", BASE_DIR)
    root = tk.Tk()
    DataLoaderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()