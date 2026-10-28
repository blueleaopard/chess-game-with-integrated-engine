"""
data_loader.py  -  Chess Move Predictor: Data Loader module

What this module does
---------------------
1. Shows a small Tkinter window where you choose a .pgn file and type the
   name of the player whose moves you want to model.
2. Reads the PGN one game at a time (the whole file is never held in memory)
   and validates every game with python-chess (headers alone are NOT enough:
   the moves must be legal and readable).
3. Recognizes the target player in the White / Black headers with an explicit,
   documented rule (see "NAME MATCHING" below), counting each game only once.
4. Copies the original PGN into  data/raw/  (never overwriting anything).
5. Extracts, for every move made by the target player:
       the position BEFORE the move (FEN)  +  the move actually played (UCI)
   and saves everything as ONE CSV per player in  data/processed/
   (Hikaru Nakamura -> hikaru_positions.csv). The CSV is written to a temporary
   file first and only replaces the old CSV if the whole run succeeds.
6. Shows a processing summary: games found, games recognized, games processed,
   games skipped (grouped by reason), positions extracted, output path.

It does NOT train any model. It only collects, validates, stores and
processes data.

NAME MATCHING (the explicit rule)
---------------------------------
A header name is accepted as the target player only in these three cases:
  1. full name      - same surname AND same first name
                      ('Hikaru Nakamura', 'Nakamura, Hikaru', 'Nakamura,Hikaru')
  2. abbreviated    - same surname AND the given name is the START of the
                      target's first name ('Nakamura,Hi', 'Nakamura,H', 'H. Nakamura')
                      This is NOT proof of identity: 'Nakamura,Hi' could in theory
                      be another Nakamura whose first name starts with 'Hi'.
                      Set ACCEPT_ABBREVIATED_NAMES = False for strict matching.
  3. username/alias - the whole header equals a username listed in KNOWN_PLAYERS
                      ('Hikaru')
Anything else is rejected, even if it contains similar text
('Nakamura, Ryo', 'Nakamura,Hiroshi', 'Hikaru Tanaka').

Install:   pip install python-chess pandas
Run:       python data_loader.py
"""

import difflib
import hashlib
import logging
import queue
import re
import shutil
import statistics
import sys
import threading
import tkinter as tk
import unicodedata
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

# Minimum number of valid games required, and minimum number of games that
# must be successfully processed for the target player.
MIN_GAMES = 100

# Accept abbreviated names such as 'Nakamura,Hi' or 'Nakamura,H'?
# True  = accept them (more games, small identity risk, reported in the summary)
# False = accept only full names and listed usernames
ACCEPT_ABBREVIATED_NAMES = True

# Skip a game if an identical one (same players, date, result, moves) was
# already processed. Duplicates would otherwise end up in both the training
# and the test set later on.
SKIP_DUPLICATE_GAMES = True

# Players whose different spellings we know about. The key is the normalized
# full name ("first last"). Typing the key, 'Last, First' or any username
# below into the GUI selects this player.
KNOWN_PLAYERS = {
    "hikaru nakamura": {
        "given": "hikaru",
        "surname": "nakamura",
        "usernames": ["hikaru"],
        "csv_name": "hikaru_positions.csv",
    },
}

# PGN 'Variant' values that are ordinary chess ('From Position' = normal chess
# started from a custom FEN, as exported by Lichess).
ACCEPTED_VARIANTS = ("standard", "chess", "from position")

# Chess titles removed from names before matching ('GM Hikaru Nakamura').
TITLE_WORDS = {"gm", "im", "fm", "cm", "wgm", "wim", "wfm", "wcm", "nm", "dr"}

# The CSV is written to disk in chunks of this many rows (saves memory).
CHUNK_ROWS = 20000

# All paths are relative to THIS file, not to the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOG_FILE = BASE_DIR / "data_loader.log"

# Column order of the output CSV (unchanged, feature_extractor.py relies on it).
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
# NAME HANDLING
# ---------------------------------------------------------------------------

def clean_name_text(name):
    """
    Prepare a header name for comparison:
      lower case, accents removed, '(USA)' / '[2800]' removed, dots removed
      ('H.' -> 'h'), spaces around commas removed.
    """
    text = clean_header(name)
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[\(\[].*?[\)\]]", " ", text)
    text = text.replace(".", " ")
    return re.sub(r"\s*,\s*", ",", text)


def normalize_name(name):
    """
    One comparable string per name, always in 'first last' order:
      'Nakamura, Hikaru'  -> 'hikaru nakamura'
      'Nakamura,Hi'       -> 'hi nakamura'
      '  GM HIKARU  '     -> 'hikaru'
    """
    text = clean_name_text(name)
    if "," in text:
        last, first = text.split(",", 1)
        text = f"{first} {last}"
    return " ".join(t for t in text.split() if t not in TITLE_WORDS)


def name_readings(name):
    """
    Possible (surname, first given name) readings of a header name.
      'Nakamura,Hi'     -> [('nakamura', 'hi')]            (comma = clear order)
      'Hikaru Nakamura' -> [('nakamura', 'hikaru'), ('hikaru', 'nakamura')]
      'Hikaru'          -> []                              (single word)
    Without a comma we cannot know the order, so both readings are returned.
    """
    text = clean_name_text(name)
    if "," in text:
        surname, given = text.split(",", 1)
        surname = " ".join(surname.split())
        given_tokens = [t for t in given.replace(",", " ").split()
                        if t not in TITLE_WORDS]
        if surname and given_tokens:
            return [(surname, given_tokens[0])]
        return []
    tokens = [t for t in text.split() if t not in TITLE_WORDS]
    if len(tokens) >= 2:
        return [(tokens[-1], tokens[0]), (tokens[0], tokens[-1])]
    return []


def build_matcher(typed_name):
    """
    Turn the name typed into the GUI into a small 'matcher' dictionary that
    describes who we are looking for.
    """
    typed_norm = normalize_name(typed_name)

    for canonical, info in KNOWN_PLAYERS.items():
        if typed_norm == canonical or typed_norm in info["usernames"]:
            return {"label": canonical, "given": info["given"],
                    "surname": info["surname"],
                    "usernames": set(info["usernames"]),
                    "csv_name": info.get("csv_name")}

    readings = name_readings(typed_name)
    if readings:                                   # e.g. 'Magnus Carlsen'
        surname, given = readings[0]
        return {"label": typed_norm, "given": given, "surname": surname,
                "usernames": set(), "csv_name": None}

    # A single word, probably a username: only exact matches are accepted
    return {"label": typed_norm, "given": None, "surname": None,
            "usernames": {typed_norm}, "csv_name": None}


def match_player_name(raw_name, matcher):
    """
    Decide whether a header name is the target player.
    Returns 'full name', 'abbreviated name', 'username/alias' or None.
    """
    whole = normalize_name(raw_name)
    if not whole:
        return None
    if whole in matcher["usernames"]:
        return "username/alias"
    if matcher["surname"] is None:
        return None

    for surname, given in name_readings(raw_name):
        if surname != matcher["surname"]:
            continue                               # different surname: reject
        if given == matcher["given"]:
            return "full name"
        if (ACCEPT_ABBREVIATED_NAMES and given
                and len(given) < len(matcher["given"])
                and matcher["given"].startswith(given)):
            return "abbreviated name"
    return None


def name_is_near_miss(raw_name, matcher):
    """True if the name contains the target's surname but was NOT accepted."""
    if matcher["surname"] is None:
        return False
    text = clean_name_text(raw_name).replace(",", " ")
    return f" {matcher['surname']} " in f" {' '.join(text.split())} "


def find_player_color(white_name, black_name, matcher):
    """
    Work out which side the target player played.
    Returns (color, white_category, black_category) where color is
    'white', 'black', 'ambiguous' (matches BOTH sides) or None.
    One game gives exactly one answer, so a game is never counted twice.
    """
    white_category = match_player_name(white_name, matcher)
    black_category = match_player_name(black_name, matcher)
    if white_category and black_category:
        color = "ambiguous"
    elif white_category:
        color = "white"
    elif black_category:
        color = "black"
    else:
        color = None
    return color, white_category, black_category


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


# ---------------------------------------------------------------------------
# GAME VALIDATION AND EXTRACTION
# ---------------------------------------------------------------------------

def check_game(game):
    """
    Decide whether a parsed game is valid.
    Returns (True, '', '') or (False, 'short reason', 'technical detail').

    python-chess replays every move while parsing. If a move is illegal or
    unreadable it stores the problem in game.errors, which we check here.
    """
    variant = game.headers.get("Variant", "Standard").strip().lower()
    if variant not in ACCEPTED_VARIANTS:
        return False, f"Unsupported chess variant ({variant})", ""

    if game.errors:
        detail = str(game.errors[0])
        message = detail.lower()
        if "illegal san" in message:
            reason = "Illegal move in the move text"
        elif "ambiguous san" in message:
            reason = "Ambiguous move notation"
        elif "invalid san" in message:
            reason = "Unreadable move notation"
        else:
            reason = "Other PGN parsing error"
        return False, reason, detail

    try:
        game.board()  # builds the starting position (handles custom FEN)
    except ValueError as error:
        return False, "Invalid starting position (FEN)", str(error)

    if next(iter(game.mainline_moves()), None) is None:
        return False, "Game contains no moves", ""

    return True, "", ""


def game_fingerprint(game):
    """
    A short code that identifies a game by players, date, result, start
    position and moves. Two copies of the same game give the same code.
    """
    h = game.headers
    moves = " ".join(move.uci() for move in game.mainline_moves())
    text = "|".join([
        normalize_name(h.get("White", "")), normalize_name(h.get("Black", "")),
        clean_header(h.get("Date")), clean_header(h.get("Result")),
        clean_header(h.get("FEN")), moves,
    ])
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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


def record_match_type(tier_info, category, game, color):
    """Remember Elo and year of a processed game, grouped by how the name matched."""
    info = tier_info.setdefault(category, {"games": 0, "elos": [], "years": []})
    info["games"] += 1
    elo_text = clean_header(game.headers.get("WhiteElo" if color == "white" else "BlackElo"))
    if elo_text.isdigit():
        info["elos"].append(int(elo_text))
    date_text = clean_header(game.headers.get("Date"))
    if date_text[:4].isdigit():
        info["years"].append(int(date_text[:4]))


# ---------------------------------------------------------------------------
# MAIN PROCESSING (runs in a background thread)
# ---------------------------------------------------------------------------

def get_csv_path(target_player):
    """ONE fixed CSV filename per player (hikaru_positions.csv for Hikaru)."""
    matcher = build_matcher(target_player)
    if matcher["csv_name"]:
        return PROCESSED_DIR / matcher["csv_name"]
    return PROCESSED_DIR / f"{safe_name(matcher['label']).lower()}_positions.csv"


def append_rows_to_csv(rows, csv_path, first_chunk):
    """Write rows to the (temporary) CSV. The first chunk creates the file."""
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame.to_csv(csv_path, mode="w" if first_chunk else "a",
                 header=first_chunk, index=False, encoding="utf-8")


def process_pgn_file(pgn_path, target_player, min_games, progress_callback, temp_csv_path):
    """
    Read the PGN file game by game, validate games, and write the target
    player's rows to `temp_csv_path` in chunks. Returns a dictionary with
    the results. Nothing is written to the final CSV here.
    """
    stats = {"games_found": 0, "valid_games": 0, "invalid_games": 0,
             "matched_games": 0, "player_games": 0, "skipped_games": 0,
             "examples": 0}
    skipped_target = Counter()   # reasons for skipped games of the target player
    invalid_other = Counter()    # reasons for invalid games of OTHER players
    name_counts = Counter()      # how often each player name appears
    spellings = Counter()        # (header spelling, match type) -> games
    near_misses = Counter()      # same surname but NOT accepted
    tier_info = {}               # match type -> games, Elo list, year list
    seen_games = set()           # fingerprints of processed games
    missing_name_games = 0
    notes = []                   # warnings shown to the user
    buffer = []                  # rows waiting to be written
    wrote_any = False

    matcher = build_matcher(target_player)
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

            white_raw = game.headers.get("White", "")
            black_raw = game.headers.get("Black", "")
            white_clean, black_clean = clean_header(white_raw), clean_header(black_raw)
            date_text = clean_header(game.headers.get("Date"))
            description = f"{game_id} ({white_clean or '?'} vs {black_clean or '?'}, {date_text or '?'})"

            for name in (white_clean, black_clean):
                if name:
                    name_counts[name] += 1
            if not white_clean or not black_clean:
                missing_name_games += 1

            # 1) Is the target player in this game? (decided once per game)
            color, white_category, black_category = find_player_color(
                white_raw, black_raw, matcher)
            for name, category in ((white_clean, white_category), (black_clean, black_category)):
                if category:
                    spellings[(name, category)] += 1
                elif name and name_is_near_miss(name, matcher):
                    near_misses[name] += 1

            # 2) Is the game itself valid?
            is_valid, reason, detail = check_game(game)
            if is_valid:
                stats["valid_games"] += 1
            else:
                stats["invalid_games"] += 1

            # 3) Decide what happens to the game
            if color is None:
                if not is_valid:
                    invalid_other[reason] += 1
                    logger.info("Ignored invalid game, target not in it: %s: %s | %s",
                                description, reason, detail)
            else:
                stats["matched_games"] += 1
                game_rows = None
                skip_reason = None
                skip_detail = ""
                fingerprint = None

                if not is_valid:
                    skip_reason, skip_detail = reason, detail
                elif color == "ambiguous":
                    skip_reason = "Target name matches BOTH White and Black"
                else:
                    if SKIP_DUPLICATE_GAMES:
                        fingerprint = game_fingerprint(game)
                        if fingerprint in seen_games:
                            skip_reason = "Duplicate game (same players, date, result and moves)"
                    if skip_reason is None:
                        target_name = white_clean if color == "white" else black_clean
                        try:
                            game_rows = extract_examples(game, game_id, color, target_name)
                        except ValueError as error:
                            skip_reason = "Illegal move found while replaying"
                            skip_detail = str(error)
                        else:
                            if not game_rows:
                                skip_reason = "Target player made no moves in this game"

                if skip_reason is not None:
                    stats["skipped_games"] += 1
                    skipped_target[skip_reason] += 1
                    logger.info("Skipped %s: %s %s", description, skip_reason,
                                f"| {skip_detail}" if skip_detail else "")
                else:
                    stats["player_games"] += 1
                    stats["examples"] += len(game_rows)
                    if fingerprint:
                        seen_games.add(fingerprint)
                    category = white_category if color == "white" else black_category
                    record_match_type(tier_info, category, game, color)

                    buffer.extend(game_rows)
                    if len(buffer) >= CHUNK_ROWS:      # write to disk, free memory
                        append_rows_to_csv(buffer, temp_csv_path, first_chunk=not wrote_any)
                        wrote_any = True
                        buffer = []

            # Update the GUI every 25 games (checking the file position
            # on every game would slow things down).
            if game_number % 25 == 0:
                try:
                    percent = min(99.0, handle.tell() * 100.0 / file_size)
                except (OSError, ValueError):
                    percent = 0.0
                progress_callback(dict(stats), percent)

    if buffer:
        append_rows_to_csv(buffer, temp_csv_path, first_chunk=not wrote_any)
        wrote_any = True
    progress_callback(dict(stats), 100.0)

    if stats["matched_games"] != stats["player_games"] + stats["skipped_games"]:
        logger.warning("Internal counts do not add up: %s", stats)

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
            f"Player '{target_player}' was successfully processed in only "
            f"{stats['player_games']} game(s) ({stats['matched_games']} game(s) contained a "
            f"recognized name, {stats['skipped_games']} of them were skipped); "
            f"at least {min_games} are required.")
        similar = suggest_player_names(matcher["label"], name_counts)
        if similar:
            notes.append("Similar names found in the file: " + ", ".join(similar))
        if name_counts:
            common = ", ".join(f"{name} ({count})" for name, count in name_counts.most_common(5))
            notes.append("Most common player names in this file: " + common)
        notes.append("Accepted spellings: full name ('First Last' or 'Last, First'), "
                     "abbreviated first name ('Last,Fi'), or a known username. "
                     "Case, spacing and chess titles are ignored.")

    if missing_name_games:
        notes.append(f"{missing_name_games} game(s) have a missing player name ('?').")

    return {"stats": stats, "problems": problems, "notes": notes,
            "target_label": matcher["label"], "skipped_target": skipped_target,
            "invalid_other": invalid_other, "spellings": spellings,
            "near_misses": near_misses, "tier_info": tier_info}


def format_summary(result, csv_path):
    """Build the processing summary as a list of text lines."""
    s = result["stats"]
    lines = ["", "=" * 64,
             f"PROCESSING SUMMARY   (target player: {result['target_label']})",
             "=" * 64,
             f"Total games encountered              : {s['games_found']}",
             f"Games containing a recognized name   : {s['matched_games']}",
             f"Games successfully processed         : {s['player_games']}",
             f"Games skipped (recognized, not used) : {s['skipped_games']}"]
    for reason, count in result["skipped_target"].most_common():
        lines.append(f"      - {count} x {reason}")

    others = s["games_found"] - s["matched_games"]
    lines.append(f"Other games (name not recognized)    : {others}")
    if result["invalid_other"]:
        lines.append(f"      of which invalid/unreadable    : {sum(result['invalid_other'].values())}")
        for reason, count in result["invalid_other"].most_common():
            lines.append(f"      - {count} x {reason}")

    lines.append(f"Positions extracted for the player   : {s['examples']}")
    if s["player_games"]:
        lines.append(f"Average positions per processed game : {s['examples'] / s['player_games']:.1f}")
    lines.append(f"Output CSV                           : {csv_path if csv_path else 'not saved'}")

    if result["spellings"]:
        lines.append("")
        lines.append("Accepted name spellings (games per spelling):")
        for (name, category), count in result["spellings"].most_common(15):
            lines.append(f"      '{name}'  [{category}]  x {count}")

    if result["tier_info"]:
        lines.append("Processed games by match type:")
        for category, info in result["tier_info"].items():
            elo_text = (f"Elo {min(info['elos'])}-{max(info['elos'])} "
                        f"(median {int(statistics.median(info['elos']))})"
                        if info["elos"] else "Elo not available")
            year_text = (f"years {min(info['years'])}-{max(info['years'])}"
                         if info["years"] else "years not available")
            lines.append(f"      {category}: {info['games']} games, {elo_text}, {year_text}")

    if "abbreviated name" in result["tier_info"]:
        lines.append("NOTE: abbreviated names (e.g. 'Nakamura,Hi') are accepted because the "
                     "surname matches exactly and the given name is the start of the target's "
                     "first name. This cannot PROVE the player is the same person. Compare "
                     "the Elo/year ranges above with the 'full name' group; set "
                     "ACCEPT_ABBREVIATED_NAMES = False for strict matching.")

    if result["near_misses"]:
        lines.append("Names with the same surname that were NOT accepted:")
        for name, count in result["near_misses"].most_common(10):
            lines.append(f"      '{name}'  x {count}")
    return lines


def save_outputs(pgn_path, csv_path, temp_csv_path):
    """
    Copy the raw PGN to data/raw (unique name) and move the finished temporary
    CSV over the final CSV. Only called after processing succeeded, so an
    unsuccessful run never replaces a valid CSV.
    """
    create_folders()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_base = f"{safe_name(pgn_path.stem)}_{stamp}"
    raw_path = make_unique_path(RAW_DIR, raw_base, ".pgn")
    shutil.copy2(pgn_path, raw_path)          # original file is left untouched

    temp_csv_path.replace(csv_path)           # replaces the old CSV in one step
    logger.info("Saved raw PGN to %s and CSV to %s", raw_path, csv_path)
    return raw_path, csv_path


def worker(pgn_path, target_player, messages):
    """Background thread: does the heavy work and reports through a queue."""
    temp_csv_path = None
    try:
        create_folders()
        csv_path = get_csv_path(target_player)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_csv_path = csv_path.with_name(f"{csv_path.name}.{stamp}.tmp")

        def report(stats, percent):
            messages.put(("progress", stats, percent))

        result = process_pgn_file(pgn_path, target_player, MIN_GAMES, report, temp_csv_path)

        if result["problems"]:
            result["success"] = False
            saved_csv = None
        else:
            raw_path, saved_csv = save_outputs(pgn_path, csv_path, temp_csv_path)
            result["success"] = True
            result["raw_path"] = raw_path
            result["csv_path"] = saved_csv

        result["summary"] = format_summary(result, saved_csv)
        for line in result["summary"]:
            logger.info(line)
        messages.put(("done", result))
    except PermissionError as error:
        logger.exception("Permission error while saving")
        messages.put(("error", "Could not write or replace the output file. Is the CSV open in "
                               f"Excel? Close it and try again. The previous CSV was kept. ({error})"))
    except Exception as error:
        logger.exception("Unexpected error during processing")
        messages.put(("error", f"Unexpected error: {error}"))
    finally:
        if temp_csv_path is not None:
            temp_csv_path.unlink(missing_ok=True)   # no leftover temporary file


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class DataLoaderApp:
    """The Tkinter window."""

    STAT_LABELS = [
        ("file", "Selected file"),
        ("player", "Target player"),
        ("games_found", "Games discovered"),
        ("matched_games", "Games with a recognized player name"),
        ("player_games", "Games successfully processed"),
        ("skipped_games", "Player games skipped (see messages)"),
        ("valid_games", "Valid games (all players)"),
        ("invalid_games", "Invalid games (all players)"),
        ("examples", "Training examples extracted"),
    ]
    STAT_KEYS = [key for key, _ in STAT_LABELS if key not in ("file", "player")]

    def __init__(self, root):
        self.root = root
        self.root.title("Chess Move Predictor — Data Loader")
        self.root.geometry("780x780")

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
        for key in self.STAT_KEYS:
            self.stat_vars[key].set(str(stats[key]))

    def reset_stats(self):
        for key in self.STAT_KEYS:
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
        if not normalize_name(player):
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
        for line in result["summary"]:
            self.log_message(line)

        if result["success"]:
            self.progress["value"] = 100
            self.log_message("")
            self.log_message("SUCCESS: processing finished.")
            self.log_message(f"Raw PGN saved to:   {result['raw_path']}")
            self.log_message(f"Processed CSV saved to: {result['csv_path']}")
            messagebox.showinfo(
                "Success",
                f"Processing finished.\n\n"
                f"Games processed: {result['stats']['player_games']}\n"
                f"Training examples: {result['stats']['examples']}\n\n"
                f"PGN copy:\n{result['raw_path']}\n\nCSV file:\n{result['csv_path']}")
        else:
            for problem in result["problems"]:
                self.log_message("ERROR: " + problem)
            self.log_message("Nothing was saved because the data did not meet the requirements. "
                             "Any previous CSV was kept.")
            messagebox.showerror("Not enough data", "\n\n".join(result["problems"]))


def main():
    create_folders()
    logger.info("Starting Data Loader. Base folder: %s", BASE_DIR)
    root = tk.Tk()
    DataLoaderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()