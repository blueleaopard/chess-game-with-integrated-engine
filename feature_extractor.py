"""
feature_extractor.py  -  Chess Move Predictor: Feature Extractor module

What this module does
---------------------
Reads   the newest  data/processed/*_positions.csv  file created by
        data_loader.py  (one row = one move played by the target player)
and writes data/processed/hikaru_move_features.csv

For every position it:
  1. rebuilds the board from the FEN,
  2. lists ALL legal moves (the "candidate moves"),
  3. gives the move Hikaru actually played the label 1 and every other
     legal move the label 0,
  4. computes numerical features for each candidate move.

The output is meant for a scikit-learn classifier (train_model.py) that learns
which moves look like Hikaru's choices. No engine (Stockfish) is used.

Install:   pip install python-chess pandas
Run:       python feature_extractor.py      (from the project root)
"""

import sys
import time
from collections import Counter
from pathlib import Path

try:
    import chess
    import pandas as pd
except ImportError as import_error:  # friendly message for beginners
    print("A required package is missing:", import_error)
    print("Install everything with:  pip install python-chess pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

# Paths are relative to THIS file, not to the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
POSITIONS_PATTERN = "*_positions.csv"   # files created by data_loader.py
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "hikaru_move_features.csv"
TEMP_CSV = OUTPUT_CSV.with_name(OUTPUT_CSV.name + ".tmp")   # written first

REQUIRED_COLUMNS = ["fen", "move_uci"]

# Columns that are ONLY for identification / debugging. They must NOT be used
# as model input. (actual_move_uci reveals the answer, so it is metadata only.)
METADATA_COLUMNS = ["position_id", "game_id", "ply", "fen",
                    "actual_move_uci", "candidate_move_uci"]
TARGET_COLUMN = "label"          # 1 = move Hikaru played, 0 = other legal move
# Every other column in the output file is a model input feature.

MAX_POSITIONS = None   # set e.g. to 200 for a quick test run; None = all rows
CHUNK_POSITIONS = 200  # write to disk after this many positions (saves memory)
PROGRESS_EVERY = 500   # print a progress message every N input rows

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

# Piece types we count for each side (kings are skipped: always exactly 1).
COUNTED_PIECES = [("pawns", chess.PAWN), ("knights", chess.KNIGHT),
                  ("bishops", chess.BISHOP), ("rooks", chess.ROOK),
                  ("queens", chess.QUEEN)]


# ---------------------------------------------------------------------------
# STEP 1: READ AND VALIDATE THE INPUT
# ---------------------------------------------------------------------------

def find_positions_csv_files(folder):
    """
    Find every positions CSV made by data_loader.py inside `folder`.
    Returns a list of paths, newest (most recently modified) first.

    Only names ending in _positions.csv are considered. Our own output file,
    temporary files and hidden/backup files are excluded.
    """
    if not folder.exists():
        return []

    candidates = []
    for path in folder.glob(POSITIONS_PATTERN):
        name = path.name.lower()
        if not path.is_file():
            continue
        if path == OUTPUT_CSV or path == TEMP_CSV:           # our own files
            continue
        if "move_features" in name or name.endswith(".tmp"):  # feature/temp files
            continue
        if name.startswith((".", "~")):                       # hidden/backup files
            continue
        candidates.append(path)

    # Newest modification time first (file name breaks ties)
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return candidates


def load_positions(path):
    """Read the positions CSV and check that the required columns exist."""
    if not path.exists():
        print(f"ERROR: input file not found:\n  {path}")
        print("Run data_loader.py first to create it.")
        sys.exit(1)

    try:
        # dtype=str keeps everything as text; empty cells become "" (not NaN)
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        print("ERROR: the input CSV is empty.")
        sys.exit(1)
    except Exception as error:
        print(f"ERROR: could not read the input CSV: {error}")
        sys.exit(1)

    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        print(f"ERROR: the input CSV is missing required column(s): {missing}")
        print(f"Columns found: {list(data.columns)}")
        sys.exit(1)

    if data.empty:
        print("ERROR: the input CSV has a header but no rows.")
        sys.exit(1)

    if "game_id" not in data.columns:
        print("WARNING: no 'game_id' column found. The output will have an empty "
              "game_id, so train_model.py will NOT be able to split by game.")
    return data


def parse_position(fen, move_uci):
    """
    Turn a FEN + UCI string into a chess.Board and chess.Move.
    Returns (board, move, None) on success, or (None, None, reason) on failure.
    """
    if not fen or not move_uci:
        return None, None, "Missing FEN or move_uci"

    try:
        board = chess.Board(fen)
    except ValueError:
        return None, None, "Invalid FEN"

    if not board.is_valid():
        return None, None, "FEN is not a valid chess position"

    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return None, None, "Invalid UCI move"

    if move not in board.legal_moves:
        return None, None, "Recorded move is not legal in this position"

    return board, move, None


# ---------------------------------------------------------------------------
# STEP 2: FEATURES
# ---------------------------------------------------------------------------

def extract_position_features(board):
    """
    Features describing the POSITION (same for every candidate move in it).
    Computed once per position. Booleans are stored as 0/1.
    """
    mover = board.turn
    opponent = not mover

    features = {
        "side_to_move_is_white": int(mover == chess.WHITE),
        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
        "in_check": int(board.is_check()),
        "num_legal_moves": board.legal_moves.count(),
        "mover_can_castle_kingside": int(board.has_kingside_castling_rights(mover)),
        "mover_can_castle_queenside": int(board.has_queenside_castling_rights(mover)),
    }

    material = {chess.WHITE: 0, chess.BLACK: 0}
    total_pieces = 0
    for color, prefix in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        for name, piece_type in COUNTED_PIECES:
            count = len(board.pieces(piece_type, color))
            features[f"{prefix}_{name}"] = count
            material[color] += count * PIECE_VALUES[piece_type]
            total_pieces += count

    features["total_pieces_excluding_kings"] = total_pieces
    features["white_material"] = material[chess.WHITE]
    features["black_material"] = material[chess.BLACK]
    # Material balance BEFORE the move, from the point of view of the side to move
    features["material_balance_for_mover"] = material[mover] - material[opponent]
    return features


def extract_move_features(board, move):
    """
    Features describing ONE candidate move.

    The original `board` is never changed: anything that needs the position
    AFTER the move is calculated on a separate copy.
    """
    mover = board.turn
    opponent = not mover
    piece = board.piece_at(move.from_square)   # always exists for a legal move

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    file_change = to_file - from_file
    rank_change = to_rank - from_rank
    # Positive = towards the opponent's side of the board, negative = backwards
    forward_progress = rank_change if mover == chess.WHITE else -rank_change

    is_en_passant = board.is_en_passant(move)
    is_capture = board.is_capture(move)      # also True for en passant
    captured_type = 0
    if is_en_passant:
        captured_type = chess.PAWN
    elif is_capture:
        captured_type = board.piece_at(move.to_square).piece_type

    # Position AFTER the move, on a copy (the real board stays untouched)
    after = board.copy(stack=False)
    after.push(move)

    gives_check = board.gives_check(move)
    dest_attacked = after.is_attacked_by(opponent, move.to_square)
    dest_defended = after.is_attacked_by(mover, move.to_square)

    home_rank = 0 if mover == chess.WHITE else 7
    is_minor = piece.piece_type in (chess.KNIGHT, chess.BISHOP)

    return {
        # the moving piece
        "piece_type": piece.piece_type,                  # 1=P 2=N 3=B 4=R 5=Q 6=K
        "piece_is_white": int(piece.color == chess.WHITE),
        "piece_value": PIECE_VALUES[piece.piece_type],
        # squares (0-63) and their file/rank (0-7)
        "from_square": move.from_square,
        "to_square": move.to_square,
        "from_file": from_file,
        "from_rank": from_rank,
        "to_file": to_file,
        "to_rank": to_rank,
        # distance and direction
        "file_change": file_change,
        "rank_change": rank_change,
        "forward_progress": forward_progress,
        "move_distance": max(abs(file_change), abs(rank_change)),
        # special move types
        "is_capture": int(is_capture),
        "captured_piece_type": captured_type,            # 0 = no capture
        "captured_piece_value": PIECE_VALUES.get(captured_type, 0),
        "is_en_passant": int(is_en_passant),
        "is_castling": int(board.is_castling(move)),
        "is_kingside_castling": int(board.is_kingside_castling(move)),
        "is_queenside_castling": int(board.is_queenside_castling(move)),
        "is_promotion": int(move.promotion is not None),
        "promotion_piece_type": move.promotion if move.promotion else 0,
        # consequences of the move
        "gives_check": int(gives_check),
        "gives_checkmate": int(gives_check and after.is_checkmate()),
        "opponent_mobility_after": after.legal_moves.count(),
        # safety of the squares involved
        "from_square_attacked_by_opponent": int(board.is_attacked_by(opponent, move.from_square)),
        "dest_attacked_by_opponent": int(dest_attacked),     # checked after the move
        "dest_defended_by_own": int(dest_defended),          # checked after the move
        "dest_hanging": int(dest_attacked and not dest_defended),
        # simple strategic hints
        "dest_in_center": int(to_file in (3, 4) and to_rank in (3, 4)),
        "dest_in_extended_center": int(2 <= to_file <= 5 and 2 <= to_rank <= 5),
        "is_minor_piece_development": int(is_minor and from_rank == home_rank),
    }


# ---------------------------------------------------------------------------
# STEP 3: POSITIVE AND NEGATIVE EXAMPLES
# ---------------------------------------------------------------------------

def build_position_examples(board, actual_move, metadata):
    """
    Create one row per legal move in this position.
    label = 1 for the move Hikaru played, 0 for all other legal moves.
    """
    position_features = extract_position_features(board)   # computed once
    rows = []
    for move in list(board.legal_moves):
        row = dict(metadata)                                # identifiers
        row["candidate_move_uci"] = move.uci()
        row[TARGET_COLUMN] = int(move == actual_move)
        row.update(position_features)
        row.update(extract_move_features(board, move))
        rows.append(row)
    return rows


def check_position_examples(board, actual_move, rows):
    """
    Sanity check for one position. Returns None if everything is fine,
    otherwise a short text describing the problem.
    """
    positives = [r for r in rows if r[TARGET_COLUMN] == 1]
    if len(positives) != 1:
        return f"expected exactly 1 positive candidate, found {len(positives)}"
    if positives[0]["candidate_move_uci"] != actual_move.uci():
        return "the positive candidate is not the recorded move"
    if chess.Move.from_uci(positives[0]["candidate_move_uci"]) not in board.legal_moves:
        return "the positive move is not legal"
    if len({r["candidate_move_uci"] for r in rows}) != len(rows):
        return "duplicate candidate moves"
    return None


# ---------------------------------------------------------------------------
# STEP 4: SAVING AND VERIFYING
# ---------------------------------------------------------------------------

def write_chunk(rows, columns, path, first_chunk):
    """Write rows to the temporary CSV (first chunk creates the file)."""
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, mode="w" if first_chunk else "a",
                 header=first_chunk, index=False, encoding="utf-8")


def verify_output_file(path, expected_positions, expected_rows):
    """Re-read the finished file and compare row / positive counts."""
    rows = 0
    positives = 0
    for chunk in pd.read_csv(path, usecols=[TARGET_COLUMN], chunksize=200000):
        rows += len(chunk)
        positives += int(chunk[TARGET_COLUMN].sum())
    return rows == expected_rows and positives == expected_positions


def get_optional_column(data, name, default=""):
    """Return a column as a list, or a list of defaults if it does not exist."""
    if name in data.columns:
        return data[name].tolist()
    return [default] * len(data)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    print("=== Feature Extractor ===")

    # Automatically pick the most recently modified *_positions.csv file
    matching_files = find_positions_csv_files(PROCESSED_DIR)
    if not matching_files:
        print(f"ERROR: no positions file ({POSITIONS_PATTERN}) was found in:")
        print(f"  {PROCESSED_DIR}")
        print("Please run data_loader.py first to create it.")
        sys.exit(1)

    input_csv = matching_files[0]
    print(f"Input file selected : {input_csv.name}")
    print(f"Input location      : {input_csv.parent}")
    if len(matching_files) > 1:
        print(f"NOTE: {len(matching_files)} positions files were found; the most "
              "recently modified one is used. The others are:")
        for other in matching_files[1:]:
            print(f"        {other.name}")
    print(f"Output              : {OUTPUT_CSV}")

    data = load_positions(input_csv)
    if MAX_POSITIONS is not None:
        data = data.head(MAX_POSITIONS)
        print(f"NOTE: test mode, only the first {len(data)} rows are used.")
    total_input_rows = len(data)
    print(f"Read {total_input_rows} rows from the positions file.")

    if OUTPUT_CSV.exists():
        print("NOTE: an older features file exists; it is replaced only after "
              "this run finishes successfully.")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fens = data["fen"].tolist()
    moves = data["move_uci"].tolist()
    game_ids = get_optional_column(data, "game_id")
    plies = get_optional_column(data, "ply")

    skipped = Counter()          # reason -> how many rows
    first_example = {}           # reason -> first input row number (for debugging)
    valid_positions = 0
    total_candidate_rows = 0
    buffer = []                  # rows waiting to be written
    positions_in_buffer = 0
    columns = None               # column order, taken from the first valid row
    wrote_any = False

    try:
        for i in range(total_input_rows):
            fen = fens[i].strip()
            move_uci = moves[i].strip()

            board, actual_move, reason = parse_position(fen, move_uci)
            if reason is None:
                ply_text = str(plies[i]).strip()
                metadata = {
                    "position_id": f"pos_{i:06d}",   # row number in the input CSV
                    "game_id": game_ids[i],
                    "ply": int(ply_text) if ply_text.isdigit() else "",
                    "fen": fen,
                    "actual_move_uci": move_uci,
                }
                rows = build_position_examples(board, actual_move, metadata)
                problem = check_position_examples(board, actual_move, rows)
                if problem is not None:
                    reason = "Sanity check failed: " + problem

            if reason is not None:
                skipped[reason] += 1
                first_example.setdefault(reason, i)
            else:
                if columns is None:
                    columns = list(rows[0].keys())
                buffer.extend(rows)
                positions_in_buffer += 1
                valid_positions += 1
                total_candidate_rows += len(rows)

                if positions_in_buffer >= CHUNK_POSITIONS:
                    write_chunk(buffer, columns, TEMP_CSV, first_chunk=not wrote_any)
                    wrote_any = True
                    buffer = []
                    positions_in_buffer = 0

            if (i + 1) % PROGRESS_EVERY == 0:
                print(f"  ... {i + 1}/{total_input_rows} rows examined, "
                      f"{valid_positions} valid positions, "
                      f"{total_candidate_rows} candidate rows")

        if buffer:
            write_chunk(buffer, columns, TEMP_CSV, first_chunk=not wrote_any)
            wrote_any = True

        if not wrote_any:
            print("ERROR: no valid positions were found, so no output was written.")
            print_skip_report(skipped, first_example)
            sys.exit(1)

        print("Checking the finished file ...")
        if not verify_output_file(TEMP_CSV, valid_positions, total_candidate_rows):
            raise RuntimeError("verification failed: row or positive-label counts "
                               "do not match")

        TEMP_CSV.replace(OUTPUT_CSV)     # replace the old file only now

    except SystemExit:
        TEMP_CSV.unlink(missing_ok=True)
        raise
    except PermissionError:
        TEMP_CSV.unlink(missing_ok=True)
        print(f"ERROR: could not write {OUTPUT_CSV.name}. Is it open in Excel? "
              "Close it and run again. Any previous file was kept.")
        sys.exit(1)
    except Exception as error:
        TEMP_CSV.unlink(missing_ok=True)
        print(f"ERROR: processing failed ({error}). Any previous file was kept.")
        sys.exit(1)

    # ----- final summary -----------------------------------------------------
    positives = valid_positions
    negatives = total_candidate_rows - positives
    feature_columns = [c for c in columns
                       if c not in METADATA_COLUMNS and c != TARGET_COLUMN]

    print()
    print("=== Summary ===")
    print(f"Input rows read              : {total_input_rows}")
    print(f"Valid positions processed    : {valid_positions}")
    print(f"Rows skipped                 : {sum(skipped.values())}")
    print_skip_report(skipped, first_example)
    print(f"Candidate rows written       : {total_candidate_rows}")
    print(f"  positive (label 1)         : {positives}")
    print(f"  negative (label 0)         : {negatives}")
    print(f"Average candidates/position  : {total_candidate_rows / valid_positions:.1f}")
    print(f"Sanity check                 : passed (exactly 1 legal positive per position)")
    print(f"Metadata columns             : {METADATA_COLUMNS}")
    print(f"Target column                : {TARGET_COLUMN}")
    print(f"Feature columns              : {len(feature_columns)}")
    print(f"Saved to                     : {OUTPUT_CSV}")
    print(f"Time taken                   : {time.time() - start_time:.1f} seconds")


def print_skip_report(skipped, first_example):
    """Print why rows were skipped (if any were)."""
    for reason, count in skipped.items():
        print(f"  - {count} row(s): {reason} "
              f"(first at input row number {first_example[reason]}, counting from 0)")


if __name__ == "__main__":
    main()