from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_ROOT = BASE_DIR / "models" / "player_workspaces"
MIN_GAMES = 100
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

tasks = {}
tasks_lock = threading.Lock()
training_lock = threading.Lock()
training_active = False


def find_stockfish():
    """Find Stockfish on PATH or in this project's ./stockfish folder."""
    on_path = shutil.which("stockfish")
    if on_path:
        return on_path
    stockfish_dir = BASE_DIR / "stockfish"
    if stockfish_dir.is_dir():
        for candidate in stockfish_dir.rglob("stockfish*.exe"):
            if candidate.is_file():
                return str(candidate)
        for candidate in stockfish_dir.rglob("stockfish"):
            if candidate.is_file():
                return str(candidate)
    return None


ENGINE_MODES = {
    "fast": {"label": "Fast", "depth": 16},
    "balanced": {"label": "Balanced", "depth": 22},
    "deep": {"label": "Deep", "depth": 28},
}


def _format_engine_score(score, turn):
    """Format a score from the position's side-to-move perspective."""
    pov_score = score.pov(turn)
    mate = pov_score.mate()
    if mate is not None:
        label = f"Mate in {mate}" if mate >= 0 else f"Mated in {abs(mate)}"
        return label, "mate", None
    cp = pov_score.score()
    return (f"{cp / 100:+.2f}" if cp is not None else "Unavailable"), "centipawns", cp


def analyze_with_stockfish(board, chess, mode="balanced", candidate_moves=None):
    """Analyse the position and evaluate each supplied candidate with bounded MultiPV."""
    mode = mode if mode in ENGINE_MODES else "balanced"
    settings = ENGINE_MODES[mode]
    depth = settings["depth"]
    engine_path = find_stockfish()
    if not engine_path:
        return {
            "available": False, "mode": mode, "target_depth": depth,
            "candidate_evaluations": [],
            "message": "Stockfish was not found. Put its executable in the project's stockfish folder or add it to PATH.",
        }

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=30)
        try:
            engine.configure({"Threads": 2, "Hash": 128})
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            pass

        # First search is the engine's unrestricted best move at the selected depth.
        info = engine.analyse(board, chess.engine.Limit(depth=depth), game=object())
        best_move = info.get("pv", [None])[0]
        result = {
            "available": True, "engine_name": "Stockfish", "mode": mode,
            "mode_label": settings["label"], "target_depth": depth,
            "depth": info.get("depth"), "candidate_evaluations": [],
            "message": f"Stockfish {settings['label'].lower()} analysis completed.",
        }
        if best_move is not None:
            result["best_move_uci"] = best_move.uci()
            result["best_move_san"] = board.san(best_move)
        score = info.get("score")
        if score is not None:
            label, score_type, cp = _format_engine_score(score, board.turn)
            result.update({"evaluation": label, "evaluation_type": score_type,
                           "evaluation_cp": cp, "evaluation_perspective": "side to move"})

        # MultiPV constrained to the ML's candidates compares exactly those moves.
        # Use a slightly shallower target for this second search to keep UI latency bounded.
        candidates = list(candidate_moves or [])[:3]
        if candidates:
            candidate_depth = max(12, depth - 3)
            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=candidate_depth),
                multipv=len(candidates),
                root_moves=candidates,
                game=object(),
            )
            if isinstance(infos, dict):
                infos = [infos]
            by_uci = {}
            for candidate_info in infos:
                pv = candidate_info.get("pv", [])
                move = pv[0] if pv else None
                score = candidate_info.get("score")
                if move is None or score is None:
                    continue
                label, score_type, cp = _format_engine_score(score, board.turn)
                by_uci[move.uci()] = {
                    "uci": move.uci(), "san": board.san(move),
                    "evaluation": label, "evaluation_type": score_type,
                    "evaluation_cp": cp, "depth": candidate_info.get("depth", candidate_depth),
                }
            result["candidate_depth_target"] = candidate_depth
            result["candidate_evaluations"] = [
                by_uci[m.uci()] for m in candidates if m.uci() in by_uci
            ]
            result["message"] = f"Stockfish {settings['label'].lower()} analysis completed; candidate moves evaluated separately."
        return result
    except Exception as exc:
        app.logger.warning("Stockfish analysis failed: %s", exc)
        return {
            "available": False, "mode": mode, "target_depth": depth,
            "candidate_evaluations": [],
            "message": f"Stockfish was found but could not finish this analysis: {type(exc).__name__}: {exc}",
        }
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def safe_slug(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()
    return value[:50] or "player_model"


def update_task(task_id, **values):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(values)


def read_model_record(workspace):
    record_path = workspace / "model_record.json"
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        info_path = workspace / "models" / "hikaru_move_model_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            results = info.get("test_results", {})
            ranking = results.get("test_ranking", {})
            record["metrics"] = {
                "top1": ranking.get("top1"),
                "top3": ranking.get("top3"),
                "mrr": ranking.get("mrr"),
                "training_positions": info.get("training_positions", 0),
                "test_positions": info.get("test_positions", 0),
                "candidate_rows": info.get("training_rows", 0) + info.get("test_rows", 0),
                "training_games": info.get("training_games", 0),
                "test_games": info.get("test_games", 0),
                "number_of_features": info.get("number_of_features", 0),
            }
        return record
    except (OSError, ValueError, TypeError):
        return None


def list_models():
    result = []
    if not MODEL_ROOT.exists():
        return result
    for workspace in MODEL_ROOT.iterdir():
        if not workspace.is_dir():
            continue
        record = read_model_record(workspace)
        if record and (workspace / "models" / "hikaru_move_model.joblib").exists():
            result.append(record)
    result.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return result


def run_script(task_id, workspace, script_name, stage, start_percent, end_percent):
    command = [sys.executable, "-u", str(workspace / script_name)]
    update_task(task_id, stage=stage, percent=start_percent, message=stage)
    process = subprocess.Popen(
        command, cwd=str(workspace), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1
    )
    total_rows = None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        with tasks_lock:
            if task_id in tasks:
                tasks[task_id].setdefault("logs", []).append(line)
                tasks[task_id]["logs"] = tasks[task_id]["logs"][-120:]
        if script_name == "feature_extractor.py":
            match = re.search(r"Read\s+([\d,]+)\s+rows", line)
            if match:
                total_rows = int(match.group(1).replace(",", ""))
            match = re.search(r"\.\.\.\s*([\d,]+)/([\d,]+)\s+rows examined", line)
            if match:
                done = int(match.group(1).replace(",", ""))
                total = int(match.group(2).replace(",", "")) or 1
                pct = start_percent + (end_percent - start_percent) * done / total
                update_task(task_id, percent=round(min(end_percent, pct), 1),
                            message=f"Generating candidate-move features: {done:,}/{total:,} positions")
        elif script_name == "train_model.py":
            # scikit-learn does not expose reliable fit progress for this Random Forest.
            if "Training Random Forest" in line or "Training model" in line:
                update_task(task_id, message="Training Random Forest model…")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{script_name} exited with code {return_code}. Check the latest log lines.")
    update_task(task_id, percent=end_percent, message=f"{stage} complete")


def train_worker(task_id, uploaded_path, player_name, model_name, workspace):
    global training_active
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (workspace / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (workspace / "models").mkdir(parents=True, exist_ok=True)

        # Each saved model has its own copy of all scripts and all intermediate data.
        source_scripts = BASE_DIR / "project_files"
        for script in ("data_loader.py", "feature_extractor.py", "train_model.py"):
            shutil.copy2(source_scripts / script, workspace / script)

        update_task(task_id, stage="Reading PGN", percent=2, message="Reading and validating PGN games")
        spec = importlib.util.spec_from_file_location(
            f"data_loader_{task_id.replace('-', '')}", workspace / "data_loader.py"
        )
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)

        positions_csv = loader.get_csv_path(player_name)
        temp_csv = positions_csv.with_name(positions_csv.stem + ".temporary.csv")

        def loader_progress(stats, percent):
            pct = 2 + 0.23 * float(percent)
            update_task(
                task_id, percent=round(pct, 1),
                message=(f"Reading PGN: {stats.get('games_found', 0):,} games scanned; "
                         f"{stats.get('player_games', 0):,} target-player games accepted"),
                stats=stats
            )

        result = loader.process_pgn_file(
            uploaded_path, player_name, MIN_GAMES, loader_progress, temp_csv
        )
        if result.get("problems"):
            temp_csv.unlink(missing_ok=True)
            raise RuntimeError("PGN validation failed: " + " ".join(result["problems"]))

        loader.save_outputs(uploaded_path, positions_csv, temp_csv)
        update_task(task_id, stage="Extracting positions", percent=25,
                    message=f"Saved {result['stats']['player_games']:,} games for {player_name}")

        run_script(task_id, workspace, "feature_extractor.py",
                   "Generating move features", 27, 68)

        update_task(task_id, stage="Training model", percent=70,
                    message="Training and evaluating the Random Forest model")
        run_script(task_id, workspace, "train_model.py", "Training model", 70, 96)

        info_path = workspace / "models" / "hikaru_move_model_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["target_player"] = player_name
        info["display_model_name"] = model_name
        info["note"] = f"Ranks candidate moves by resemblance to {player_name}'s historical choices; it does not identify the objectively best move."
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        record = {
            "id": workspace.name,
            "model_name": model_name,
            "target_player": player_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "workspace": str(workspace.relative_to(BASE_DIR)),
            "model_file": str((workspace / "models" / "hikaru_move_model.joblib").relative_to(BASE_DIR)),
            "status": "ready"
        }
        (workspace / "model_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        record = read_model_record(workspace)
        update_task(task_id, status="completed", stage="Complete", percent=100,
                    message="Model trained and saved successfully.", model=record,
                    metrics=record.get("metrics", {}))
    except Exception as error:
        update_task(task_id, status="failed", stage="Failed", message=str(error),
                    error=str(error), traceback=traceback.format_exc())
    finally:
        with training_lock:
            training_active = False


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/analysis")
def analysis_page():
    return send_from_directory(FRONTEND_DIR, "analysis.html")


@app.route("/<path:filename>")
def frontend_file(filename):
    if filename in ("style.css", "script.js", "analysis.html"):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": "Not found"}), 404


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"status": "running", "message": "ChessLab backend is connected"})


@app.route("/api/models", methods=["GET"])
def models():
    return jsonify({"models": list_models()})


@app.route("/api/train", methods=["POST"])
def train():
    global training_active
    if "pgn" not in request.files:
        return jsonify({"error": "Upload a PGN file using the pgn field."}), 400
    uploaded = request.files["pgn"]
    player_name = request.form.get("player_name", "").strip()
    model_name = request.form.get("model_name", "").strip()
    if not uploaded.filename or not uploaded.filename.lower().endswith(".pgn"):
        return jsonify({"error": "Please upload a .pgn file."}), 400
    if not player_name:
        return jsonify({"error": "Enter the target player's name."}), 400
    if not model_name:
        model_name = f"{player_name} model"

    with training_lock:
        if training_active:
            return jsonify({"error": "A model is already training. Wait for it to finish before starting another."}), 409
        training_active = True

    task_id = str(uuid.uuid4())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{safe_slug(player_name)}_{safe_slug(model_name)}_{stamp}_{task_id[:8]}"
    workspace = MODEL_ROOT / folder_name
    workspace.mkdir(parents=True, exist_ok=True)
    upload_path = workspace / "uploaded_games.pgn"
    uploaded.save(upload_path)

    with tasks_lock:
        tasks[task_id] = {
            "id": task_id, "status": "running", "stage": "Queued", "percent": 1,
            "message": "Training task started.", "logs": [], "model": None, "metrics": {}
        }

    thread = threading.Thread(
        target=train_worker,
        args=(task_id, upload_path, player_name, model_name, workspace),
        daemon=True
    )
    thread.start()
    return jsonify({"task_id": task_id, "message": "Training started"}), 202


@app.route("/api/progress/<task_id>", methods=["GET"])
def progress(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            return jsonify({"error": "Training task not found"}), 404
        # Return a snapshot so the caller cannot mutate our task state.
        return jsonify(dict(task))



@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        import chess
        import chess.engine
        import joblib
        import pandas as pd
    except ImportError as exc:
        return jsonify({"error": f"Missing dependency: {exc}. Install python-chess, pandas, and joblib."}), 500

    payload = request.get_json(silent=True) or {}
    model_id = str(payload.get("model_id", "")).strip()
    fen = str(payload.get("fen", "")).strip()
    engine_mode = str(payload.get("engine_mode", "balanced")).strip().lower()
    if engine_mode not in ENGINE_MODES:
        return jsonify({"error": "Choose a valid Stockfish mode: fast, balanced, or deep."}), 400
    if not model_id or not fen:
        return jsonify({"error": "Choose a saved model and enter a FEN position."}), 400

    # Restrict model selection to known workspace folders; never accept arbitrary paths.
    workspace = (MODEL_ROOT / model_id).resolve()
    if workspace.parent != MODEL_ROOT.resolve() or not workspace.is_dir():
        return jsonify({"error": "The selected model workspace was not found."}), 404
    model_path = workspace / "models" / "hikaru_move_model.joblib"
    info_path = workspace / "models" / "hikaru_move_model_info.json"
    feature_path = workspace / "feature_extractor.py"
    if not model_path.exists() or not info_path.exists() or not feature_path.exists():
        return jsonify({"error": "This saved model is missing files required for predictions. Rebuild it with the latest ChessLab backend."}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "That FEN could not be read. Check its format and try again."}), 400
    if not board.is_valid():
        return jsonify({"error": "That FEN describes an invalid chess position."}), 400
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return jsonify({"error": "This position has no legal moves (the game may be over)."}), 400

    try:
        spec = importlib.util.spec_from_file_location("chesslab_inference_features", feature_path)
        feature_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_module)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        feature_columns = info["feature_columns"]
        model = joblib.load(model_path)
        position_features = feature_module.extract_position_features(board)
        rows = []
        for move in legal_moves:
            row = dict(position_features)
            row.update(feature_module.extract_move_features(board, move))
            rows.append(row)
        frame = pd.DataFrame(rows)
        for column in feature_columns:
            if column not in frame.columns:
                frame[column] = 0
        scores = model.predict_proba(frame[feature_columns])[:, 1]
        ranked = sorted(zip(legal_moves, scores), key=lambda item: float(item[1]), reverse=True)[:3]
        predictions = [{
            "uci": move.uci(),
            "san": board.san(move),
            "score": round(float(score), 6),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
        } for move, score in ranked]
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {type(exc).__name__}: {exc}"}), 500

    candidate_moves = [board.parse_uci(item["uci"]) for item in predictions]
    engine_result = analyze_with_stockfish(board, chess, engine_mode, candidate_moves)
    engine_result["candidate_evaluations_by_uci"] = {item["uci"]: item for item in engine_result.get("candidate_evaluations", [])}
    return jsonify({
        "fen": board.fen(),
        "side_to_move": "White" if board.turn == chess.WHITE else "Black",
        "legal_move_count": len(legal_moves),
        "predictions": predictions,
        "score_note": "Scores are Random Forest ranking scores, not calibrated probabilities or chess evaluations.",
        "engine": engine_result
    })


def board_payload(board, chess):
    """Serialize a legal chess position for the interactive browser board."""
    pieces = {}
    for square, piece in board.piece_map().items():
        pieces[chess.square_name(square)] = {
            "symbol": piece.symbol(),
            "color": "white" if piece.color == chess.WHITE else "black",
            "piece": piece.symbol().upper(),
        }
    moves = []
    for move in board.legal_moves:
        moves.append({
            "uci": move.uci(),
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "san": board.san(move),
            "promotion": chess.piece_name(move.promotion) if move.promotion else None,
        })
    return {
        "fen": board.fen(),
        "pieces": pieces,
        "turn": "White" if board.turn == chess.WHITE else "Black",
        "legal_moves": moves,
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
    }


@app.route("/api/board-state", methods=["POST"])
def board_state():
    try:
        import chess
    except ImportError:
        return jsonify({"error": "Missing dependency: install python-chess."}), 500
    payload = request.get_json(silent=True) or {}
    fen = str(payload.get("fen", "")).strip()
    if not fen:
        return jsonify({"error": "Enter a FEN position."}), 400
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "That FEN could not be read. Check its format."}), 400
    if not board.is_valid():
        return jsonify({"error": "That FEN describes an invalid chess position."}), 400
    return jsonify(board_payload(board, chess))


@app.route("/api/board-move", methods=["POST"])
def board_move():
    try:
        import chess
    except ImportError:
        return jsonify({"error": "Missing dependency: install python-chess."}), 500
    payload = request.get_json(silent=True) or {}
    fen = str(payload.get("fen", "")).strip()
    from_square = str(payload.get("from", "")).strip()
    to_square = str(payload.get("to", "")).strip()
    promotion = str(payload.get("promotion", "q")).strip().lower()
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "That FEN could not be read."}), 400
    if not board.is_valid():
        return jsonify({"error": "That FEN describes an invalid chess position."}), 400
    try:
        origin = chess.parse_square(from_square)
        target = chess.parse_square(to_square)
    except ValueError:
        return jsonify({"error": "Invalid source or destination square."}), 400
    matches = [m for m in board.legal_moves if m.from_square == origin and m.to_square == target]
    if not matches:
        return jsonify({"error": "That move is not legal in this position."}), 400
    move = next((m for m in matches if m.promotion == {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}.get(promotion)), matches[0])
    board.push(move)
    return jsonify(board_payload(board, chess))

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "The PGN file is too large. Maximum upload size is 500 MB."}), 413


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=5000, threaded=True)
