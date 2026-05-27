#!/usr/bin/env python3
"""
Utilities for KataGo search Elo experiments.

This script intentionally uses only the Python standard library so it can run on
benchmark machines without the training dependencies installed.
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SGF_GAME_RE = re.compile(
    r"PB\[((?:\\.|[^\]])*)\].*?"
    r"PW\[((?:\\.|[^\]])*)\].*?"
    r"RE\[((?:\\.|[^\]])*)\]",
    re.DOTALL,
)


@dataclass
class PairRecord:
    first: str
    second: str
    win: float = 0.0
    loss: float = 0.0
    draw: float = 0.0

    @property
    def total(self) -> float:
        return self.win + self.loss + self.draw

    @property
    def score(self) -> float:
        return self.win + 0.5 * self.draw


@dataclass
class ParsedGame:
    black: str
    white: str
    result: str
    winner: Optional[str]


@dataclass(frozen=True)
class SearchParamCandidate:
    name: str
    priority: str
    match_default: str
    sweep: str
    note: str


SEARCH_PARAM_CANDIDATES: List[SearchParamCandidate] = [
    SearchParamCandidate(
        "uncertaintyUtilityFactor",
        "high",
        "0.0",
        "0.03,0.06,0.10,0.15",
        "adds uncertainty-seeking when behind and uncertainty-avoidance when ahead; pair with score utility ablation",
    ),
    SearchParamCandidate(
        "uncertaintyUtilityScoreWeight",
        "high",
        "1.0",
        "0.0,0.5,1.0,2.0",
        "converts shortterm score error, divided by sqrt(board area), into the uncertainty utility term",
    ),
    SearchParamCandidate(
        "uncertaintyUtilityAdvantageScale",
        "high",
        "0.25",
        "0.15,0.25,0.40,0.60",
        "smoothness of switching between uncertainty-seeking and uncertainty-avoidance",
    ),
    SearchParamCandidate(
        "rootDesiredPerChildVisitsCoeff",
        "high",
        "0.0",
        "0.25,0.5,1.0,2.0",
        "root breadth; often most useful at low or medium fixed visits",
    ),
    SearchParamCandidate(
        "cpuctExploration",
        "high",
        "1.0",
        "0.85,0.95,1.05,1.15",
        "main exploration scale; sweep with cpuctExplorationLog only after single-param tests",
    ),
    SearchParamCandidate(
        "cpuctExplorationLog",
        "high",
        "0.45",
        "0.25,0.35,0.55,0.65",
        "visit-dependent exploration; can interact strongly with visit budget",
    ),
    SearchParamCandidate(
        "rootFpuReductionMax",
        "high",
        "0.1",
        "0.0,0.05,0.15,0.2",
        "root first-play urgency; high leverage for move ordering",
    ),
    SearchParamCandidate(
        "fpuReductionMax",
        "high",
        "0.2",
        "0.1,0.15,0.25,0.35",
        "tree first-play urgency; usually test after rootFpuReductionMax",
    ),
    SearchParamCandidate(
        "policyOptimism",
        "high",
        "1.0",
        "0.5,0.75,0.9,1.0",
        "optimistic policy transform away from root; can help or overtrust policy",
    ),
    SearchParamCandidate(
        "rootPolicyOptimism",
        "high",
        "0.2",
        "0.0,0.1,0.3,0.5",
        "root optimistic policy transform; likely model- and visits-dependent",
    ),
    SearchParamCandidate(
        "useNoisePruning",
        "high",
        "true",
        "false",
        "ablate value aggregation pruning before tuning noisePruneUtilityScale",
    ),
    SearchParamCandidate(
        "noisePruneUtilityScale",
        "high",
        "0.15",
        "0.10,0.20,0.30",
        "how aggressively noisy excess visits are discounted",
    ),
    SearchParamCandidate(
        "valueWeightExponent",
        "high",
        "0.25",
        "0.0,0.15,0.35,0.5",
        "downweights bad children in value aggregation",
    ),
    SearchParamCandidate(
        "useUncertainty",
        "high",
        "true",
        "false",
        "ablate uncertainty weighting before tuning uncertaintyCoeff",
    ),
    SearchParamCandidate(
        "uncertaintyCoeff",
        "high",
        "0.25",
        "0.15,0.20,0.30,0.40",
        "uncertainty weighting strength",
    ),
    SearchParamCandidate(
        "subtreeValueBiasFactor",
        "high",
        "0.45",
        "0.0,0.25,0.6,0.8",
        "empirical subtree bias correction; important to validate across visits",
    ),
    SearchParamCandidate(
        "subtreeValueBiasWeightExponent",
        "high",
        "0.85",
        "0.5,0.7,1.0",
        "weighting exponent for subtree bias samples",
    ),
    SearchParamCandidate(
        "futileVisitsThreshold",
        "high",
        "0.0",
        "0.02,0.05,0.08",
        "root pruning; useful for speed/time but watch tactical misses",
    ),
    SearchParamCandidate(
        "graphSearchCatchUpLeakProb",
        "high",
        "0.0",
        "0.01,0.03,0.05",
        "occasionally deepens transposed nodes instead of only catching up edge visits",
    ),
    SearchParamCandidate(
        "graphSearchCatchUpProp",
        "high",
        "0.0",
        "0.002,0.005,0.01,0.02",
        "new experimental graph-search edge catch-up rate; requires useGraphSearch=true",
    ),
    SearchParamCandidate(
        "rootPolicyTemperature",
        "medium",
        "1.0",
        "0.9,1.05,1.1,1.2",
        "root policy sharpening/flattening; avoid mixing with cpuct sweeps initially",
    ),
    SearchParamCandidate(
        "nnPolicyTemperature",
        "medium",
        "1.0",
        "0.9,1.05,1.1",
        "global policy temperature inside the tree",
    ),
    SearchParamCandidate(
        "lcbStdevs",
        "medium",
        "5.0",
        "4.0,4.5,5.5,6.0",
        "final move LCB confidence; only matters with useLcbForSelection=true",
    ),
    SearchParamCandidate(
        "minVisitPropForLCB",
        "medium",
        "0.15",
        "0.05,0.10,0.20,0.30",
        "minimum visits for LCB override during final move selection",
    ),
    SearchParamCandidate(
        "useEvalCache",
        "medium",
        "false",
        "true",
        "graph-search eval reuse; pair with evalCacheMinVisits and test correctness carefully",
    ),
    SearchParamCandidate(
        "evalCacheMinVisits",
        "medium",
        "100",
        "20,50,200",
        "minimum visits before storing eval-cache entries",
    ),
    SearchParamCandidate(
        "graphSearchRepBound",
        "medium",
        "11",
        "7,9,13,17",
        "transposition safety bound; lower may reuse more but is riskier",
    ),
    SearchParamCandidate(
        "numVirtualLossesPerThread",
        "medium",
        "1.0",
        "0.5,1.5,2.0,3.0",
        "only relevant when numSearchThreads > 1",
    ),
    SearchParamCandidate(
        "rootEndingBonusPoints",
        "medium",
        "0.5",
        "0.0,0.25,0.75",
        "endgame cleanup/pass behavior; rules dependent",
    ),
    SearchParamCandidate(
        "fillDameBeforePass",
        "medium",
        "false",
        "true",
        "territory-scoring pass behavior; not a universal Elo win",
    ),
    SearchParamCandidate(
        "enablePassingHacks",
        "medium",
        "false",
        "true",
        "GTP/analysis default is true, match default is false; test if pass issues matter",
    ),
    SearchParamCandidate(
        "chosenMoveTemperature",
        "low",
        "0.1",
        "0.0",
        "set to 0 for deterministic fixed-visit Elo; not a search-quality knob",
    ),
    SearchParamCandidate(
        "rootNoiseEnabled",
        "low",
        "false",
        "false",
        "keep off for strength A/B unless testing self-play diversity",
    ),
    SearchParamCandidate(
        "rootSymmetryPruning",
        "low",
        "false",
        "true",
        "can reduce root breadth; mainly analysis/GTP behavior",
    ),
    SearchParamCandidate(
        "antiMirror",
        "low",
        "false",
        "true",
        "special-case mirror countermeasure, not a general Elo sweep",
    ),
]


def list_params(args: argparse.Namespace) -> int:
    priorities = {"high", "medium", "low"} if args.priority == "all" else {args.priority}
    params = [p for p in SEARCH_PARAM_CANDIDATES if p.priority in priorities]
    if args.json:
        payload = [p.__dict__ for p in params]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("Bot-indexed stock search params for match A/B")
    print("Use as KEY0=baseline and KEY1=candidate, or via --candidate-param KEY=VALUE.")
    print("")
    for p in params:
        print(f"[{p.priority}] {p.name}")
        print(f"  match default: {p.match_default}")
        print(f"  suggested:     {p.sweep}")
        print(f"  note:          {p.note}")
    return 0


def unescape_sgf_value(value: str) -> str:
    out: List[str] = []
    escaping = False
    for ch in value:
        if escaping:
            if ch not in "\r\n":
                out.append(ch)
            escaping = False
        elif ch == "\\":
            escaping = True
        else:
            out.append(ch)
    return "".join(out).strip()


def parse_result(result: str, black: str, white: str) -> Optional[str]:
    result = result.strip()
    upper = result.upper()
    if upper.startswith("B+"):
        return black
    if upper.startswith("W+"):
        return white
    if upper in ("0", "DRAW", "JIGO"):
        return "draw"
    if upper in ("VOID", "NORESULT", "NO RESULT"):
        return None
    return None


def iter_game_files(paths: Iterable[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            for child in iterator:
                if child.is_file() and child.suffix.lower() in (".sgf", ".sgfs"):
                    yield child
        elif path.is_file():
            yield path


def parse_sgf_file(path: Path) -> List[ParsedGame]:
    text = path.read_text(encoding="utf-8", errors="replace")
    games: List[ParsedGame] = []
    for match in SGF_GAME_RE.finditer(text):
        black = unescape_sgf_value(match.group(1))
        white = unescape_sgf_value(match.group(2))
        result = unescape_sgf_value(match.group(3))
        winner = parse_result(result, black, white)
        if not black or not white:
            continue
        games.append(ParsedGame(black=black, white=white, result=result, winner=winner))
    return games


def add_record(records: Dict[Tuple[str, str], PairRecord], first: str, second: str, outcome: str) -> None:
    key = (first, second)
    if key not in records:
        records[key] = PairRecord(first=first, second=second)
    record = records[key]
    if outcome == "win":
        record.win += 1.0
    elif outcome == "loss":
        record.loss += 1.0
    elif outcome == "draw":
        record.draw += 1.0
    else:
        raise ValueError(f"unknown outcome: {outcome}")


def load_records(paths: Iterable[Path], recursive: bool) -> Tuple[Dict[Tuple[str, str], PairRecord], int, int]:
    records: Dict[Tuple[str, str], PairRecord] = {}
    files = 0
    games = 0
    skipped = 0
    for path in iter_game_files(paths, recursive):
        files += 1
        for game in parse_sgf_file(path):
            if game.winner is None:
                skipped += 1
                continue
            games += 1
            if game.winner == "draw":
                add_record(records, game.black, game.white, "draw")
            elif game.winner == game.black:
                add_record(records, game.black, game.white, "win")
            elif game.winner == game.white:
                add_record(records, game.black, game.white, "loss")
            else:
                skipped += 1
    return records, files, games


def combined_record(records: Dict[Tuple[str, str], PairRecord], first: str, second: str) -> PairRecord:
    direct = records.get((first, second), PairRecord(first, second))
    reverse = records.get((second, first), PairRecord(second, first))
    return PairRecord(
        first=first,
        second=second,
        win=direct.win + reverse.loss,
        loss=direct.loss + reverse.win,
        draw=direct.draw + reverse.draw,
    )


def all_players(records: Dict[Tuple[str, str], PairRecord]) -> List[str]:
    players = set()
    for first, second in records.keys():
        players.add(first)
        players.add(second)
    return sorted(players)


def elo_from_record(record: PairRecord, prior_games: float) -> Tuple[float, float, float, float]:
    total = record.total
    if total <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    adjusted_total = total + prior_games
    adjusted_score = record.score + 0.5 * prior_games
    p = min(1.0 - 1e-12, max(1e-12, adjusted_score / adjusted_total))
    elo = 400.0 * math.log10(p / (1.0 - p))
    strength_stderr = math.sqrt(1.0 / max(1e-12, adjusted_total * p * (1.0 - p)))
    elo_stderr = 400.0 / math.log(10.0) * strength_stderr
    los = 0.5 * (1.0 + math.erf(elo / max(1e-12, elo_stderr) / math.sqrt(2.0)))
    return elo, elo_stderr, los, p


def sprt_status(record: PairRecord, elo0: float, elo1: float, alpha: float, beta: float) -> Dict[str, object]:
    total = record.total
    if total <= 0:
        return {"llr": 0.0, "lower": None, "upper": None, "status": "no games"}

    score = record.score
    p0 = 1.0 / (1.0 + math.pow(10.0, -elo0 / 400.0))
    p1 = 1.0 / (1.0 + math.pow(10.0, -elo1 / 400.0))
    p0 = min(1.0 - 1e-12, max(1e-12, p0))
    p1 = min(1.0 - 1e-12, max(1e-12, p1))
    llr = score * math.log(p1 / p0) + (total - score) * math.log((1.0 - p1) / (1.0 - p0))
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    if llr >= upper:
        status = f"accept H1: Elo >= {elo1:g}"
    elif llr <= lower:
        status = f"accept H0: Elo <= {elo0:g}"
    else:
        status = "continue"
    return {"llr": llr, "lower": lower, "upper": upper, "status": status}


def format_record_line(record: PairRecord, prior_games: float) -> str:
    elo, elo_stderr, los, _ = elo_from_record(record, prior_games)
    total = record.total
    pct = 100.0 * record.score / total if total > 0 else float("nan")
    return (
        f"{record.first} vs {record.second}: "
        f"{record.score:.1f}/{total:.0f} ({pct:.2f}%), "
        f"Elo {elo:+.1f} +/- {elo_stderr:.1f}, "
        f"LOS {100.0 * los:.2f}% "
        f"[W-L-D {record.win:.0f}-{record.loss:.0f}-{record.draw:.0f}]"
    )


def summarize(args: argparse.Namespace) -> int:
    records, files, games = load_records([Path(p) for p in args.paths], args.recursive)
    players = all_players(records)
    payload = {
        "files": files,
        "games": games,
        "players": players,
        "pairs": {},
    }

    if not args.json:
        print(f"Files: {files}")
        print(f"Scored games: {games}")
        print(f"Players: {', '.join(players) if players else '(none)'}")

    pairs_to_print: List[Tuple[str, str]] = []
    if args.candidate and args.baseline:
        pairs_to_print.append((args.candidate, args.baseline))
    else:
        for i, first in enumerate(players):
            for second in players[i + 1 :]:
                pairs_to_print.append((first, second))

    for first, second in pairs_to_print:
        record = combined_record(records, first, second)
        elo, elo_stderr, los, p = elo_from_record(record, args.prior_games)
        pair_payload = {
            "score": record.score,
            "total": record.total,
            "win": record.win,
            "loss": record.loss,
            "draw": record.draw,
            "score_rate": p,
            "elo": elo,
            "elo_stderr": elo_stderr,
            "likelihood_of_superiority": los,
        }
        if args.sprt:
            pair_payload["sprt"] = sprt_status(record, args.elo0, args.elo1, args.alpha, args.beta)
        payload["pairs"][f"{first}__vs__{second}"] = pair_payload

        if not args.json:
            print(format_record_line(record, args.prior_games))
            if args.sprt:
                sprt = pair_payload["sprt"]
                if sprt["lower"] is None or sprt["upper"] is None:
                    print(f"  SPRT [{args.elo0:g},{args.elo1:g}] {sprt['status']}")
                else:
                    print(
                        f"  SPRT [{args.elo0:g},{args.elo1:g}] "
                        f"LLR {sprt['llr']:.3f}, bounds [{sprt['lower']:.3f}, {sprt['upper']:.3f}], "
                        f"{sprt['status']}"
                    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def split_param(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"empty key in {text!r}")
    return key, value


def config_line(key: str, value: object) -> str:
    return f"{key} = {value}"


def find_config_key_line(lines: List[str], key: str) -> Optional[int]:
    for idx, line in enumerate(lines):
        if "=" not in line:
            continue
        existing_key = line.split("=", 1)[0].strip()
        if existing_key == key:
            return idx
    return None


def make_config(args: argparse.Namespace) -> int:
    baseline_params = [split_param(p) for p in args.baseline_param]
    candidate_params = [split_param(p) for p in args.candidate_param]
    common_overrides = [split_param(p) for p in args.override]

    lines: List[str] = []
    lines.append("# Generated by python/search_eval.py make-config")
    lines.append("# Run with: katago match -config <this-file> -sgf-output-dir <dir> -log-file <log>")
    lines.append("")
    lines.extend(
        [
            config_line("logSearchInfo", "false"),
            config_line("logMoves", "false"),
            config_line("logGamesEvery", args.log_games_every),
            config_line("logToStdout", "true"),
            "",
            config_line("numBots", 2),
            config_line("botName0", args.baseline_name),
            config_line("botName1", args.candidate_name),
            config_line("nnModelFile", args.model),
            "",
            config_line("numGameThreads", args.game_threads),
            config_line("numGamesTotal", args.games),
            config_line("maxMovesPerGame", args.max_moves),
            config_line("matchEloReport", "true"),
            config_line("matchEloReportPeriodGames", args.elo_report_period_games),
            config_line("matchEloBaselineBot", 0),
            config_line("matchEloCandidateBot", 1),
            config_line("matchEloPriorGames", args.prior_games),
            config_line("matchEloSprtEnabled", str(args.sprt).lower()),
            config_line("matchEloSprtElo0", args.elo0),
            config_line("matchEloSprtElo1", args.elo1),
            config_line("matchEloSprtAlpha", args.alpha),
            config_line("matchEloSprtBeta", args.beta),
            config_line("allowResignation", str(args.allow_resignation).lower()),
            config_line("resignThreshold", args.resign_threshold),
            config_line("resignConsecTurns", args.resign_consec_turns),
            "",
            config_line("koRules", args.ko_rules),
            config_line("scoringRules", args.scoring_rules),
            config_line("taxRules", args.tax_rules),
            config_line("multiStoneSuicideLegals", args.multi_stone_suicide_legals),
            config_line("hasButtons", args.has_buttons),
            config_line("bSizes", args.board_sizes),
            config_line("bSizeRelProbs", args.board_size_rel_probs),
            config_line("komiAuto", "false"),
            config_line("komiMean", args.komi),
            config_line("handicapProb", "0.0"),
            "",
            config_line("maxVisits", args.visits),
            config_line("numSearchThreads", args.search_threads),
            config_line("chosenMoveTemperatureEarly", "0.0"),
            config_line("chosenMoveTemperature", "0.0"),
            config_line("chosenMovePrune", "1.0"),
            config_line("rootNoiseEnabled", "false"),
            "",
            config_line("nnMaxBatchSize", args.nn_max_batch_size),
            config_line("nnCacheSizePowerOfTwo", args.nn_cache_size_power_of_two),
            config_line("nnMutexPoolSizePowerOfTwo", args.nn_mutex_pool_size_power_of_two),
            config_line("nnRandomize", "true"),
            config_line("numNNServerThreadsPerModel", args.nn_server_threads),
        ]
    )
    if args.trt:
        lines.extend(
            [
                config_line("trtUseFP16", "true"),
                config_line("trtDeviceToUse", args.device),
            ]
        )
    lines.append("")

    appended_common_overrides = False
    for key, value in common_overrides:
        existing_idx = find_config_key_line(lines, key)
        if existing_idx is not None:
            lines[existing_idx] = config_line(key, value)
        else:
            if not appended_common_overrides:
                lines.append("# Common override parameters")
                appended_common_overrides = True
            lines.append(config_line(key, value))
    if appended_common_overrides:
        lines.append("")

    lines.append("# Baseline-specific parameters")
    for key, value in baseline_params:
        lines.append(config_line(f"{key}0", value))
    lines.append("")
    lines.append("# Candidate-specific parameters")
    for key, value in candidate_params:
        lines.append(config_line(f"{key}1", value))
    lines.append("")

    text = "\n".join(lines)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KataGo search Elo evaluation helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="summarize match SGFs and estimate Elo")
    summarize_parser.add_argument("paths", nargs="+", help="SGF/SGFS files or directories")
    summarize_parser.add_argument("--recursive", action="store_true", help="search directories recursively")
    summarize_parser.add_argument("--baseline", help="baseline bot name")
    summarize_parser.add_argument("--candidate", help="candidate bot name")
    summarize_parser.add_argument("--prior-games", type=float, default=1.0, help="symmetric prior games for Elo estimate")
    summarize_parser.add_argument("--json", action="store_true", help="write JSON output")
    summarize_parser.add_argument("--sprt", action="store_true", help="print an approximate SPRT status")
    summarize_parser.add_argument("--elo0", type=float, default=0.0, help="SPRT lower Elo hypothesis")
    summarize_parser.add_argument("--elo1", type=float, default=5.0, help="SPRT upper Elo hypothesis")
    summarize_parser.add_argument("--alpha", type=float, default=0.05, help="SPRT alpha")
    summarize_parser.add_argument("--beta", type=float, default=0.05, help="SPRT beta")
    summarize_parser.set_defaults(func=summarize)

    params_parser = subparsers.add_parser("list-params", help="list bot-indexed search params worth A/B testing")
    params_parser.add_argument("--priority", choices=["all", "high", "medium", "low"], default="all")
    params_parser.add_argument("--json", action="store_true", help="write JSON output")
    params_parser.set_defaults(func=list_params)

    config_parser = subparsers.add_parser("make-config", help="generate a two-bot match config")
    config_parser.add_argument("--output", "-o", default="-", help="output config path, or '-' for stdout")
    config_parser.add_argument("--model", required=True, help="model path")
    config_parser.add_argument("--baseline-name", default="baseline")
    config_parser.add_argument("--candidate-name", default="candidate")
    config_parser.add_argument("--baseline-param", action="append", default=[], help="baseline-only KEY=VALUE")
    config_parser.add_argument("--candidate-param", action="append", default=[], help="candidate-only KEY=VALUE")
    config_parser.add_argument("--override", action="append", default=[], help="common KEY=VALUE")
    config_parser.add_argument("--games", type=int, default=400)
    config_parser.add_argument("--game-threads", type=int, default=16)
    config_parser.add_argument("--visits", type=int, default=400)
    config_parser.add_argument("--search-threads", type=int, default=1)
    config_parser.add_argument("--max-moves", type=int, default=1200)
    config_parser.add_argument("--allow-resignation", action="store_true")
    config_parser.add_argument("--resign-threshold", default="-0.95")
    config_parser.add_argument("--resign-consec-turns", default="6")
    config_parser.add_argument("--board-sizes", default="19")
    config_parser.add_argument("--board-size-rel-probs", default="1")
    config_parser.add_argument("--komi", default="7.5")
    config_parser.add_argument("--ko-rules", default="POSITIONAL")
    config_parser.add_argument("--scoring-rules", default="AREA")
    config_parser.add_argument("--tax-rules", default="NONE")
    config_parser.add_argument("--multi-stone-suicide-legals", default="false")
    config_parser.add_argument("--has-buttons", default="false")
    config_parser.add_argument("--nn-max-batch-size", type=int, default=64)
    config_parser.add_argument("--nn-cache-size-power-of-two", type=int, default=21)
    config_parser.add_argument("--nn-mutex-pool-size-power-of-two", type=int, default=17)
    config_parser.add_argument("--nn-server-threads", type=int, default=1)
    config_parser.add_argument("--log-games-every", type=int, default=50)
    config_parser.add_argument("--elo-report-period-games", type=int, default=200)
    config_parser.add_argument("--prior-games", type=float, default=1.0)
    config_parser.add_argument("--sprt", action="store_true")
    config_parser.add_argument("--elo0", type=float, default=0.0)
    config_parser.add_argument("--elo1", type=float, default=5.0)
    config_parser.add_argument("--alpha", type=float, default=0.05)
    config_parser.add_argument("--beta", type=float, default=0.05)
    config_parser.add_argument("--trt", action="store_true", help="include TensorRT FP16/device lines")
    config_parser.add_argument("--device", default="0", help="GPU device for --trt")
    config_parser.set_defaults(func=make_config)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
