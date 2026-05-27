# Search Elo Evaluation

This note describes a lightweight A/B loop for search changes. It is aimed at
search parameter and search logic experiments where the same neural net plays
against itself with one controlled difference.

## Why This Flow

`katago match` is the fastest local way to compare two search configurations
because many games share neural net batching. For Elo-sensitive search changes,
prefer fixed visits and `numSearchThreads = 1` first. This avoids timing noise,
thread scheduling effects, and hardware speed differences from changing the
meaning of the result.

After a candidate survives fixed-visit tests, run a second stage under the
deployment shape that matters: real backend, tuned `nnMaxBatchSize`,
`numNNServerThreadsPerModel`, intended search threads, and optionally a
time-based limit.

## Generate A Config

`katago match` has built-in Elo reporting when `matchEloReport = true` is set.
Use `python/search_eval.py make-config` only as a convenience for producing a
two-bot config; the actual reporting happens inside the C++ match runner.

Example:

```bash
python python/search_eval.py make-config \
  --output /tmp/kg-search-ab.cfg \
  --model /path/to/model.bin.gz \
  --baseline-name baseline \
  --candidate-name candidate \
  --games 800 \
  --game-threads 32 \
  --visits 400 \
  --candidate-param rootDesiredPerChildVisitsCoeff=1.0
```

Add or override these lines if the generator did not already include them:

```cfg
matchEloReport = true
matchEloReportPeriodGames = 200
matchEloBaselineBot = 0
matchEloCandidateBot = 1
matchEloSprtEnabled = true
matchEloSprtElo0 = 0
matchEloSprtElo1 = 5
matchEloSprtAlpha = 0.05
matchEloSprtBeta = 0.05
```

Run the match:

```bash
./katago match \
  -config /tmp/kg-search-ab.cfg \
  -log-file /tmp/kg-search-ab.log \
  -sgf-output-dir /tmp/kg-search-ab-sgfs
```

The match log will contain standings, pair Elo, approximate standard error,
likelihood of superiority, SPRT state, and average move time by bot. SGFs are
still useful as the durable artifact. To summarize them later:

```bash
python python/search_eval.py summarize /tmp/kg-search-ab-sgfs \
  --baseline baseline \
  --candidate candidate \
  --sprt --elo0 0 --elo1 5
```

The summarizer reports score, Elo difference, an approximate standard error,
likelihood of superiority, and optional SPRT state. Draws count as half a win.
No-result games are skipped.

To list stock bot-indexed search params worth testing:

```bash
python python/search_eval.py list-params --priority high
```

## Recommended Gates

Use three stages before treating a search change as promising:

1. Smoke: 200-400 games at low visits, fixed 19x19 rules, no resignation.
2. Confirmation: 2k+ games, same setup, mirrored colors from `match`.
3. Deployment: time or production visit budget with the intended backend,
   model, search thread count, and batching configuration.

For PR-quality confidence, a small search change should either clear an SPRT
gate such as `[0, 5]` Elo or show a stable positive result across different
visit budgets. A single +2 Elo run with a 10+ Elo error bar is just a hint, not
evidence.

## Candidate Search Experiments

These are the current code-level knobs worth testing before deeper rewrites:

- `rootDesiredPerChildVisitsCoeff`: forces some policy-proportional root
  breadth. Try `0.25, 0.5, 1.0, 2.0`; likely most relevant at low visits.
- `futileVisitsThreshold`: prunes root moves that can no longer catch up. Try
  `0.02, 0.05, 0.08`; watch for tactical misses at low visits.
- `graphSearchCatchUpLeakProb`: occasionally deepens a transposed child instead
  of only catching up edge visits. Try `0.01, 0.03, 0.05` with graph search on.
- `useEvalCache` plus `evalCacheMinVisits`: may improve reuse around
  transpositions, but it is correctness-sensitive and should be tested with and
  without graph search.
- `cpuctExploration`, `cpuctExplorationLog`, `rootFpuReductionMax`,
  `policyOptimism`, and `rootPolicyOptimism`: these are high-leverage but
  coupled. Sweep one at a time first, then test combinations.

Avoid mixing search Elo experiments with backend speed changes. If a candidate
both changes search behavior and changes throughput, run fixed visits first,
then time-based games second.
