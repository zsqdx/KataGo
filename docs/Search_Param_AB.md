# Search Parameter A/B Notes

This note lists stock KataGo search parameters that are already bot-indexed in
`katago match`, plus one local experimental graph-search knob. In a match config,
write `param0 = ...` for the baseline bot and `param1 = ...` for the candidate
bot. The `python/search_eval.py make-config` helper does this automatically for
`--baseline-param KEY=VALUE` and `--candidate-param KEY=VALUE`.

Example:

```bash
python python/search_eval.py make-config \
  --output runs/fdx6d-trt-root-breadth.cfg \
  --model models/kata1-zhizi-b40c768nbt-fdx6d.bin.gz \
  --games 2000 \
  --game-threads 32 \
  --visits 400 \
  --search-threads 1 \
  --nn-max-batch-size 256 \
  --nn-server-threads 2 \
  --trt --device 0 \
  --sprt --elo0 0 --elo1 5 \
  --candidate-param rootDesiredPerChildVisitsCoeff=1.0
```

To print the maintained candidate list:

```bash
python python/search_eval.py list-params --priority high
```

## High-Potential First Sweeps

- `uncertaintyUtilityFactor`: try `0.03, 0.06, 0.10, 0.15`, usually with
  `staticScoreUtilityFactor = 0.0` and `dynamicScoreUtilityFactor = 0.0` for a
  clean "replace score margin with uncertainty" test. Positive values prefer
  lower short-term uncertainty when ahead and higher uncertainty when behind.
- `uncertaintyUtilityScoreWeight`: try `0.0, 0.5, 1.0, 2.0`. This converts the
  net's short-term score error, divided by `sqrt(board area)`, into the
  uncertainty term.
- `uncertaintyUtilityAdvantageScale`: try `0.15, 0.25, 0.40, 0.60`. Smaller
  values switch more sharply between uncertainty-seeking and uncertainty-avoiding.
- `rootDesiredPerChildVisitsCoeff`: try `0.25, 0.5, 1.0, 2.0`. Root breadth is
  often the cleanest low/medium-visit test.
- `cpuctExploration`, `cpuctExplorationLog`: high leverage, but coupled. Sweep
  one at a time before testing pairs.
- `rootFpuReductionMax`, `fpuReductionMax`: changes first-play urgency. Root
  FPU is usually the less noisy first test.
- `policyOptimism`, `rootPolicyOptimism`: model-dependent policy trust knobs.
- `useNoisePruning`, `noisePruneUtilityScale`, `valueWeightExponent`: value
  aggregation and noisy-visit discounting.
- `useUncertainty`, `uncertaintyCoeff`: uncertainty-weighted backup behavior.
- `subtreeValueBiasFactor`, `subtreeValueBiasWeightExponent`: empirical subtree
  bias correction.
- `futileVisitsThreshold`: can help practical time-based play, but fixed-visits
  A/B should check that Elo does not drop.
- `graphSearchCatchUpLeakProb`: existing stock knob for occasionally deepening
  transposed children.
- `graphSearchCatchUpProp`: local experimental knob. Default `0.0` preserves
  stock behavior; small values let graph-search edge visits catch up in larger
  chunks.

## Medium-Potential Sweeps

- `rootPolicyTemperature`, `nnPolicyTemperature`: policy shape changes; avoid
  mixing with cpuct in first-pass tests.
- `lcbStdevs`, `minVisitPropForLCB`, `useLcbForSelection`: final move selection.
- `useEvalCache`, `evalCacheMinVisits`: only meaningful with graph search; test
  carefully because it changes reuse behavior.
- `graphSearchRepBound`: lower values may increase transpositions but are less
  conservative.
- `numVirtualLossesPerThread`: relevant when `numSearchThreads > 1`.
- `rootEndingBonusPoints`, `fillDameBeforePass`, `enablePassingHacks`,
  `enableMorePassingHacks`: rule- and endgame-dependent.

## Lower-Priority Or Special-Case

- `chosenMoveTemperature`, `chosenMoveTemperatureEarly`,
  `chosenMoveTemperatureHalflife`, `chosenMoveTemperatureOnlyBelowProb`,
  `chosenMoveSubtract`, `chosenMovePrune`: keep deterministic for fixed-visit
  Elo unless explicitly testing move sampling.
- `rootNoiseEnabled`, `rootDirichletNoiseTotalConcentration`,
  `rootDirichletNoiseWeight`, `wideRootNoise`: mostly self-play diversity or
  analysis behavior, not clean strength A/B.
- `rootSymmetryPruning`, `rootNumSymmetriesToSample`: useful for analysis or
  root-only behavior tests, but may change compute meaning.
- `antiMirror`: special-case mirror response.
- Human SL parameters require a human policy model and are not part of a normal
  fdx6d strength sweep.

## Practical Gates

Start each candidate with fixed visits, `numSearchThreads = 1`, deterministic
move choice, and no root noise. If it is positive, repeat at another visit budget
and then run the deployment shape: TensorRT, tuned `nnMaxBatchSize`,
`numNNServerThreadsPerModel`, and the intended time or visit budget.
