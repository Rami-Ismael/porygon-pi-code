# Measured Ling cost, October 8 pilot

The 63-call instant-mode run cost **$0.0073227**, or **0.73227 cents**. All calls, including invalid completions, are included. The six-call setup smoke cost $0.0023551696; recorded combined API cost is **$0.0096778696**. This is provider-reported usage cost, not an account invoice reconciliation. Local battle compute is excluded.

The earlier prompt estimate of 14,000 tokens was too low: actual input averaged 19,027.56 tokens. That did not make the bill large because 1,098,944 of 1,198,736 input tokens were cached (91.675%), and output averaged only 157.57 tokens (9,927 total). Instant mode reported zero reasoning tokens. The setup smoke averaged 2,964.67 output tokens, including reasoning, and is a separate experiment.

The usage records exactly fit these **historical effective rates**, inferred from returned `cost_details` and token counts: uncached input $0.021/million, cached input $0.0042/million, output $0.0616/million. These are not a claim about current advertised prices. The maximum per-request reconstruction error is below $0.000000000001.

`cost = ((input − cached) × 0.021 + cached × 0.0042 + output × 0.0616) / 1,000,000`

Input cost was $0.0067111968; output cost was $0.0006115032. Cache savings versus entirely uncached inputs of the same length were $0.0184622592. Use token-based scenarios, not a dollar estimate based only on number of teams.

| Calls | Observed cache mix and lengths | Cold cache, same lengths | Cold cache, 4,096 output tokens/call |
|---:|---:|---:|---:|
| 63 | $0.00732 | $0.02578 | $0.04107 |
| 1,000 | $0.11623 | $0.40929 | $0.65189 |
| 8,100 (six-task full grid) | $0.94149 | $3.31521 | $5.28033 |
| 14,100 (including mixed) | $1.63889 | $5.77092 | $9.19168 |

All projections assume the historical effective rates and mean input length. The maximum-output column is a scenario, not a global cost ceiling: prompt sizes and rates can change. Pilot task weights differ from the full grid; a larger run should budget by task and remeasure output length. Retries, other providers, enabled reasoning and cache eviction require recalculation. No larger run was launched.

Source: `code/results/llm-baseline-instant-pilot/completions/*.json` and separate `llm-baseline-pilot/completions/*.json`. The dashboard builder recomputes usage and validates the inferred formula from these records.
