# Fix Comparison

**Headline verdict change:** NO STABLE EDGE -> NO STABLE EDGE

## 6/1 Primary Schedule
| run | fill_model | chosen_parameter_set | oos_pnl_cents | sharpe | win_month_rate | top_3_removed_pnl_cents | randomized_entry_p | distinct_winners_across_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Before fixes | Model A - mid + dynamic cost | a=300s \| b=0.3x sigma \| c=1800s \| d=14400s \| reversal | -291.338 | -0.914615 | 0.222222 | -578.411 | 0.31968 | 7 |
| Before fixes | Model B - actual bid/ask | a=120s \| b=0.5x sigma \| c=900s \| d=7200s \| reversal | -234.923 | -0.734536 | 0.333333 | -526.923 | 0.366633 | 6 |
| After fixes | Model A - mid + dynamic cost | a=120s \| b=0.9x sigma \| c=900s \| d=14400s \| continuation | 608.419 | 3.30818 | 0.444444 | 284.25 | 0.00699301 | 5 |
| After fixes | Model B - actual bid/ask | a=120s \| b=0.9x sigma \| c=900s \| d=14400s \| continuation | 753.483 | 3.74752 | 0.555556 | 431.481 | 0.014985 | 5 |

## 3/1 Schedule
| run | fill_model | chosen_parameter_set | oos_pnl_cents | sharpe | win_month_rate | top_3_removed_pnl_cents | randomized_entry_p | distinct_winners_across_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Before fixes | Model A - mid + dynamic cost | a=300s \| b=0.3x sigma \| c=1800s \| d=14400s \| reversal | -1610.8 | -2.59198 | 0.166667 | -2018.66 | 0.871129 | 10 |
| Before fixes | Model B - actual bid/ask | a=120s \| b=0.5x sigma \| c=900s \| d=7200s \| reversal | -1377.92 | -2.22675 | 0.166667 | -1790.92 | 0.785215 | 10 |
| After fixes | Model A - mid + dynamic cost | a=120s \| b=0.9x sigma \| c=900s \| d=14400s \| continuation | -140.992 | -0.386475 | 0.333333 | -531.129 | 0.836164 | 10 |
| After fixes | Model B - actual bid/ask | a=120s \| b=0.9x sigma \| c=900s \| d=14400s \| continuation | 95.363 | 0.257877 | 0.416667 | -298.637 | 0.759241 | 10 |

## Trade Count Diff
| before_total_oos_trades | after_total_oos_trades | oos_trade_count_change | before_oos_trades_lt5_bdays_to_expiry | postfix_near_expiry_candidate_trades_skipped | postfix_forward_filled_entry_candidate_trades_skipped |
| --- | --- | --- | --- | --- | --- |
| 712 | 360 | -352 | 106 | 1794 | 27888 |

## Per-Fold Winner Stability
| schedule | fill_model | before_distinct_winners | after_distinct_winners |
| --- | --- | --- | --- |
| 3m_train_1m_test | mid_dynamic_cost | 10 | 10 |
| 3m_train_1m_test | actual_bid_ask | 10 | 10 |
| 6m_train_1m_test | mid_dynamic_cost | 7 | 5 |
| 6m_train_1m_test | actual_bid_ask | 6 | 5 |

## Heatmap Regeneration
- `outputs\diagnostics\heatmap_1_ab_postfix.svg`
- `outputs\diagnostics\heatmap_2_ab_postfix.svg`
- `outputs\diagnostics\heatmap_3_ab_postfix.svg`

No stable green regions appeared in the representative post-fix heatmaps; the high-composite areas remain fold-specific rather than broad and persistent.

## Honest Conclusion
Verdict remains NO STABLE EDGE - the bugs were real but fixing them did not produce a tradable result; the signal family genuinely lacks edge on this data.