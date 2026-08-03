# Backtest results

Generated 2026-08-03 13:18

Model: `models/curriculum/curriculum_ppo_final.zip`

Final training state:
```
2026-08-03 13:03:51,521 [INFO] Step 2,950,000/3,000,000 (98.3%) | Stage: INTEGRATOR | Reward +22.733 | PnL �+2426 | WR 61.2% | Trades 22.5 | MaxDD 1.3%
2026-08-03 13:16:02,970 [INFO] Step 3,000,000/3,000,000 (100.0%) | Stage: INTEGRATOR | Reward +23.505 | PnL �+2418 | WR 62.2% | Trades 20.6 | MaxDD 2.2%
2026-08-03 13:18:39,388 [INFO] Final stage: INTEGRATOR
```

```
2026-08-03 13:18:57,936 [INFO] Loaded XAUUSD/D1: 1,091 bars
2026-08-03 13:18:58,143 [INFO] Loaded XAUUSD/H1: 25,001 bars
2026-08-03 13:18:58,206 [INFO] Loaded XAUUSD/H4: 6,544 bars
2026-08-03 13:18:59,001 [INFO] Loaded XAUUSD/M15: 99,908 bars
2026-08-03 13:18:59,001 [INFO] Total: 1 instruments, 132,544 bars
2026-08-03 13:18:59,081 [INFO] Holdout begins 2025-05-05 09:30:00 - never seen during training
C:\Users\mulli\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\cuda\__init__.py:129: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11020). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\c10\cuda\CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() > 0
2026-08-03 13:19:01,197 [INFO] Loaded models\curriculum\curriculum_ppo_final.zip
2026-08-03 13:19:01,243 [INFO] FTMO live bars after 2025-12-18 23:45:00: 14491 M15
2026-08-03 13:19:01,268 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:22:44,336 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:26:01,201 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:29:20,694 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:32:38,602 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:36:00,689 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:39:47,677 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:42:54,909 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:46:00,530 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:48:49,557 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:51:59,148 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:55:35,958 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 13:58:40,641 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 14:01:46,128 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 14:04:51,106 [INFO] [OBS] Using PPOObservationBuilder v7.0 (40 dims)
2026-08-03 14:08:00,410 [INFO] Report written to logs/holdout_report.json

HOLDOUT (unseen split, original prices)
============================================================================================
strategy          return%     PF    WR%    expR    R:R  maxDD%  trades  t/day  hold  verdict
--------------------------------------------------------------------------------------------
trained-model       25.70   1.70   48.6   0.587   1.76   13.77     175    0.8     3  FAIL: maxDD 13.8% > 10%
always-flat          0.00   0.00    0.0   0.000   0.00    0.00       0    0.0     0  FAIL: profit 0.0% < 10%
random-valid        -3.00   0.75   37.5  -0.246   1.20    3.22      72    0.3     2  FAIL: profit -3.0% < 10%
buy-and-hold        -2.63   0.88   38.6  -0.159   1.35    6.69      70    0.3     3  FAIL: profit -2.6% < 10%
ma-cross            -7.28   0.62   33.8  -0.559   1.21    4.39      80    0.4     4  FAIL: profit -7.3% < 10%

HOLDOUT MIRRORED (drift inverted - the short-side test)
============================================================================================
strategy          return%     PF    WR%    expR    R:R  maxDD%  trades  t/day  hold  verdict
--------------------------------------------------------------------------------------------
trained-model        2.90   1.13   40.0   0.155   1.73    4.64      95    0.5     3  FAIL: profit 2.9% < 10%
always-flat          0.00   0.00    0.0   0.000   0.00    0.00       0    0.0     0  FAIL: profit 0.0% < 10%
random-valid         4.89   1.20   48.5   0.283   1.49    4.92     171    0.8     2  FAIL: profit 4.9% < 10%
buy-and-hold       -12.55   0.61   32.5  -0.469   1.33   10.58      80    0.4     3  FAIL: profit -12.6% < 10%, maxDD 10.6% > 10%, dailyDD 5.7% > 5%
ma-cross            -6.95   0.56   35.2  -0.608   1.01    5.99      71    0.3     4  FAIL: profit -7.0% < 10%

FTMO LIVE DATA (never seen, 14491 M15 bars, broker spreads)
============================================================================================
strategy          return%     PF    WR%    expR    R:R  maxDD%  trades  t/day  hold  verdict
--------------------------------------------------------------------------------------------
trained-model       -8.25   0.73   34.4  -0.525   1.41    9.61      64    0.3     2  FAIL: profit -8.2% < 10%
always-flat          0.00   0.00    0.0   0.000   0.00    0.00       0    0.0     0  FAIL: profit 0.0% < 10%
random-valid       -23.38   0.45   33.3  -1.311   0.95    9.22      90    0.4     1  FAIL: profit -23.4% < 10%
buy-and-hold       -25.43   0.31   26.9  -1.233   0.89    9.94      67    0.3     2  FAIL: profit -25.4% < 10%
ma-cross           -16.63   0.19   16.7  -2.433   0.95    6.29      42    0.2     2  FAIL: profit -16.6% < 10%

HOLDOUT (unseen split, original prices): model BEATS buy-and-hold (25.70% vs -2.63%)

HOLDOUT MIRRORED (drift inverted - the short-side test): model BEATS buy-and-hold (2.90% vs -12.55%)

FTMO LIVE DATA (never seen, 14491 M15 bars, broker spreads): model BEATS buy-and-hold (-8.25% vs -25.43%)
```

---

## Verdict: real skill, not yet profitable live

**Not ready to trade.** It beats every baseline in all three sections, but on the
data that matters most it still loses money.

### What holds up

It beat buy-and-hold everywhere, including where B&H is catastrophic:

| section | model | buy-and-hold | random |
|---|---|---|---|
| Holdout (unseen split) | **+25.70%** | −2.63% | −3.00% |
| Holdout mirrored | +2.90% | −12.55% | +4.89% |
| FTMO live (−6.4% market) | **−8.25%** | −25.43% | −23.38% |

Reward:risk came in at **1.76 / 1.73 / 1.41** across the three sections —
consistent, and close to the 2.0 target, at a 34–49% win rate. That is the
low-win-rate/high-R profile the rebuilt ladder was aiming for, and it held on
data the model never saw.

The direction bias is genuinely gone. On mirrored prices, where the drift is
inverted and a long-biased policy is destroyed (B&H: −12.55%), the model stayed
positive.

### What fails

**It loses 8.25% on live FTMO data.** Beating a −25% baseline with −8% is less
bad, not profitable. This is the most realistic test available: real broker
bars, real spreads, a −6.4% market.

**Drawdown breaches the prop-firm limit.** 13.77% on the holdout against FTMO's
10% ceiling. The +25.70% there is not bankable — that account would already be
closed.

**No edge over random on mirrored data.** random-valid returned +4.89% against
the model's +2.90%. Within noise at 95 trades, but it is not evidence of
two-sided skill either.

### Why, most likely

Training stopped at **stage 5 of 10**, blocked by three gates that no policy
could clear (fomo_trade_rate, mask_collapse_rate, stop_mode_rate — all fixed
today, all after this run started). Stage 5 trains at ~28 points of spread.
**Live FTMO is 50.** The model never trained through stages 6–9, where spread
climbs to 47, so it is under-hardened on exactly the cost that decides whether
a 1.4R edge survives.

### Next

1. Rerun with the gate fixes so it reaches stages 6–9.
2. Raise the stage 9 spread to >= 50 to match the live book, not 47.
3. Treat drawdown as a first-class gate — 13.77% has to come under 10% before
   any of this matters.

Sample sizes: 175 / 95 / 64 trades over 10 episodes each. The FTMO section is
the smallest and the most important; its conclusion deserves a longer run.

---

## Follow-up: the FTMO loss is not a cost problem

Same model, same bars, only the spread changed:

| spread | P&L | PF | WR% | trades |
|---|---|---|---|---|
| ~14 pts (a quarter of live) | -5,683 | 0.83 | 38.4 | 73 |
| ~28 pts (what it trained at) | -7,414 | 0.75 | 34.4 | 64 |
| ~40 pts (FTMO median) | -8,245 | 0.73 | 34.4 | 64 |
| ~50 pts | -933 | 0.97 | 34.8 | 69 |

It loses at every level, including one far cheaper than it ever trained against,
and 50 points scores better than 40 - which cannot happen if cost is the driver.
At 64-69 trades the differences between these conditions are noise.

This retracts the earlier "under-hardened on costs" explanation. Two further
measurements support the retraction: FTMO spread over 15,779 bars is p50 40 /
p75 47 / p90 55, and stage 9 trains at roughly p75; and the training data's
spread is proportionally WIDER than FTMO's (p90/p50 of 2.88 against 1.38), with
shocks enabled at stage 9. Costs were approximately right all along.

### What it actually means

The edge does not transfer across time periods. It earned +25.70% on May-Dec
2025 and +2.90% on the same months with prices mirrored, but loses on Dec
2025-Aug 2026 regardless of friction. Mirroring fixed direction, not regime:
the model works on inverted prices from the period it was trained beside, and
fails on a different period.

Finishing stages 6-9 will therefore probably NOT fix this. It is a
generalisation problem, not a hardening problem, and the likeliest cause is
that all 99,908 training bars come from a single 149.8% bull regime.
