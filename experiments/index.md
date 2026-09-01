# 实验总账 — native-react-posttraining（107 个 run 目录）

> 自动生成于 2026-09-01 18:55，来源：服务器 artifact 目录名 + training_report/eval report 提取。
> 裁决含义：quarantine_pending_agent_loop_eval=未获晋升资格 | ZERO-VARIANCE=GRPO 组内无梯度 | has-variance=有学习信号

## 概览

**?**: 22 **AUDIT**: 8 **EVAL**: 2 **GRPO**: 42 **SFT**: 32 **SMOKE**: 1

| 日期 | 类型 | Run（截短） | 状态 | 关键指标 | 裁决 |
|---|---|---|---|---|---|
| 08-30 11:28 | SFT | native-react-sft-v1 |  |  |  |
| 08-30 13:39 | AUDIT | curriculum-audit-v1 | audit | groups=4 zero_var=0 success=0.000 |  |
| 08-30 13:44 | AUDIT | curriculum-audit-v2 | audit | groups=4 zero_var=1 success=0.375 |  |
| 08-30 13:50 | AUDIT | curriculum-audit-v3-offset4 | audit | groups=12 zero_var=4 success=0.667 |  |
| 08-30 13:59 | ? | curriculum-validation-baseline-v1 | audit | groups=8 zero_var=1 success=0.594 |  |
| 08-30 14:16 | ? | curriculum-validation-baseline-sanitized-v1 | audit | groups=8 zero_var=2 success=0.688 |  |
| 08-30 14:20 | ? | curriculum-train-sanitized-v1 | audit | groups=9 zero_var=3 success=0.750 |  |
| 08-30 14:34 | ? | decision-state-validation-baseline-v1 | audit | groups=6 zero_var=6 success=1.000 |  |
| 08-30 14:36 | ? | decision-state-train-explore12-v1 | audit | groups=5 zero_var=5 success=1.000 |  |
| 08-30 14:39 | ? | decision-state-train-real-schema-v1 | audit | groups=5 zero_var=4 success=0.975 |  |
| 08-30 14:46 | ? | decision-state-history-train-v1 | audit | groups=5 zero_var=4 success=0.025 |  |
| 08-30 14:53 | ? | decision-state-history-bridge-train-v1 | audit | groups=5 zero_var=2 success=0.825 |  |
| 08-30 15:01 | ? | decision-state-history-validation-bridge-v1 | audit | groups=6 zero_var=2 success=0.875 |  |
| 08-30 15:03 | GRPO | decision-state-history-validation-grpo-step1-v1 | audit | groups=6 zero_var=3 success=0.896 |  |
| 08-30 15:06 | GRPO | decision-state-history-validation-grpo-step2-v1 | audit | groups=6 zero_var=2 success=0.875 |  |
| 08-30 15:10 | ? | full-history-validation-bridge-v1 | audit | groups=8 zero_var=5 success=0.844 |  |
| 08-30 15:15 | GRPO | full-history-validation-grpo-step1-v1 | audit | groups=8 zero_var=1 success=0.531 |  |
| 08-30 15:23 | ? | full-history-validation-bridge-controller-owned-v2 | audit | groups=8 zero_var=4 success=1.000 |  |
| 08-30 15:26 | GRPO | full-history-validation-grpo-step1-controller-owned-v2 | audit | groups=8 zero_var=5 success=1.000 |  |
| 08-30 15:33 | ? | full-history-validation-bridge-offset8-12x4-v3 | audit | groups=12 zero_var=7 success=0.917 |  |
| 08-30 15:37 | GRPO | full-history-validation-grpo-step1-offset8-12x4-v3 | audit | groups=12 zero_var=7 success=0.938 |  |
| 08-30 15:39 | ? | full-history-validation-bridge-offset20-3x4-v4 | audit | groups=3 zero_var=1 success=1.000 |  |
| 08-30 15:41 | GRPO | full-history-validation-grpo-step1-offset20-3x4-v4 | audit | groups=3 zero_var=0 success=0.917 |  |
| 08-30 15:53 | GRPO | decision-state-history-validation-grpo-kl001-lr5e7-v3 | audit | groups=6 zero_var=6 success=1.000 |  |
| 08-30 15:56 | GRPO | full-history-validation-grpo-kl001-first8-8x4-v5 | audit | groups=8 zero_var=5 success=1.000 |  |
| 08-30 16:02 | GRPO | full-history-validation-grpo-kl001-offset8-12x4-v5 | audit | groups=12 zero_var=7 success=0.917 |  |
| 08-30 16:03 | GRPO | full-history-validation-grpo-kl001-offset20-3x4-v5 | audit | groups=3 zero_var=1 success=1.000 |  |
| 08-30 23:00 | ? | logs |  |  |  |
| 08-31 02:23 | SFT | verifier-repair-sft-baseline-val12x4 | audit | groups=12 zero_var=12 success=0.000 |  |
| 08-31 02:50 | SFT | verifier-repair-sft-e05-val12x4 | audit | groups=12 zero_var=4 success=0.479 |  |
| 08-31 03:03 | SFT | verifier-repair-sft-e05-train24x4-routing | audit | groups=24 zero_var=12 success=0.760 |  |
| 08-31 03:20 | GRPO | verifier-repair-grpo-c9-val12x4 | audit | groups=12 zero_var=3 success=0.438 |  |
| 08-31 03:25 | GRPO | verifier-repair-grpo-c11-val12x4 | audit | groups=12 zero_var=3 success=0.458 |  |
| 08-31 03:30 | SFT | verifier-repair-sft-e05-val12x4-fixed | audit | groups=12 zero_var=4 success=0.479 |  |
| 08-31 03:36 | SFT | verifier-repair-sft-e05-train24x4-dense-reroute |  |  |  |
| 08-31 03:46 | SFT | verifier-repair-sft-e05-train-offset8-24x4-dense | audit | groups=24 zero_var=8 success=0.729 |  |
| 08-31 05:06 | GRPO | verifier-repair-grpo-dense-v3-c4-val12x4 | audit | groups=12 zero_var=1 success=0.458 |  |
| 08-31 05:12 | GRPO | verifier-repair-grpo-dense-v3-c8-val12x4 | audit | groups=12 zero_var=1 success=0.521 |  |
| 08-31 05:16 | GRPO | verifier-repair-grpo-dense-v3-c12-val12x4 | audit | groups=12 zero_var=2 success=0.479 |  |
| 08-31 05:20 | GRPO | verifier-repair-grpo-dense-v3-c16-val12x4 | audit | groups=12 zero_var=1 success=0.458 |  |
| 08-31 05:24 | GRPO | verifier-repair-grpo-dense-v3-final-val12x4 | audit | groups=12 zero_var=1 success=0.500 |  |
| 08-31 05:35 | SFT | verifier-repair-sft-e05-val48x4-fixed | audit | groups=48 zero_var=6 success=0.464 |  |
| 08-31 05:48 | GRPO | verifier-repair-grpo-dense-v3-c8-val48x4 | audit | groups=48 zero_var=5 success=0.495 |  |
| 08-31 05:56 | SFT | verifier-repair-sft-e05-envelope-smoke3x4-v2 | audit | groups=3 zero_var=1 success=0.417 |  |
| 08-31 05:59 | GRPO | verifier-repair-grpo-dense-v3-c8-envelope-smoke3x4-v2 | audit | groups=3 zero_var=1 success=0.500 |  |
| 08-31 06:02 | SFT | verifier-repair-sft-e05-envelope-cache-smoke3x4 | audit | groups=3 zero_var=1 success=0.417 |  |
| 08-31 06:33 | SFT | verifier-repair-semantic-sft-v2-c40-val12x4 | audit | groups=12 zero_var=5 success=0.729 |  |
| 08-31 06:36 | SFT | verifier-repair-semantic-sft-v2-c60-val12x4 | audit | groups=12 zero_var=8 success=0.875 |  |
| 08-31 06:40 | SFT | verifier-repair-semantic-sft-v2-c68-val12x4 | audit | groups=12 zero_var=8 success=0.896 |  |
| 08-31 06:53 | SFT | verifier-repair-semantic-sft-v2-c68-val48x4 | audit | groups=48 zero_var=21 success=0.807 |  |
| 08-31 06:59 | SFT | verifier-repair-semantic-sft-v2-train-o0-24x4 | audit | groups=24 zero_var=17 success=0.885 |  |
| 08-31 07:09 | SFT | verifier-repair-semantic-sft-v2-rlchallenge-o0-24x4 | audit | groups=24 zero_var=16 success=0.844 |  |
| 08-31 07:16 | SFT | verifier-repair-semantic-sft-v2-rlchallenge-o8-24x4 | audit | groups=24 zero_var=18 success=0.885 |  |
| 08-31 07:23 | SFT | verifier-repair-semantic-sft-v2-rlchallenge-o16-24x4 | audit | groups=24 zero_var=16 success=0.875 |  |
| 08-31 07:30 | GRPO | verifier-repair-grpo-rlchallenge-b0-e3-lr2e7-v4-interrupted-prestep |  |  |  |
| 08-31 07:31 | GRPO | verifier-repair-grpo-rlchallenge-b0-e1-lr2e7-v4 |  |  |  |
| 08-31 08:13 | GRPO | verifier-repair-grpo-rlchallenge-v4-c2-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.729 |  |
| 08-31 08:17 | GRPO | verifier-repair-grpo-rlchallenge-v4-c4-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.771 |  |
| 08-31 08:23 | GRPO | verifier-repair-grpo-rlchallenge-v4-c6-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.750 |  |
| 08-31 08:28 | GRPO | verifier-repair-grpo-rlchallenge-v4-c8-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.750 |  |
| 08-31 08:31 | GRPO | verifier-repair-grpo-rlchallenge-v4-c10-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.771 |  |
| 08-31 08:47 | GRPO | verifier-repair-grpo-rlchallenge-v4-c4-val48x4 | audit | groups=48 zero_var=22 success=0.812 |  |
| 08-31 08:49 | GRPO | verifier-repair-grpo-rlchallenge-b01-e1-lr5e8-v5 |  |  |  |
| 08-31 09:30 | GRPO | verifier-repair-grpo-rlchallenge-v5-c12-val-o4-12x4 | audit | groups=12 zero_var=3 success=0.688 |  |
| 08-31 09:37 | GRPO | verifier-repair-grpo-rlchallenge-v5-c6-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.729 |  |
| 08-31 10:00 | SFT | verifier-repair-semantic-sft-v2-rlchallenge-retry-o24-48x8-t12 | audit | groups=48 zero_var=28 success=0.917 |  |
| 08-31 10:08 | GRPO | verifier-repair-grpo-rlchallenge-b008-e1-lr1e7-v6 |  |  |  |
| 08-31 10:59 | GRPO | verifier-repair-grpo-rlchallenge-v6-c8-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.750 |  |
| 08-31 11:05 | GRPO | verifier-repair-grpo-rlchallenge-v6-c4-val-o4-12x4 | audit | groups=12 zero_var=4 success=0.750 |  |
| 08-31 11:10 | GRPO | verifier-repair-grpo-rlchallenge-v6-c12-val-o4-12x4 | audit | groups=12 zero_var=3 success=0.729 |  |
| 08-31 11:24 | GRPO | verifier-repair-grpo-rlchallenge-b008-e1-lr1e7-g8t12-v7-oom-attempt1 |  |  |  |
| 08-31 11:30 | GRPO | verifier-repair-grpo-rlchallenge-b008-e1-lr1e7-g8t12-v7 |  |  |  |
| 08-31 14:19 | SMOKE | verifier-repair-p1-smoke-runtime-audit |  |  |  |
| 08-31 14:27 | SFT | verifier-repair-p1-sft-baseline-smoke12x4 s20260909 | audit | groups=12 zero_var=8 success=0.354 |  |
| 08-31 14:29 | GRPO | verifier-repair-p1-grpo-smoke3step-g4 s20260909 |  |  |  |
| 08-31 14:38 | SFT | verifier-repair-p1-sft-train3x4 s20260909 | audit | groups=3 zero_var=1 success=0.500 |  |
| 08-31 17:57 | GRPO | verifier-repair-p1-grpo-smoke1step-g4 s20260909-v2 |  |  |  |
| 08-31 18:57 | GRPO | verifier-repair-p1-grpo-retry-step1-g4 s20260909-v1 |  |  |  |
| 08-31 19:52 | SFT | verifier-repair-p1-v5-route-scoped-frozen-sft-3x4 s20260909-v1 | audit | groups=3 zero_var=2 success=0.500 |  |
| 08-31 20:21 | SFT | verifier-repair-p1-v5-route-scoped-frozen-sft-shared-render-3x4 s20260909-v1 | audit | groups=3 zero_var=1 success=0.417 |  |
| 08-31 21:00 | GRPO | verifier-repair-p1-v5-retry-shared-render-grpo-diagnostic-step1-g4 s2026090... |  |  |  |
| 08-31 21:23 | ? | verifier-repair-p1-v5-retry-shared-render-actual-trainer-rollout-only-g4 s2... |  |  |  |
| 08-31 21:28 | ? | verifier-repair-p1-v5-retry-shared-render-actual-trainer-rollout-only-g4 s2... |  |  |  |
| 08-31 22:16 | SFT | Q3-1.7B-verifier-tradeoff-grounding-sft-e2-lr1e5 s20260910-v1 | trained | train_loss=0.09733 eval_loss=0.03288 tok_acc=0.999 | quarantine_pending_agent_loop_eval |
| 08-31 22:32 | ? | tradeoff-grounding-internal-dev-baseline s20260910-v1 |  | action=0.700 reason=0.680 R=-1.000 |  |
| 08-31 22:49 | ? | tradeoff-grounding-internal-dev-baseline s20260910-v2 |  | full=0.260 action=0.700 reason=0.680 R=0.263 |  |
| 08-31 23:00 | ? | tradeoff-grounding-internal-dev-baseline s20260910-v3 |  | full=0.260 action=0.700 reason=0.680 R=0.263 |  |
| 08-31 23:04 | ? | tradeoff-grounding-internal-dev-epoch1 s20260910-v1 |  | full=0.300 action=0.840 reason=0.820 R=0.520 |  |
| 08-31 23:08 | ? | tradeoff-grounding-internal-dev-epoch2 s20260910-v1 |  | full=0.420 action=0.900 reason=0.900 R=0.663 |  |
| 08-31 23:51 | ? | tradeoff-grounding-internal-dev-epoch2-controller-options-v2 s20260910-v1 |  | full=0.900 action=0.940 reason=0.900 R=0.869 |  |
| 09-01 00:50 | AUDIT | p0-contract-audit-v1 |  |  |  |
| 09-01 01:01 | EVAL | formal-zero-opt-epoch2-v4-internal-dev-p0-r2 |  | full=0.600 action=0.700 reason=0.600 R=0.367 |  |
| 09-01 01:23 | AUDIT | p0-contract-audit-v2 |  |  |  |
| 09-01 01:32 | EVAL | formal-zero-opt-epoch2-v5-internal-dev-p0 |  | full=0.450 action=0.812 reason=0.450 R=0.508 |  |
| 09-01 11:07 | AUDIT | p0-contract-audit-v3 |  |  |  |
| 09-01 13:17 | SFT | Q3-1.7B-verifier-tradeoff-grounding-sft-v8-e2-lr1e5 s20260901* | trained | train_loss=0.0245 eval_loss=0.0006077 tok_acc=1.000 full=0.775 action=0.912 reason=0.775 R=0.778 | quarantine_pending_agent_loop_eval |
| 09-01 13:46 | SFT | Q3-1.7B-verifier-tradeoff-grounding-sft-v9-balanced-e2-lr1e5 s20260901* | trained | train_loss=0.02643 eval_loss=0.0006039 tok_acc=1.000 full=0.825 action=0.963 reason=0.825 R=0.881 | quarantine_pending_agent_loop_eval |
| 09-01 14:01 | GRPO | Q3-1.7B-verifier-v9-balanced-grpo-p0-b008-e1-lr5e7-g4 s20260901* | ineligible | group_rewards=[1.0, 1.0, 1.0, 1.0] uniq=1 | ZERO-VARIANCE (no gradient) |
| 09-01 14:06 | GRPO | Q3-1.7B-verifier-v9-balanced-grpo-p0-retry-rollout-g4 s20260903* | ineligible | group_rewards=[1.0, 1.0, 1.0, 1.0] uniq=1 | ZERO-VARIANCE (no gradient) |
| 09-01 14:11 | AUDIT | Q3-1.7B-verifier-v9-balanced-trainonly-retry-variance-audit-g4-t08 s20260911 | audit | groups=12 zero_var=2 success=0.354 |  |
| 09-01 14:22 | AUDIT | Q3-1.7B-verifier-v9-balanced-trainonly-tradeoff-abort-variance-audit-g4-t08... | audit | groups=24 zero_var=21 success=0.938 |  |
| 09-01 16:20 | SFT | Q3-1.7B-verifier-reason-quality-sft-v10-from-v9-e2-lr5e6 s20260921* | trained | train_loss=1.944 eval_loss=1.69 tok_acc=0.815 full=0.000 action=0.925 reason=0.850 R=0.632 | quarantine_pending_agent_loop_eval |
| 09-01 16:34 | SFT | Q3-1.7B-verifier-reason-quality-sft-v11-step0-base-eval |  | full=0.050 action=0.875 reason=0.263 R=0.419 |  |
| 09-01 16:50 | SFT | Q3-1.7B-verifier-reason-quality-sft-v11-rationale-first-balanced-diagnostic... | trained | train_loss=1.742 eval_loss=1.292 tok_acc=0.783 full=0.000 action=0.887 reason=0.487 R=0.459 | quarantine_pending_agent_loop_eval |
| 09-01 17:18 | SFT | Q3-1.7B-verifier-reason-quality-sft-v11-rationale-first-balanced-h24-stop12... | trained | train_loss=1.757 eval_loss=1.182 tok_acc=0.791 full=0.013 action=0.925 reason=0.537 R=0.512 | quarantine_pending_agent_loop_eval |
| 09-01 17:28 | SFT | Q3-1.7B-verifier-reason-quality-sft-v12-rationale-weight3-h24-stop12 s20260... | trained | train_loss=38.1 eval_loss=2.24 | quarantine_pending_agent_loop_eval |
| 09-01 17:46 | SFT | Q3-1.7B-verifier-reason-quality-sft-v13-rationale-weight3-gradacc-fixed-h24... | trained | train_loss=3.17 eval_loss=2.235 full=0.000 action=0.863 reason=0.362 R=0.398 | quarantine_pending_agent_loop_eval |
