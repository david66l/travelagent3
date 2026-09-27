# H006 内部纠错训练运行记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent run mode
- Mode: run
- Date: 2026-09-04
- Verification Status: COMPLETED, EXIT CODE 0
- Version Label: h006-internal-corrective-sft-v2-seed20260930

## 范围

本轮仅训练隔离的内部候选。它不替换线上模型，不允许访问 promotion/sealed payload，
不取得 promotion 资格。最终验收、protected registry 和功效分析保持为后续强制门。

## 冻结输入与预检

- 数据：`h006-internal-corrective-mix-v2`
- dataset version：`h006-internal-07eeda22a48a32ac`
- train：406 示例 / 300 独立来源；validation：90 示例 / 60 独立来源
- train source ratio：hard/easy/recovery = 0.35/0.41/0.24
- train completion-token ratio：0.41790179/0.38150746/0.20059075
- 每来源最多 2 条；source-equal mass 验证通过
- 与 1780 条已知禁用旧数据的 source ID/source hash/tool hash 重合均为 0
- sequence：496/496 可编码；max 3351，p95 3271，max_length 6144，0 truncation
- termination boundary：496/496 通过
- 输入快照：`run_input_snapshot.json`，9 个代码文件 + 7 个数据/审计文件逐项校验
- 起点：H001 checkpoint-48；继续同一个 adapter，不 merge、不 stack

第一次 v1 mix 因 validation 有 1 个 source 跨类别重复而作废；v2 已新增跨类别硬检查，
最终为 360 个独立 train+shadow source。v1 不可用于训练。

## 冻结训练配置

- 1 epoch，约 34 optimizer steps
- lr 5e-6，AdamW，linear scheduler，warmup 0.05，weight decay 0
- microbatch 1，gradient accumulation 12，effective batch 12
- LoRA r16/alpha32/dropout 沿用 checkpoint；max grad norm 1
- completion-only loss，packing false，source-equal loss
- seed 20260930；checkpoint/eval 每 8 steps，最多保留 4 个
- output 始终 quarantine；外部冻结 test 不加载进 Trainer

## 命令和监控

已确认并启动的唯一命令：

```bash
bash /root/autodl-tmp/TravelAgent2-h005-eval-20260903/scripts/run_h006_internal_corrective_sft.sh
```

- 启动时间：2026-09-04 12:12:46 +08:00
- 远端 PID：102787
- 启动后约 17 秒 RSS：705124 KiB
- 启动脚本 SHA256：`d662b20de592854e8e78234cb5fee68eef1fcfe09f2f4473b6bf20c00797a8c9`
- 输入快照 SHA256：`b5d275297076158a68dff4479b399ba323a6c41b99419715f6f1347fa35118be`

期望输出：

- `artifacts/native-react-posttraining/h006-internal-corrective-sft-v2-seed20260930/`
- `artifacts/native-react-posttraining/logs/h006-internal-corrective-sft-v2-seed20260930.log`
- `training_report.json` 及 checkpoint-8/16/24/32

监控：进程存活、日志增长、GPU、loss/grad_norm、输出 checkpoint；硬超时 30 分钟。
除硬超时外，异常只报告，不自动重试或自动停止。崩溃保留原日志，不改参数重跑。

## 运行结果

- 完成时间：2026-09-04 12:21:03 +08:00；全流程约 8 分 17 秒
- Trainer：34/34 optimizer steps，1.0 epoch；训练主体 436.6794 秒
- 平均 train loss：0.22613573
- validation loss：step 8/16/24/32 = 0.34509045 / 0.29989755 / 0.28421548 / 0.28101045
- 最终 adapter validation loss：0.28181732
- 最低 validation loss 在 `checkpoint-32`；最终根目录 adapter 对应 step 34
- 保存 checkpoint：16、24、32、34；checkpoint-8 曾成功保存，随后按 `save_total_limit=4` 正常清理
- 源 adapter SHA256 前后均为 `734a7fcdea9509071fddb07ce7c9d3020bad1ed35690b81e6d46c92d98019924`
- 输出 adapter SHA256：`3aabbe1f65c3ff962e659798ffd4ba2569736b1e0a2a692ec061b391a85bb515`
- 进程正常退出；收尾后 GPU 显存回落到 1 MiB
- `training_report.json` SHA256：`47628e0decc201de5da52cfac03473876d58112061c10963d2dbed0d6a36ace4`
- `training.log` SHA256：`190d469f273017ab5bd9edc18aa8070cec2ac7b743b79c359b521d4a400ec4bc`

报告自检结果：`status=trained`；训练/验证 source-equal 均通过；模型与终止边界预检均通过。
候选仍是 `quarantine_pending_agent_loop_eval`，`promotion_eligible=false`。这表示优化器训练健康完成，
不等于 Agent 任务效果已经提升，也不允许直接 promotion。

## 下一道门

先用同一套内部 Agent Loop 用例盲测 H001 基线、H006 checkpoint-32 和 H006 step-34，
核对任务成功率、工具协议、重试恢复和安全回归；根据冻结规则选 checkpoint。通过后才进入外部冻结测试与功效分析。
