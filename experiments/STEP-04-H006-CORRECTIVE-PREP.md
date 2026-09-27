# H006 纠错示范准备续记（2026-09-04）

Material Passport:
- Origin Skill: academic-research-suite
- Mode: implementation and bounded verification; no model training executed
- Date: 2026-09-04
- Verification Status: candidate generation and focused tests verified; final training preflight pending
- Version Label: corrective-candidates-v1 / source-equal-loss-v1

## 结论

补齐了来源明确的普通工具链和重试解释示范，但目前只是候选，不是 H001 自行答对的数据，
也不是最终训练集。本次没有运行 optimizer 更新，没有生成新模型权重，没有打开
promotion/sealed 题目，没有修改旧模型、生产权限、评分合同或 readiness 开关。

用户当前要求是“继续把准备工作弄完，然后开始后训练”。已提出待确认的流程调整：
训练数据安全检查通过后先做隔离的内部纠错训练，最终验收准备继续完成，晋升/上线门槛不变。
截至本记录写入，该调整没有收到明确回答，不视为获准，不能据此启动 GPU 训练。

## 已生成的纠错候选

位置：`artifacts/native-react-posttraining/step4-h006-data-20260903/corrective-oracle-v1/`。

| 候选池 | train 来源 / 示例 | shadow 来源 / 示例 |
|---|---:|---:|
| 普通工具链 | 180 / 360 | 36 / 72 |
| 可见冲突后的重试 | 60 / 60 | 12 / 12 |
| 合计 | 240 / 420 | 48 / 84 |

每个普通来源最多两条、不同动作的示范；主示范是实际执行过的 policy-owned
`get_poi_detail`，纠正模型在该状态过早调用路线工具的问题。补充示范是搜索或知识检索。
控制器执行的后续动作没有冒充模型目标。

普通示范复用冻结的 `easy-anchor-candidates-v2/source_feasibility.jsonl`，核对
manifest SHA256，再做 episode replay 和终局 hard-pass 检查。原始证明文件不变。
来源显式标为 synthetic / deterministic-rule-oracle。

重试示范来自 `hard-donor-candidates` 的独立 retry 来源，6 个模板族轮流选取。
教师只接收真实重放后、当时对模型可见的状态；必须看到允许 retry、solvable 能力和
失败的 validation report，才引用明确冲突作为 reason。教师不读取隐藏正确动作；
隐藏 metadata 仅由离线选题、环境重放和血缘审计使用。输入不含随后重算成功的结果。
监督目标仅含 reason，不含系统负责的 strategy、options 或固定连接句。

72/72 条重试来源均通过 H005 raw semantic、assembled system、执行、单次决策、
权限/组装合同、终局 hard pass、episode replay 检查。

重要：原始 legacy reward 对这 72 条仍全部标为 `task_failed`，因为它额外要求模型自己
输出固定连接句。原始字段保留，没有改为成功。候选筛选沿用冻结的 H005 raw/system
分离合同，不修改历史成绩，不把规则教师表现作为模型提升。

主要 SHA256：

- ordinary-train: `a4b16a66e285b150f21ff141544a47b5ee4e18eb1b92d196d204f755f210ce69`
- retry-train: `b1aa0035ebf845afe92a85ffd47a3577850712a95fd81c501df1dbfd44025d09`
- lineage: `52791dd8a488677796db0577af89f5d76d06ffaf26b5c13f808daa0e315c2219`
- retry_verified_rollouts: `c606bf8449d648785546aa1966555b2c1dfecca3f094faad6720a525617f53d3`

完整输入/输出 hash、target provenance 和原始验证字段见候选目录 manifest/provenance/rollouts。

## Source 等权训练实现

新增 `backend/src/agentic/source_equal_sft.py`，训练入口新增显式 opt-in：
`--source-equal-loss --source-lineage <lineage.jsonl>`。

目标：每个 source 等权，source 内每条示范等权，每条示范只对 completion token 求平均。
共有 N 条示范、S 个 source，一个 source 有 n 条示范时，其每条例子乘 N/(S*n)。
乘数在 token 平均之外，避免 microbatch=1 时被分母约掉。

v1 限制单进程、microbatch=1、完整 train/shadow、每 source 最多两条。
缺失/重复血缘、跨 split 来源、错误 split、过量示范均拒绝。
TRL tokenization 后按 source_record_id 核对权重和行数，避免权重丢失或配错。
source 权重只进入 loss，不进入模型输入；packing 仍关闭。

未开启新参数时保留原 token-weighted loss 行为。报告记录新目标合同和血缘 hash。
当前仅验证实现及数学/梯度测试，没有最终混合数据上的 Trainer preflight，
因此不能把 `trainer_source_weighting_verified` 置 true。

## 验证结果与问题

- 新候选完整生成，原始数据保留；输出目录拒绝覆盖。
- 本地聚焦 H006/SFT/source-weight 测试：55 passed、2 skipped（本地没有 torch）。
- 服务器 source 权重和原有 action/boundary 权重测试：15 passed，包含数学与梯度验证。
  训练镜像没有 SQLAlchemy，使用 `--confcutdir=backend/tests/unit/agentic` 排除无关
  Web 数据库 conftest；没有安装依赖或跳过所选测试断言。
- 本次相关文件 Ruff 通过。
- 扩大检查曾得到 58 passed、4 skipped、1 failed；失败为旧
  `test_status_bridge_is_verified_balanced_and_split_safe`，报
  `boundary target failed verification: curriculum-04159-...-paired-tradeoff:task_failed`。
  本次未修改该构建器；失败单独保留，不宣称仓库全绿。
- 服务器包依赖声明检查：torch 2.6.0、transformers 4.57.6、datasets 5.0.1、trl 1.9.2、
  peft 0.20.0、bitsandbytes 0.50.0、accelerate 1.14.0，已检查包间未发现声明冲突。

## 尚需完成，不能省略

1. 确认数据来源变化及“内部训练 / 最终晋升”的安排；旧冻结规则目前仍有效。
2. 组装正式 40/40/20 source/token 混合数据：复用 H004 合格 hard donor 和 H001
   boundary anchors，规则示范独立标记。补齐 shadow anchors，检查有效加权 token
   分布与最终跨池血缘。候选示范条数不等于独立来源数。
3. 衔接混合导出与训练入口：目前候选 mix manifest 不符合 `DatasetManifest` 所需字段，
   空 test.jsonl 还会触发 `TEST_SPLIT_EMPTY`。不得读取 sealed 或把 shadow 复制为 test
   以满足入口；需明确外部冻结测试合同。
4. protected hash-only registry、promotion 功效分析仍缺失；不伪造证明、不直接改开关。
5. 最终 tokenizer 0-truncation、实际 Trainer 权重、adapter hash、脏工作区代码快照、
   optimizer/总步数/seed、停止条件预检通过后，才执行获准的训练命令。

下一次从本记录继续，不重复跑已完成的 H001 普通题失败采样，不重新采样已经够用的
H004 donor，不把“导出了示范”当作“已经开训”。

## 2026-09-04 用户确认后的续记

用户同意：数据检查通过后先做隔离的内部纠错训练，最终验收和上线标准不变。
最终 mix v1 在 preflight 中发现 validation 有 1 个 source 跨类别重复，已停用；
v2 新增跨类别来源唯一性门后通过。v2 train 为 406 示例 / 300 来源，validation 为
90 示例 / 60 来源；已知禁用旧数据 1780 条的三类 hash 重合为 0，496/496 条 0 truncation，
source-equal mass 全部相等。训练配置、命令和监控见 `H006-INTERNAL-TRAINING-RUN.md`。
