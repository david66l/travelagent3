# 性能优化日志 — 2026-04-29

## 优化目标

基于实际运行数据（6分7秒 → 目标2~3分钟），从5个维度实施优化：合并Proposal、搜索去LLM化、Planner精简、删除Validation、意图识别瘦身。

---

## 修改文件清单

### 1. `backend/src/agents/itinerary_planner.py`

**修改点A：精简POI上下文**
- `pois[:35]` → `pois[:20]`（减少43%的POI输入）
- 去掉字段：简介(description)、门票(ticket_price)、开放时间(open_time)、室内外类型(indoor_outdoor)
- 只保留：名称、分类、标签、区域、建议时长、最佳时间

**修改点B：去掉思考过程要求**
- 删除"Step 1-6 思考过程"的要求（减少~800 completion tokens）
- prompt直接要求输出 Markdown方案 + `===JSON===` + JSON

**修改点C：直接输出Markdown方案**
- user_prompt 要求 LLM 先输出完整 Markdown 旅行方案（含概述、按天行程、实用贴士、预算汇总、住宿建议、避坑清单）
- 然后输出 `===JSON===` 分隔符 + JSON

**修改点D：解析逻辑升级**
- `_parse_llm_response` 返回 `(markdown_text, json_dict)` 元组
- planner_node 同时设置 `assistant_response` 和 `current_itinerary`
- 新增 `_last_proposal` 属性供外部读取

**预期收益**：
- 单步耗时：139s → ~95s（省44s，31%）
- 砍掉了独立的 proposal LLM 调用（135s）

---

### 2. `backend/src/graph/nodes.py`

**修改点：planner_node 返回 assistant_response**
```python
proposal_text = getattr(agent, '_last_proposal', None)
if proposal_text:
    result["assistant_response"] = proposal_text
    result["proposal_text"] = proposal_text
    result["waiting_for_confirmation"] = True
```

**预期收益**：
- proposal_node 不再需要，图结构中删除

---

### 3. `backend/src/graph/graph.py`

**修改点A：删除 proposal_node**
- 移除导入：`proposal_node`
- 移除节点注册：`builder.add_node("proposal_node", proposal_node)`
- 移除边：`apply_routes_node → proposal_node → format_output_node`
- 改为：`apply_routes_node → format_output_node`

**修改点B：删除 validation_node**
- 移除导入：`validation_node`
- 移除节点注册：`builder.add_node("validation_node", validation_node)`
- 移除边：`planner_node → validation_node → apply_routes_node`
- 改为：`planner_node → route_node/budget_calc_node → apply_routes_node`

**预期收益**：
- proposal LLM 调用：135s → 0s
- validation 网络搜索：9s → 0s

---

### 4. `backend/src/skills/poi_search.py`

**修改点A：新增规则提取方法 `_extract_pois_by_rule`**
- 正则提取：`"名称"`、书名号《名称》、编号列表
- 城市 fallback 列表精确匹配
- 零LLM调用，纯正则+字符串匹配

**修改点B：调整调用优先级**
- PRIMARY：`_extract_pois_by_rule`（规则提取）
- SECONDARY：`_extract_pois_from_answer`（LLM提取，仅当规则提取<10个时触发）
- TERTIARY：`_extract_from_snippets`（snippet提取，仅当总数<15时触发）

**预期收益**：
- 搜索景点：76s → ~15s（去掉LLM提取的~35s + 队列等待的~30s）

---

### 5. `backend/src/agents/intent_recognition.py`

**修改点：精简消息历史**
- `messages[-10:]` → `messages[-2:]`
- prompt token 从 ~2000 → ~500

**预期收益**：
- 意图识别：7s → ~3s（TTFT减少，prompt prefill更快）

---

## 测试结果

```
platform darwin -- Python 3.13.7, pytest-9.0.3
asyncio: mode=Mode.STRICT

tests/test_date_resolution.py ........
tests/test_poi_matching.py ......
tests/test_validation.py ...

============================== 20 passed in 0.22s ==============================
```

所有20个测试通过，无回归。

---

## 预期效果汇总

| 优化项 | 当前耗时 | 优化后 | 收益 |
|--------|---------|--------|------|
| Planner 精简 | 139s | ~95s | -44s |
| 砍掉 Proposal LLM | 135s | 0s | -135s |
| 搜索去LLM化 | 76s | ~15s | -61s |
| 删除 Validation | 9s | 0s | -9s |
| 意图识别瘦身 | 7s | ~3s | -4s |
| **总计** | **366s** | **~113s** | **-253s (69%)** |

**保守估计**：6分7秒 → 2.5~3分钟（节省50~60%）

---

## 后续可继续优化

1. **按天分片并行规划**：5天行程拆成5个并行LLM调用，planner可再省~60s
2. **预计算模板库**：热门城市×常见天数提前生成模板，匹配时直接微调
3. **缓存TTL延长**：POI缓存1小时→7天，上下文缓存24小时→30天
4. **流式输出**：Markdown方案可流式推送到前端，用户更快看到内容

---

## 备注

- `validation_node` 和 `proposal_node` 的代码仍保留在 `nodes.py` 和 `agents/` 中，但 `graph.py` 不再引用，属于死代码，如需清理可后续删除
- `proposal_generation.py` 文件保留，但不再被 graph 调用
- 前端无需修改：planner_node 直接输出的 `assistant_response` 格式与之前 proposal_node 一致（Markdown）
