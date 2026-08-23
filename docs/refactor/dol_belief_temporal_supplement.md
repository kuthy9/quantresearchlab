# DOL、Path Belief 与 Temporal Research 补充边界

状态：2026-08-23 只读数据审计后的实施与研究边界。

本文件补充 DOL probability architecture、Trading Brain 概率更新、相关证据、
belief expiry、temporal relation 和 branching path research。它不改写既有
SMC 语义，也不把 development association 升级为 calibration、预测、因果、
OOS 或交易权限。

## Current

### 已存在的工程边界

- Trading Eye 继续只发布确定性的 DOL candidates 与上下文事实；当前
  [DOL protocol](../../configs/dol_probability.json) 保留
  `no_target_before_common_horizon`，并把 calibration 状态明确标为
  `not_fitted_not_admitted`、authority 标为 `shadow_only`。
- Foundation DOL adapter 只消费 `MarketSnapshot` 中仍 active/rearmed 且绑定
  exact interaction generation 的 level；Acceptance、retired level 和没有
  exact live source 的 formed pool 不会回流。只有 ACTIVE StructureGeneration
  的 exact protected-Swing assignment 才能把冻结 rank 单向提升为 external；
  legacy tracker-only protected 标签不回填 canonical candidate。非 tick-grid
  formed-pool midpoint 保留为 source fact，公开 target price 使用 lifecycle
  冻结的 near-side tradable anchor。
- [Path protocol](../../configs/path_hypotheses.json) 已表达 continuation、
  deeper retracement、reversal、balance、failed breakout 和 residual unknown
  六个竞争路径，使用有限 log weights 和共同 horizon。没有获准的 likelihood
  artifact 时，两项 Phase 6 evidence rule 的增量均为零。
- Path protocol v1.2 将 `correlation_key` 定义为跨 evidence family 的全局
  dependency-cluster 身份。同一 cluster 的 fitted cross-family contributions
  若没有注册的 joint/history-conditioned 表示会 fail closed；当前 neutral
  source ledger 可以保留多族 provenance，但不会应用 Bayesian multiplier。
  Brain 会拒绝只有 JSON 自报 admission、却没有完整外部 pins/artifacts 的
  likelihood protocol。当前 artifact schema 还没有注册 dependency resolver，
  因而获准测试路径会把整个 competition set 收窄为一个 unresolved cluster：
  重复/跨族 multiplier 无法被默认累加。
- decay 在没有独立 temporal artifact 时保持为零；共同 horizon、显式
  invalidation/expiry 和 zero-probability terminal 已存在。当前没有把统一
  logit subtraction 误称为有效 decay。
- Signal Research 已区分 strict ancestry、constituent-bar composition 和
  `registered_temporal_episode`。Temporal episode 明确不证明 ancestry，也不
  证明因果。
- 通用研究层现在可从显式 `TypedLinkSpec` 投影确定性的 branching episode 和
  typed edge ledger，并保留全部合格支路；冻结的 protocol-v3 r2 仍是历史
  `E1→E5` 线性诊断，未被回写或冒充 branching 实证。
- Scene Graph 的 `PRECEDES` 仍可作为诊断性 temporal-neighborhood edge 查询，
  但 causal/open-thesis closure、通用 action connectivity 和 FAVR eligibility
  使用明确 allowlist，不再让该近邻边形成 action authority。
- 当前 runtime 可以承载 branching path state、DOL ranking 以及未来 fitted
  DOL/no-target layer。没有 exact fitted/admitted、且反向绑定 path
  protocol/model 的 artifact 时，Brain probability map 为空，standalone
  marginalizer 也拒绝返回 probability；ranking 仍只是 development diagnostic
  weight，不是 calibrated posterior，也没有 action authority。

### 2024-06 输入及角色

输入身份由 [data split registry](../../configs/data_splits.json) 绑定：

- causal OHLCV：
  `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet`，
  SHA-256 `84c9ed4d1de379382bdc41e0fe02e3611373ba182cfeebe09e3832d29cbafd7b`；
- raw MBO partition manifest：SHA-256
  `9fbaef324de51cdbc60e13ea07acb7015cce6c409545761d7148ec5f7e8045cd`；
- June BBO artifact：
  `data/processed/mbo_execution_dev_202406_v2_3_clockfix.parquet`，
  SHA-256 `8dc1df19e4b3bcf3c829a2b07c69552a583e667224d89305f78aaae21b73b1ed`。

| Slice | Exact interval (UTC) | Clock census and contract | Permitted role |
|---|---|---|---|
| W1 | `2024-06-02T22:00Z` to `2024-06-07T21:01Z` | 6,899 real OHLCV + one registered synthetic clock / 6,900 MBO minutes; NQM4/13743 | Already revealed Phase 6 development/design evidence |
| W2 | `2024-06-09T22:00Z` to `2024-06-14T21:01Z` | 6,899 real OHLCV + one registered synthetic clock / 6,900 MBO minutes; NQM4/13743 | Already revealed registered extension; development/design only |
| W3 | `2024-06-16T22:00Z` to `2024-06-21T21:01Z` | 6,659 OHLCV / 6,660 BBO; NQM4 to NQU4 roll; Juneteenth removes 240 minutes plus one OHLCV gap | Parked; later roll/holiday robustness slice only |
| W4 | `2024-06-23T22:00Z` to `2024-06-28T21:01Z` | 6,900 OHLCV / 6,900 BBO, exact 1:1 join; NQU4/4358 | Frozen outcome-blind development design; execution bindings pending and run unauthorized |

W1 的 synthetic decision clock 是 `2024-06-07T03:10Z`；W2 是
`2024-06-10T04:14Z`。它们有真实 MBO/BBO 行，但不能成为 sample-eligible
M5 source/control。W3 唯一 BBO-only clock 是 `2024-06-17T03:07Z`。

W4 是本补充研究的首选：五个 Globex sessions 各 1,380 分钟，join key
`(decision_time, symbol, instrument_id)` 无重复、无 orphan、无 null；OHLCV
没有非法 OHLC、负 volume 或 off-tick price；BBO 没有 invalid/crossed book、
future observation、负 depth 或 off-tick quote。六个 all-instrument raw
partitions 共 75,087,497 行，6/6 物理 hashes 与 manifest 相符。这里只完成了 outcome-blind
input preflight。新的
[W4 temporal/branching design](../../experiments/manifests/smc_semantics_v1_2_2024_06_week4_temporal_branching_construct_v1.yaml)
已经冻结，严格 identity validator 在不打开市场数据的前提下通过；但 execution
bindings、W4 mechanism-flow artifact、materialization、semantic replay 和研究结果
均不存在，`execution_authorized=false`。

虽然 OHLCV 总 split 把 2024-06 放在 rolling-OOF 日历段，June prices 已经被
Phase 6 使用；MBO registry 本身也把该月标为 development。因此本补充研究
必须把整个 2024-06 解释为 development / within-month diagnostic。W4 不能称为
独立 validation 或 OOS。

### 已有证据的正确解释

最终的两周 Phase 6 [manifest](../../experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml)
和 [result](../../experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.json)
仍是冻结的 source of truth。其文件与三个 compact ledger 的 hashes、row counts
均通过复核；当前代码变化不会改写历史结果。

| Proposition | Final development result | Permitted interpretation |
|---|---|---|
| Acceptance continuation | matched n=39, mean effect about 0.14522, supported | Acceptance 切出不同的 contemporaneous order-flow regime；不是 future edge |
| ACTIVE Displacement impact | matched n=75, mean effect about 0.01965, supported | ACTIVE Displacement 对应不同的 contemporaneous price-impact regime；不是预测概率 |
| Sweep rejection | matched n=24, underpowered | Promising but unconfirmed；不能称 validated edge |
| MSS flow shift | matched n=2, underpowered | Unknown；不能称 reversal confirmation |
| Historical FVG first-concrete-lifecycle proxy (the frozen Phase 6 label called it a retest response) | matched n=31, mean effect about -0.00012, unsupported | 当前冻结 estimand 不支持；它没有证明无触区 invalidation 是 first retest，也不能事后改变 lifecycle grouping 来挽救结论 |
| Displacement score monotonicity | Week 1 rho about 0.02496; Week 2 rho about 0.02406 | Score 不是 calibrated strength 或 probability |

Phase 7 allowlist 仍严格只有 `acceptance_continuation` 和
`displacement_impact`。这两项允许被 ledgered 为 evidence families，但在没有
path-conditional likelihood artifact 时仍不得产生非零概率更新。

## Problem

1. 旧 Phase 6 不能直接续跑。最终 result 已记录
   `registered_extension_consumed=true` 和 `further_extension_authorized=false`；
   旧 loader 还锁定 NQM4/13743。2026-08-23 审计快照中，当前代码相对最终
   manifest 的 42 项 identity bindings 已有 14 项变化，所以 loader 会按设计
   fail closed；后续代码变化只会要求新的 hash binding，不能改写旧证据。
2. 当前 compact ledgers 不满足跨 hypothesis 独立性。最终 171 个 primary
   pairs 中，3,675 个 inference clocks 有 685 个（18.64%）跨 hypotheses 重用；
   2,170 个 treatment clocks 有 285 个（13.13%）重用；2,749 个 source-M5
   event IDs 有 1,081 个（39.32%）进入 2–5 个 hypotheses。
3. 相同 impulse 可以同时生成 Sweep、Acceptance control、Displacement、MSS
   或 FVG context。现有 hypothesis-local no-reuse 不能授权把这些事件当作独立
   likelihood multipliers。
4. 当前没有 fitted/admitted conditional likelihood、hypothesis-specific
   hazard、prior-reversion artifact、path calibration 或 DOL calibration。
5. 当前 temporal episode 是正确的 research link 边界，W4 design 也已冻结
   relation definitions、lag grid、branch outcomes 和 inference plan；但尚无绑定后的
   relation ledger、lag census、branch labels 或研究结果。
6. 2024-06 只有有限的同月 sessions。它可以做 construct/temporal diagnostic，
   不能同时承担反复定义、拟合、calibration 和独立验证。
7. 冻结 Phase 6 的 FVG unit 是 earliest concrete lifecycle，不是真正由 creation
   zone 与首个 post-creation completed M5 overlap BAR 定义的
   `first_retest_event`。旧结果必须保留原义；foundation v2 已预注册并实现新的
   几何事件定义，但对应 empirical estimator/study 尚未预注册或运行。

## Why it matters

- 无条件相乘相关事件会把一次 price impulse double/triple count，制造过度自信
  的 path 与 DOL weights。
- 把 W4 偷接到旧实验会破坏 manifest authority、合约身份和 frozen-before-run
  原则。
- 用 wall-clock seconds 代替 completed-bar index 会在日常 maintenance、
  Juneteenth 和 roll 周产生错误窗口及 off-by-one。
- formation MBO facts 可以在 semantic `known_at` 时可用；事件之后的 response
  window 只能在窗口结束时可用。把 retrospective post features 回填到根事件
  会产生 leakage。
- Acceptance/Displacement 当前证明的是 construct association，不是 future
  structural outcome。直接训练 BUY/SELL 或称为 calibrated probability 会超过
  现有证据权限。

## Minimal change

1. 不新建第二套 Eye、Brain、MBO replay 或 event store。继续复用 causal reader、
   existing MBO materializer、canonical events、path competition set、
   `correlation_key` 和 `REGISTERED_TEMPORAL_EPISODE`。
   Scene Graph 也只增加关系 allowlist；通用诊断路径仍保留。
2. 在查看 W4 semantic/path outcomes 前，保持已经冻结的 design manifest 不变；
   不把它称为旧 Phase 6 extension。另建 still-closed executable revision，合同固定为
   NQU4/4358，并绑定当前 code、config、OHLCV、BBO、六个 raw partition hashes 和
   materialized mechanism artifact。
3. 已冻结 design 预先固定：relation definitions、candidate lag grid、common horizon、
   competing first outcomes、no-transition/no-target、matching、controls、embargo、
   cluster unit、censoring、primary/secondary metrics、multiple-testing family、
   minimum sample 和 stopping rule。`model_fit_allowed` 与 `probability_claim` 均为
   false。
4. 单独写 temporal relation rows，不改变 source DAG。每行至少记录 source/target
   event IDs、两端 `known_at`、`delta_completed_bars`、`delta_seconds`、direction、
   generation、impulse/correlation group、parent context 和 censor reason。
5. W1–W2 只用于已揭示 design/reference；W3 保持 parked；W4 只运行一次冻结的
   branching/temporal confirmation。Split、bootstrap 和 uncertainty 都按 session/
   day/crossing generation/source-M5 impulse group 隔离，不能按 ledger row 随机切分。
6. W4 先只输出 branching census、lag table、conditional/nested ablation、first
   structural outcome 和 no-transition/no-target census。不要训练 BUY/SELL，也不要
   发布 posterior。
7. 只有 coverage、class counts、independent groups 和 calibration admission 均通过
   预注册门后，才建立另一份 path-likelihood / competing-risk / DOL calibration
   manifest。任何新权重在此前都保持 neutral 或明确命名为 diagnostic score。

Clock contract 保持：OHLCV `ts` 是 bar start，可用时间是 `ts + 1 minute`；MBO
使用 `ts_recv` 并沿用 exact-boundary-inclusive 的 minute ceiling；temporal
`delta_completed_bars` 使用 canonical closed-M5 index，同时另存
`delta_seconds`。Post-response feature 的 availability 是其窗口末端，而不是 root
event 的 `known_at`。

## Tests

本次只读审计已经验证：

- 最终 Phase 6 manifest/result/ledger file hashes 与 ledger row counts；
- 4,390 个 episode IDs 唯一，176 个 pair keys 无重复；
- primary pairs 内 formation clocks 不晚于 `known_at`，post clocks 全部晚于
  `known_at`；
- hypothesis 内 inference-clock 无复用，同时量化了跨 hypothesis 重叠；
- W4 OHLCV/BBO exact-key census、schema/domain/tick/book-clock checks；
- W4 六个 raw MBO partition hashes 和现有 materializer 的只读 preflight：
  development role、6,900 clocks、NQU4/4358、6/6 partitions、causal contract
  selection、无 loader warning。
- W4 design manifest 的 strict identity-only validator：manifest SHA-256
  `89ed4a1eabb705ec80dcd522ca4dafac7a3e8b14660a69471e96d964576dd0c4`，
  `market_data_opened=false`、`execution_authorized=false`。
- Phase 7 read-only readiness checker：完整检查 7,381 行 compact Phase 6 ledgers，
  返回 13 个 blockers、`ready_for_offline_fit=false` 和 `artifacts_written=[]`，
  没有拟合任何 artifact。

本轮代码回归已经覆盖：

- posterior/diagnostic weights normalization；
- equal-logit subtraction invariance，防止把无效统一 decay 当作 belief decay；
- same-impulse Sweep/Displacement/MSS cross-family fitted multipliers fail closed；
- common expiry，以及 non-zero decay 缺少独立 temporal artifact 时 fail closed；
- temporal predecessor 不能变成 ancestry；
- lag-window 上下边界和 same-clock precedence 的 off-by-one；
- typed branching projection 保留并行 sibling、允许缺失非必要支路，并在没有
  显式 relation spec 时 fail closed；
- diagnostic `PRECEDES` 可见但不能满足 causal/open-thesis/action/FAVR gate；
- historical FVG earliest concrete lifecycle 的同钟冲突/不同 payload fail
  closed，且以后 lifecycle 不改写冻结分类；它仍只是历史 proxy，不冒充
  foundation v2 已实现的几何 `first_retest_event`。后者的经验研究尚未运行或
  获得 admission。

正式研究前仍必须冻结并验证：

- hypothesis-specific hazard 或 prior reversion 的定义、artifact 与参数；
- conditional likelihood 的训练集、feature schema、conditioning history 与
  dependency-cluster identity；
- W4 materialized mechanism artifact 的 6,900-row exact census、MBO flow QC、
  event lineage、checkpoint/replay determinism 和 result non-overwrite。

## Research consequence

- 当前已完成的是安全的 architecture/runtime boundary：candidate-only Eye、
  competing paths、no-target、additive log-update interface、correlation identity、
  explicit expiry 和 temporal-not-ancestry research mode。
- 当前未完成的是 empirical model：conditional likelihood、effective decay/hazard、
  calibrated path probability、calibrated DOL probability、stable branching
  relationship、future structural edge 和 action authority均不存在。
- 真正的 FVG first-retest empirical treatment/study 仍未完成。其未来数据集必须
  使用 foundation-v2 creation-time zone 与首个 post-creation real completed M5
  overlap BAR，按已冻结边界处理 no-touch/invalidation/censoring，而不是复用历史
  lifecycle proxy；这不否定当前几何事件实现已经完成。
- W4 数据输入通过预检且 design 已冻结，但 executable bindings、W4 mechanism
  feature、semantic temporal diagnostic、path/DOL fitting 均未运行。
- sealed MBO OOS 未打开，final OOS 仍关闭。2024-06 的任何新结果最多是
  development evidence；它不能自行授权 Phase 7 非零 updates、Execution Research、
  Trade Intent 或 live trading。

下一道门只有一个：保持冻结的 W4 design 不变，先绑定 still-closed executable
revision，再物化 Week-4 flow artifact 并运行一次 bounded diagnostic。研究结果决定哪些 relation
可以进入以后单独冻结的 calibration study；architecture 不提前决定答案。

因此，相对完整目标提示词，本轮总状态仍是 **partial**：安全的架构边界已建立，
但 empirical fit、Execution Research、operational Shadow Live、rolling OOF 和 sealed
OOS 均未完成。
