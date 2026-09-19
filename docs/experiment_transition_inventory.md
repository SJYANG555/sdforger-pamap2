# Experiment Transition Inventory for Joint Report

本文整理当前仓库中已经出现的 PAMAP2 / MM-Fit SDForger-style 实验线，目的不是复述所有结果，而是帮助三人联合报告做选材、过渡和章节放置。

核心判断：

- 最终报告不要按“谁做了什么”组织，而要按大纲里的 RQ 组织。
- 你的工作最适合作为 RQ2 的主体证据：conditioning 旋钮、channel 选择、统计 prompt、epoch / ICA / filtering trade-off。
- 早期 smoke、5-class、GPT-2 / Gemma / Llama 对比和 MM-Fit 扩展可以保留，但多数应放在 Setup、Ablation、Appendix 或 Limitations，不宜全部进入主结果。
- 所有跨数据集、跨 split、跨 normalization 的数值都只能作为 within-protocol evidence，不能直接做绝对优劣排序。

## 1. 与联合大纲的对应关系

| 报告问题 | 你的实验能提供什么 | 建议放置 |
|---|---|---|
| RQ1: conditioning 在哪里失败 | 5-class 与 2-act 中的 class imbalance、低 synthetic-only utility、periodic/stat prompt 不稳定、channel-dependent collapse | Results 5.1，作为辅助证据 |
| RQ2: conditioning 旋钮有哪些、trade-off 如何 | channel 选择、window stats prompt、epoch、ICA dim、window size、post-hoc filtering | Results 5.2，作为主贡献 |
| RQ3: 什么打破 collapse | 你的实验没有直接证明 mode-aware 解法，但能提供 “stats/channel/ICA/filtering 仍不足” 的负控背景 | Results 5.3 前的 bridge 或 Discussion |
| Unified framework | 已有 PAMAP2 / MM-Fit builder、FastICA latent z、text serialization、LLM generation、decode/evaluation pipeline | Section 3 Framework |
| Experimental setup | 多个 dataset / channel / model / conditioning 设计矩阵 | Section 4 Setup |
| Limitations | split 不完全统一、指标不能跨 protocol 比、部分探索没有统一 summary、MM-Fit 和 PAMAP2 的 classifier protocol 不一致 | Section 7 Limitations |

## 2. 实验演进顺序

### Stage 0: Smoke test and pipeline validation

代表文件：

- `config/pamap2_sdforger_gpt2_smoke.yaml`
- `outputs/generated/gpt2_smoke/generation_summary.json`
- `docs/SMOKE_TEST.md`

目的：

- 验证 dataset build、training、checkpoint、generation、parser、reconstruction 能跑通。
- 证明 Triton GPU 路线和 Slurm 工作流可用。

结果与判断：

- `gpt2_smoke` 只生成 1 个有效 embedding，主要原因是 dedup / norm filter 后严重收缩。
- 工程上有效，科学结论上不应作为主结果。

报告建议：

- 不放 Results 主文。
- 可在 Method / Reproducibility 中一句话说明 “we first validated the full pipeline on a smoke configuration”。
- 若篇幅紧，完全移到 Appendix 或不写。

### Stage 1: Formal PAMAP2 5-class baseline

代表文件：

- `config/pamap2_sdforger_gpt2.yaml`
- `config/pamap2_sdforger_gemma2_2b.yaml`
- `outputs/generated/gpt2/generation_summary.json`
- `outputs/evaluation/utility/gpt2/utility_metrics.csv`
- `outputs/evaluation/utility/gemma/utility_metrics.csv`

目的：

- 从 smoke 进入正式 PAMAP2 5-class setting。
- 建立 “window -> FastICA latent -> text -> LLM -> latent/window” 的主 pipeline。

关键结果：

- `gpt2`: 256 valid synthetic windows；synthetic-only accuracy 约 0.319；real-plus-synthetic 基本不超过 real-only。
- `gemma`: 227 valid synthetic windows；synthetic-only accuracy 约 0.400；real-plus-synthetic 略低于 real-only。

是否有效：

- 工程有效：正式 baseline 跑通。
- 论文有效性有限：5-class 下合成数据 utility 弱，适合说明 naive SDForger-style pipeline 不自动带来强下游价值。

报告建议：

- 放在 RQ1 或 Setup 的 baseline paragraph。
- 不要写成 “GPT-2 vs Gemma 谁更好”，因为模型、训练、prompt、filter 都可能 confound。
- 可作为主线中的 early negative evidence：LLM capacity alone is not enough。

### Stage 2: 5-class v2 / compact / model comparison

代表文件：

- `config/pamap2_sdforger_gpt2_5class_v2*.yaml`
- `config/pamap2_sdforger_gemma2_2b_5class_v2*.yaml`
- `config/pamap2_sdforger_llama32_3b_5class_v2*.yaml`
- `outputs/evaluation/extended/extended_model_summary.csv`
- `outputs/evaluation/extended/extended_model_summary_compact.csv`
- `outputs/evaluation/knn_condition_5class_v2_compact/knn_condition_metrics.csv`

目的：

- 扩展到 GPT-2 / Gemma / Llama 3.2 3B。
- 比较 full、hand、chest、hand+chest 等 sensor subset。
- 引入更完整 extended metrics：fidelity、diversity、condition consistency。

关键结果：

- `gpt2_5class_v2` 和 compact 原始生成曾出现 0 valid，后来 `gpt2_5class_v2_compact_ckpt1300_reparsed` 得到 835 valid，synthetic condition consistency accuracy 约 0.695。
- `gemma_5class_v2_hand_chest` condition consistency macro-F1 约 0.666，是 5-class v2 中较强设置。
- `llama32_3b_5class_v2` condition consistency macro-F1 约 0.583。
- KNN condition check 中 real_test accuracy 约 0.778，GPT-2 synthetic 约 0.660，Gemma synthetic 约 0.635。

是否有效：

- 对工程与探索有效：证明 pipeline 可以跨模型、跨 sensor subset 跑。
- 对最终主文要谨慎：指标较多且 protocol 混杂，不适合直接当主结论。

报告建议：

- 可在 Section 4 Design Matrix 中列出。
- 可在 RQ1 中用一句话支持 “larger / different LLM did not remove controllability and fidelity gaps by itself”。
- 详细表放 Appendix。
- 不建议在主文做 LLM ranking。

### Stage 3: Periodic prompt / cadence-oriented exploration

代表文件：

- `config/pamap2_sdforger_gpt2_periodic.yaml`
- `config/pamap2_sdforger_gemma2_2b_5class_v2_periodic.yaml`
- `outputs/generated/gemma_5class_v2_periodic/generation_summary.json`
- `outputs/evaluation/similarity/gemma_5class_v2_periodic/similarity_metrics.csv`
- `outputs/evaluation/utility/gemma_5class_v2_periodic/utility_metrics.csv`

目的：

- 尝试通过 periodic / cadence-oriented prompt 增强时间结构。

关键结果：

- `gemma_5class_v2_periodic` 生成 383 valid synthetic windows。
- synthetic-only accuracy 约 0.227；real-plus-synthetic 略高于该 protocol 的 real-only，但合成数据本身分类 utility 很弱。

是否有效：

- 作为探索有效。
- 作为主结论证据不足：没有进入统一 summary 表，也没有形成稳定对照链。

报告建议：

- 不建议放主文。
- 可在 RQ1 或 Limitations 中一句话提及：more descriptive periodic prompting alone did not produce reliable synthetic utility。
- 如要放图或表，应放 Appendix。

### Stage 4: Switch to PAMAP2 two-activity walking/running

代表文件：

- `config/pamap2_sdforger_gemma2_2b_2act_*.yaml`
- `outputs/evaluation/summary/gemma_2act_13run_summary.csv`
- `docs/pamap2_2act_experiment_summary.md`

目的：

- 将 5-class 收敛到 walking / running，减少类别歧义。
- 更清楚观察周期步态中的 channel、prompt、epoch trade-off。

关键结果：

- Seven variants: full, hand, chest, hand_acc, chest_acc, hand_gyro, chest_gyro。
- `hand` 给出最强 utility-oriented result：valid 226/305，synthetic-only accuracy 约 0.921。
- `chest_gyro` 给出最强 similarity-oriented result：ED/real 约 0.72，DTW/real 约 1.13。
- 不同 channel 的最优目标不同：hand 更像 utility channel，chest_gyro 更像 fidelity channel。

是否有效：

- 非常适合最终报告。
- 它是 RQ2 的第一组核心证据：conditioning 不只是文本，sensor/channel 选择也是有效旋钮。

报告建议：

- 放 Results 5.2。
- 主文可放一张 condensed table：variant、valid rate、ED/real、DTW/real、synthetic-only acc、real+synthetic acc。
- 不要过度声称 “hand overall best”；应写成 “hand optimizes utility, chest_gyro optimizes similarity”。

### Stage 5: Epoch scaling without statistical prompt

代表文件：

- `outputs/evaluation/summary/gemma_2act_all_baseline_epoch_stats_comparison.csv`
- `docs/pamap2_2act_experiment_summary.md`

目的：

- 测试 5 / 10 / 20 epoch 是否能靠更长训练解决 generation quality。

关键结果：

- `hand`: synthetic-only accuracy 从 5 epoch 的 0.921 下降到 10 epoch 的 0.820，再到 20 epoch 的 0.377。
- `chest_gyro`: 10 epoch similarity 最好，但 synthetic-only accuracy 仍约 0.370。
- `chest`: 20 epoch utility 提升到约 0.626，但 validity 和 similarity 不稳定。

是否有效：

- 作为 negative ablation 很有价值。
- 说明 “more training” 不是可靠解法。

报告建议：

- 放 Results 5.2 的 ablation paragraph。
- 主文只保留一个小表或一张 line plot。
- 结论写成：longer training changes the trade-off but does not dominate the 5-epoch baseline。

### Stage 6: Window-level statistical prompt

代表文件：

- `config/pamap2_sdforger_gemma2_2b_2act_*_stats*.yaml`
- `outputs/evaluation/summary/gemma_2act_stats_epoch_summary.csv`
- `outputs/evaluation/summary/gemma_2act_all_baseline_epoch_stats_comparison.csv`

目的：

- 将 window-level mean / std / min / max 写入训练文本 prompt。
- 测试统计 prompt 是否能增强条件化。

关键结果：

- `hand + stats, 5 epochs`: valid 301/305，synthetic-only accuracy 约 0.928，real+synthetic accuracy 约 0.970。
- `chest_gyro + stats, 5 epochs`: ED/real 约 0.466，DTW/real 约 0.740，是强 similarity result。
- `chest_gyro + stats, 20 epochs`: synthetic-only accuracy 约 0.564，但 similarity 不如 5 epoch。
- `chest + stats` 和 `chest_acc + stats` 提高 validity，但 utility 不一定改善。

是否有效：

- 非常适合最终报告。
- 这是 RQ2 的第二组核心证据：stronger textual conditioning helps, but the effect is channel- and metric-dependent。

报告建议：

- 放 Results 5.2 主表。
- 写作重点不是 “stats prompt 全面胜利”，而是 “stats prompt exposes useful knobs but does not eliminate trade-off”。
- 和 Stage 4 并列，形成 channel x prompt 的设计空间。

### Stage 7: ICA dimensionality sweep

代表文件：

- `config/pamap2_sdforger_gemma2_2b_2act_*_stats_ica{3,12,18,24}_ep*.yaml`
- `outputs/evaluation/summary/gemma_2act_ica3_6_12_18_stats_epoch_comparison.csv`

目的：

- 测试 latent z 的 FastICA dimension 是否影响 generation / utility / similarity。

关键结果：

- 在 `chest_gyro + stats` 中，ICA12 明显提高 utility：5 epoch synthetic-only accuracy 约 0.898，10 epoch 约 0.964。
- ICA3 / ICA6 / ICA18 并不形成单调优势。
- similarity 与 utility 仍有 trade-off。

是否有效：

- 适合作为 RQ2/RQ1 的桥接证据。
- 它支持 “representation capacity matters, but more dimensions alone is not sufficient”。

报告建议：

- 主文中可放一个小型 ablation table，重点展示 ICA3 / 6 / 12 / 18 的非单调性。
- 不建议写成 “ICA12 is universally best”，只能说在该 channel/prompt/protocol 下最有效。

### Stage 8: Window size 128 + stride 64 + ICA12 + stats source comparison

代表文件：

- `config/pamap2_sdforger_gemma2_2b_2act_*_stats_ica12_win128_ep*.yaml`
- `config/pamap2_sdforger_gemma2_2b_2act_*_icastats_ica12_win128_ep*.yaml`
- `outputs/evaluation/summary/gemma_2act_stats_ica12_win128_summary.csv`
- `outputs/evaluation/summary/gemma_2act_icastats_ica12_win128_summary_ad_hoc.csv`
- `outputs/evaluation/summary/gemma_2act_icastats_vs_rawstats_ica12_win128_ad_hoc.csv`

目的：

- 用 shorter windows 增加训练/test windows 数量。
- 比较 raw-window stats 和 embedding-channel stats 两种 prompt source。

关键结果：

- `hand_acc + rawstats + ICA12 + win128`: valid rate 可达 0.545 / 0.976，synthetic-only utility 约 0.926 / 0.745。
- `chest_gyro + rawstats + ICA12 + win128`: valid rate 0.992 / 0.850，similarity 很强，但 synthetic-only utility 约 0.503 / 0.357。
- `chest_gyro + icastats + ICA12 + win128`: synthetic-only utility 约 0.955 / 0.959，但 similarity 比 rawstats 差。
- `hand + icastats` validity 很低，ep10 only 12 valid，不能作为稳定主结果。

是否有效：

- 很适合做“过渡实验”：说明 prompt 中 stats 的来源会改变 validity / utility / similarity。
- 但 ad hoc 表较多，protocol 更复杂，不适合压成主结论。

报告建议：

- 放 RQ2 的 secondary ablation 或 Appendix。
- 主文最多保留一句：stats source and window granularity change the trade-off; no single setting dominates all metrics。
- 若报告篇幅允许，放一张 appendix table。

### Stage 9: Post-hoc consistency filtering

代表文件：

- `scripts/apply_stats_consistency_filters.py`
- `scripts/refilter_generated_candidates.py`
- `outputs/evaluation/summary/gemma_2act_stats_ica12_win128_p2575_filtered_summary.csv`

目的：

- 用统计一致性分数过滤生成样本，测试 QC 是否能改善结果。

关键结果：

- p25 / p50 / p75 filtering 可以控制保留比例。
- 对部分 setting real+synthetic accuracy 有帮助，但会显著减少样本数。
- filtering 并没有从根本上解决 similarity / utility trade-off。

是否有效：

- 作为 QC / post-hoc constraint 有价值。
- 不适合作为核心方法贡献。

报告建议：

- 放 RQ2 的 “post-hoc constraint” 小段或 Appendix。
- 写作上要强调它是 filtering/QC，不是 generation 本身的改进。

### Stage 10: MM-Fit dataset extension

代表文件：

- `mmfit_forger/`
- `scripts/build_mmfit_sdforger_dataset.py`
- `scripts/generate_mmfit_synthetic.py`
- `config/mmfit_sdforger_{gpt2,gemma2_2b,llama32_3b}_5class.yaml`
- `outputs/evaluation/extended/extended_model_summary.csv`
- `outputs/evaluation/utility/mmfit_*_5class/utility_metrics.csv`
- `outputs/evaluation/similarity/mmfit_*_5class/similarity_metrics.csv`

目的：

- 将同一 SDForger-style pipeline 迁移到 MM-Fit。
- 测试第二数据集上的 model / generation behavior。

关键结果：

- MM-Fit dataset: 5 activities, 12 watch channels, window size 250, train/val/test by workout session。
- `mmfit_gemma2_2b_5class`: 369 valid synthetic windows。
- `mmfit_gpt2_5class`: 116 valid synthetic windows，且缺少 jumping_jacks。
- `mmfit_llama32_3b_5class`: 391 valid synthetic windows。
- Utility script 中 `mmfit_gpt2_5class` synthetic-only accuracy 约 0.783，但 valid count 低；extended condition consistency 中 Gemma 约 0.707。

是否有效：

- 工程上很有价值：证明 pipeline 可迁移到第二数据集。
- 科学上要谨慎：MM-Fit 当前更像 external validity / dataset extension，不应和 PAMAP2 绝对数值硬比。

报告建议：

- 放 Section 4 Design Matrix。
- Results 主文中可作为 “second dataset sanity check”。
- 详细指标放 Appendix。
- 如果要配合联合大纲，应与 Yimeng 的 MM-Fit mode-aware 结果区分：你的 MM-Fit 是 FastICA + LLM baseline / extension，不是 mode-aware 解法。

## 3. 新旧改进的有效性总结

| 改进 | 相对旧版本 | 是否有效 | 最适合的报告角色 |
|---|---|---|---|
| Smoke -> formal dataset | 从工程验证到正式 subject split | 工程有效，论文结果弱 | Method / Appendix |
| 5-class -> 2-act walking/running | 减少类别歧义，聚焦步态 | 有效，主线更清楚 | Results RQ2 |
| Full sensors -> channel subsets | 把 sensor choice 变成实验轴 | 有效，但目标不同 | Results RQ2 主表 |
| More epochs | 试图靠训练时长改善质量 | 不稳定，不是解法 | RQ2 negative ablation |
| Window stats prompt | 增强文本条件化 | 有效，但 channel-dependent | RQ2 核心结果 |
| Higher ICA dimension | 增强 latent capacity | 局部有效，非单调 | RQ2/RQ1 bridge |
| win128/stride64 | 增加样本数、缩短 temporal window | 有探索价值，证据复杂 | Appendix / secondary ablation |
| rawstats vs icastats | 比较条件统计来源 | 有启发，但 ad hoc | Appendix 或 Discussion |
| post-hoc filtering | 生成后 QC | 局部改善，不是根治 | RQ2 post-hoc constraint |
| MM-Fit extension | 第二数据集迁移 | 工程有效，科学上需统一 protocol | Setup / Appendix |
| periodic prompt | 尝试 cadence prompt | 证据不足 | Limitations / Appendix |

## 4. 最终报告建议采用哪些实验

### 主文必须采用

1. PAMAP2 2-act seven channel baseline。
   - 作用：证明 channel 是重要 conditioning / representation 旋钮。
   - 放置：Results RQ2 第一小节。

2. PAMAP2 2-act stats prompt + epoch scaling。
   - 作用：证明 stronger prompt helps but does not remove trade-off。
   - 放置：Results RQ2 第二小节。

3. Real-vs-real similarity baseline。
   - 作用：避免孤立解释 ED / DTW。
   - 放置：Setup 或 Results table caption。

4. Epoch negative ablation。
   - 作用：说明 “train longer” 不是可靠解释。
   - 放置：Results RQ2 或 Discussion。

### 主文可以采用，但要压缩

1. 5-class GPT-2 / Gemma / Llama comparison。
   - 作用：说明换模型本身不自动解决问题。
   - 放置：RQ1 或 Appendix summary。

2. ICA dimension sweep。
   - 作用：连接 “latent capacity” 与 “conditioning controllability”。
   - 放置：RQ2 ablation / Discussion。

3. MM-Fit FastICA baseline。
   - 作用：第二数据集迁移证据。
   - 放置：Setup + Appendix；主文一句总结。

### 不建议放主文，只放附录或不放

1. Smoke test。
   - 原因：工程验证，不是科学结果。

2. Periodic prompt。
   - 原因：证据链不完整，结果不强。

3. 所有 p25/p50/p75 filtering 细表。
   - 原因：太细，容易分散主线。

4. 所有 win128 rawstats/icastats ad hoc 表。
   - 原因：结果很有趣，但 protocol 复杂；除非报告空间足够，否则放 Appendix。

## 5. 推荐写作顺序

可按下面方式把你的部分接进联合报告：

1. Framework:
   - “All experiments instantiate the same pipeline: sensor window x -> compact latent z -> serialized prompt with condition c -> LLM -> generated latent z' -> reconstructed window x'.”

2. Setup:
   - 放设计矩阵：PAMAP2 5-class、PAMAP2 2-act、MM-Fit 5-class。
   - 明确声明：metrics are compared within protocol, not across datasets or normalization schemes。

3. RQ1:
   - 用早期 5-class 和 periodic/stats 不稳定现象说明 naive conditioning 容易 collapse。

4. RQ2:
   - 主讲你的 PAMAP2 2-act channel / stats / epoch / ICA experiments。
   - 结论写成 trade-off，而不是单一最优。

5. RQ3:
   - 把你的结果作为 mode-aware 解法前的负控背景：
     “channel, stats, and ICA capacity help expose useful knobs, but they do not fully solve collapse; this motivates explicit mode-aware conditioning in the third thread.”

6. Discussion:
   - 回到联合 thesis：conditioning strength depends on what the latent representation exposes and how explicitly the condition is provided。

## 6. 当前最需要补齐的共享材料

为了后续生成完整可共享项目内容，建议优先补齐：

1. 一张统一设计矩阵表：
   - dataset, activities, window size, channels, latent dim, model, conditioning, evaluation。

2. 一张主结果表：
   - 只放 PAMAP2 2-act 的 selected variants：hand, chest_gyro, hand+stats, chest_gyro+stats, best ICA12 setting。

3. 一张 “report inclusion” 表：
   - main text / appendix / omit。

4. 一个 source-of-truth summary：
   - 以 `outputs/evaluation/summary/*.csv` 为准，避免从零散日志取数。

5. 统一术语：
   - embedding -> latent z
   - prompt stats -> conditioning c
   - filtering -> post-hoc constraint
   - synthetic-only -> train-on-synthetic/test-on-real utility, unless otherwise specified
   - condition consistency -> classifier trained on real and evaluated on generated labels, not the same as TSTR utility

## 7. 一句话版本

你的实验线最适合这样进入联合报告：

> Starting from a working SDForger-style PAMAP2 pipeline, we progressively tested whether controllability improves by changing the sensor channel, prompt statistics, training length, latent dimensionality, window granularity, and post-hoc filtering. The results show that conditioning can be strengthened, but improvements are metric- and representation-dependent: hand channels improve downstream utility, chest gyroscope improves similarity, statistics improve validity and some utility, while longer training and filtering do not reliably remove collapse. These findings support the joint paper's design-space thesis and motivate explicit mode-aware conditioning as the next step.

