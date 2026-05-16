# AGENTS.md

这是本仓库的 Codex 执行手册。仓库目录名虽然是 `ChatTS`，但当前所有实验主线不是原始 ChatTS demo，而是基于 forging / SDForger-style 思路改造出的 `PAMAP2 + time-series synthetic generation` 工程。后续协作应围绕正式 GPT-2 baseline 和之后的 Gemma 实验推进，不要再把重点放在 smoke test 大修上。

## 1. 用户的实际工作方式

- Codex App 已配置远程 SSH host：`triton`。
- Windows SSH 配置位置：`C:\Users\ysj\.ssh\config`。
- 当前 SSH host 配置：

```sshconfig
Host triton
    HostName triton.aalto.fi
    User yangs9
    IdentityFile C:\Users\ysj\.ssh\codex_triton_nopass
    IdentitiesOnly yes
```

- Codex App 可通过 SSH 直接连接 Triton 并修改远程项目文件：`/scratch/work/yangs9/ChatTS`。
- 远程 Triton 已安装 Codex CLI，并已把 `codex` 加入远程 shell 的 PATH。
- 允许 Codex 在远程项目中直接上传、下载、读取、创建和修改仓库文件。
- 训练、生成、评估、完整 dataset build 等重任务仍由用户手动通过 Slurm 提交。
- Codex 不要在 login 节点直接运行重任务。
- 不要声称“已经完成 Triton 训练/生成/评估”，除非用户明确贴出 Slurm 日志或 Codex 确实只做了轻量检查。

以后给运行方案时，必须明确区分：

1. 需要修改的仓库文件。
2. 当前是在本地仓库修改，还是在 Triton 远程项目中直接修改。
3. 若是本地修改，才说明需要用户重新上传到 Triton 的文件。
4. 用户需要在 Triton 终端执行的命令。
5. 如何检查是否成功。
6. 该步骤使用 CPU 还是 GPU。

不要给脱离当前环境的泛泛 HPC 建议。命令必须尽量可直接复制执行。

## 2. 项目定位

当前项目是 `PAMAP2 + SDForger-style + GPT-2 / Gemma` 的时间序列生成工程。

核心流程：

1. 从 PAMAP2 protocol 数据构建 5 类活动窗口。
2. 将 window 编码为 SDForger-style 文本/embedding。
3. 微调 GPT-2 baseline 或 Gemma 2 2B LoRA。
4. 生成 synthetic embedding / synthetic window。
5. 用 similarity metrics 和 HAR utility 评估合成数据。
6. 绘制训练、生成、评估图表。

默认活动类别：

- `walking`
- `running`
- `cycling`
- `sitting`
- `standing`

默认窗口与通道：

- `window_size=256`
- `stride=128`
- `18` 个传感器通道

## 3. 关键目录

- `pamap2_forger/`：当前实验主线的核心代码。
  - `dataset.py`：构建 PAMAP2 split 和训练数据。
  - `embeddings.py`：window 到 embedding 的转换。
  - `text.py`：embedding 与文本模板互转。
  - `trainer.py`：Hugging Face causal LM 训练，支持 GPT-2 和 PEFT/Gemma。
  - `generate.py`：生成、解析、过滤、重构 synthetic windows。
  - `metrics.py`：similarity 和 HAR utility 指标。
  - `visualization.py`：训练曲线、窗口图、embedding scatter 和指标图。
- `scripts/`：命令行入口。
- `config/`：实验配置。
  - `config/pamap2_sdforger_gpt2.yaml`：正式 GPT-2 baseline。
  - `config/pamap2_sdforger_gpt2_smoke.yaml`：smoke test，仅用于工程验证。
  - `config/pamap2_sdforger_gemma2_2b.yaml`：Gemma 2 2B + LoRA。
- `slurm/`：Triton 作业脚本。
- `artifacts/`：预处理和 dataset builder 输出。
- `outputs/`：训练、生成、评估和绘图输出。
- `gpt2baseline/`：已归档的一次 GPT-2 baseline 结果。
- `fms-dgt-main/`：IBM fms-dgt / SDForger 参考代码副本。
- `README.md`：原始 ChatTS 说明，不能作为当前实验主线的唯一依据。

## 4. Windows 到 Triton 的实际登录流程

用户在 Windows 上操作。给命令时优先从 SSH 登录开始写；如果省略登录步骤，必须明确写“假设已登录 Triton”。

### A. 先连接 Aalto VPN

1. 打开 Cisco Secure Client。
2. 使用正确的 Aalto 账号登录。
3. VPN 连上后再打开命令行。

### B. 打开 Windows Terminal / CMD / Git Bash

常用登录命令：

```bash
ssh yangs9@triton.aalto.fi
```

### C. 登录后进入项目目录

```bash
cd /scratch/work/yangs9/ChatTS
```

### D. 激活 Triton 环境

```bash
module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/work/yangs9/ChatTS/.conda-env
```

### E. 必要时设置运行库路径

```bash
export LD_LIBRARY_PATH="/scratch/work/yangs9/ChatTS/.conda-env/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="/scratch/work/yangs9/ChatTS/.conda-env/lib/libstdc++.so.6"
```

原因：

- 之前遇到过 `libstdc++` / `CXXABI_1.3.15` 问题。
- `scipy` / `transformers` 导入问题通过上述方式解决过。

注意：

- `LD_PRELOAD` 可能让 `nano` 等交互程序崩溃。
- 如果需要在 Triton 上临时创建或修改脚本，优先给用户这种非交互写法：

```bash
cat > path/to/file.sh <<'EOF'
#!/bin/bash
echo "example"
EOF
```

- 不要默认建议用户使用 `nano`。

## 5. Triton 环境与固定路径

Triton 项目根目录：

```text
/scratch/work/yangs9/ChatTS
```

conda/mamba 环境路径：

```text
/scratch/work/yangs9/ChatTS/.conda-env
```

已验证可用的环境激活方式：

```bash
module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/work/yangs9/ChatTS/.conda-env
```

运行 Python 训练/生成/评估前，若遇到 ABI 或 import 问题，加入：

```bash
export LD_LIBRARY_PATH="/scratch/work/yangs9/ChatTS/.conda-env/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="/scratch/work/yangs9/ChatTS/.conda-env/lib/libstdc++.so.6"
```

## 6. CPU 与 GPU 作业区别

CPU 作业：

- 不显式写 `account` / `partition` 时，Triton 通常会自动分配默认 CPU batch 分区。
- 之前成功自动分配到过：`batch-hsw`、`batch-bdw`、`batch-csl`、`batch-skl`、`batch-milan`。
- CPU 适合：dataset build、similarity evaluation、utility evaluation、plotting。

GPU 作业：

- 想用 GPU，必须显式请求 GPU 资源，例如：

```bash
#SBATCH --gpus=1
```

- 不写 `--gpus=1` 不会自动给 GPU。
- “自动分配 partition”不等于“自动分配 GPU”。
- 已经成功通过“不显式写 account/partition，但显式写 `--gpus=1`”提交 GPU 作业。
- 实际曾分配到：`gpu-v100-16g` / `dgx2`。
- `nvidia-smi` 已确认可见 `Tesla V100-SXM2-16GB`。
- GPU 适合：GPT-2/Gemma training、generation。

以后给 Slurm 脚本时必须明确说明：

- 这是 CPU 脚本还是 GPU 脚本。
- GPU 请求写在哪一行。
- 是否需要 `nvidia-smi` 验证。

## 7. 已验证的 Slurm 使用规则

提交作业必须用：

```bash
sbatch slurm/<script>.slurm
```

不要用 `bash slurm/<script>.slurm` 直接执行并误以为 `#SBATCH` 会生效。`#SBATCH` 只有通过 `sbatch` 提交时才由 Slurm 解析。

常用检查命令：

```bash
squeue -u $USER
scontrol show job JOBID
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed
tail -f outputs/logs/xxx.out
cat outputs/logs/xxx.err
```

以后给运行步骤时必须包含：

- 如何提交。
- 如何看排队/运行状态。
- 如何看日志。
- 如何检查最终结果文件。

## 8. 当前阶段结论

Smoke 阶段结论：

- smoke test 已完成工程验证使命。
- 已验证训练链路通。
- 已验证 checkpoint 保存通。
- 已验证 generation 通。
- 已验证 parser 通。
- 已验证 reconstruction 通。
- 已验证 Triton GPU 路线通。
- 最新 smoke generation 不再为空，但存在 duplicate / mode collapse。
- smoke 可以结束，不应继续大修。

当前主线：

- 切换到正式 GPT-2 baseline。
- 优先保证正式 dataset、training、generation、evaluation、plotting 可复现。
- Gemma 2 2B LoRA 是后续扩展，不应阻塞 GPT-2 baseline 落地。

## 9. 正式 GPT-2 baseline 推荐顺序

以下命令假设已登录 Triton，并已执行：

```bash
cd /scratch/work/yangs9/ChatTS
module load scicomp-python-env
eval "$(mamba shell hook --shell bash)"
mamba activate /scratch/work/yangs9/ChatTS/.conda-env
export LD_LIBRARY_PATH="/scratch/work/yangs9/ChatTS/.conda-env/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="/scratch/work/yangs9/ChatTS/.conda-env/lib/libstdc++.so.6"
```

### 1. 构建正式 dataset（CPU）

```bash
python scripts/build_pamap2_sdforger_dataset.py --config config/pamap2_sdforger_gpt2.yaml
```

Slurm 脚本：

```bash
sbatch slurm/run_build_dataset.slurm
```

检查：

```bash
ls artifacts/pamap2_sdforger_dataset
cat artifacts/pamap2_sdforger_dataset/dataset_manifest.json
```

### 2. 训练正式 GPT-2 baseline（GPU）

```bash
python scripts/train_pamap2_lm.py --config config/pamap2_sdforger_gpt2.yaml
```

Slurm 脚本优先使用：

```bash
sbatch slurm/run_train_gpt2_gpu_ready.slurm
```

该类脚本必须包含 GPU 请求：

```bash
#SBATCH --gpus=1
```

检查：

```bash
squeue -u $USER
tail -f outputs/logs/gpt2_gpu-<JOBID>.out
cat outputs/logs/gpt2_gpu-<JOBID>.err
ls outputs/checkpoints/gpt2/best
cat outputs/checkpoints/gpt2/training_summary.json
```

### 3. Generation（GPU）

```bash
python scripts/generate_pamap2_synthetic.py --config config/pamap2_sdforger_gpt2.yaml --model-path outputs/checkpoints/gpt2/best --output-dir outputs/generated/gpt2
```

Slurm 脚本：

```bash
sbatch slurm/run_generate.slurm
```

检查：

```bash
ls outputs/generated/gpt2
cat outputs/generated/gpt2/generation_summary.json
```

关键结果文件：

- `outputs/generated/gpt2/generated_embeddings.csv`
- `outputs/generated/gpt2/generated_windows.npy`
- `outputs/generated/gpt2/generation_summary.json`

### 4. Similarity evaluation（CPU）

```bash
python scripts/evaluate_similarity.py --config config/pamap2_sdforger_gpt2.yaml --real-windows artifacts/pamap2_sdforger_dataset/test_windows.npy --real-metadata artifacts/pamap2_sdforger_dataset/test_metadata.csv --synthetic-windows outputs/generated/gpt2/generated_windows.npy --synthetic-metadata outputs/generated/gpt2/generated_embeddings.csv --output-dir outputs/evaluation/similarity/gpt2
```

检查：

```bash
ls outputs/evaluation/similarity/gpt2
cat outputs/evaluation/similarity/gpt2/similarity_summary.json
```

### 5. Utility evaluation（CPU）

```bash
python scripts/evaluate_utility.py --config config/pamap2_sdforger_gpt2.yaml --real-train-windows artifacts/pamap2_sdforger_dataset/train_windows.npy --real-train-metadata artifacts/pamap2_sdforger_dataset/train_metadata.csv --real-test-windows artifacts/pamap2_sdforger_dataset/test_windows.npy --real-test-metadata artifacts/pamap2_sdforger_dataset/test_metadata.csv --synthetic-windows outputs/generated/gpt2/generated_windows.npy --synthetic-metadata outputs/generated/gpt2/generated_embeddings.csv --output-dir outputs/evaluation/utility/gpt2
```

检查：

```bash
ls outputs/evaluation/utility/gpt2
cat outputs/evaluation/utility/gpt2/utility_summary.json
```

### 6. Plotting（CPU）

```bash
python scripts/make_plots.py --config config/pamap2_sdforger_gpt2.yaml --training-log-csv outputs/checkpoints/gpt2/training_log_history.csv --real-windows artifacts/pamap2_sdforger_dataset/test_windows.npy --real-metadata artifacts/pamap2_sdforger_dataset/test_metadata.csv --synthetic-windows outputs/generated/gpt2/generated_windows.npy --synthetic-metadata outputs/generated/gpt2/generated_embeddings.csv --real-embeddings artifacts/pamap2_sdforger_dataset/test_embeddings.csv --synthetic-embeddings outputs/generated/gpt2/generated_embeddings.csv --output-dir outputs/plots/gpt2
```

检查：

```bash
ls outputs/plots/gpt2
```

### 7. Evaluation Slurm 脚本

仓库已有综合评估脚本：

```bash
sbatch slurm/run_evaluate.slurm
```

它会运行：

- similarity evaluation
- utility evaluation
- plotting

## 10. 结果文件速查

正式 GPT-2 baseline 关键输出：

- Dataset：`artifacts/pamap2_sdforger_dataset/`
- Dataset manifest：`artifacts/pamap2_sdforger_dataset/dataset_manifest.json`
- Best checkpoint：`outputs/checkpoints/gpt2/best/`
- Training summary：`outputs/checkpoints/gpt2/training_summary.json`
- Training log：`outputs/checkpoints/gpt2/training_log_history.csv`
- Generated embeddings：`outputs/generated/gpt2/generated_embeddings.csv`
- Generated windows：`outputs/generated/gpt2/generated_windows.npy`
- Generation summary：`outputs/generated/gpt2/generation_summary.json`
- Similarity metrics：`outputs/evaluation/similarity/gpt2/similarity_metrics.csv`
- Similarity summary：`outputs/evaluation/similarity/gpt2/similarity_summary.json`
- Utility metrics：`outputs/evaluation/utility/gpt2/utility_metrics.csv`
- Utility summary：`outputs/evaluation/utility/gpt2/utility_summary.json`
- Plots：`outputs/plots/gpt2/`
- Logs：`outputs/logs/`

## 11. 指标与评估含义

Similarity evaluation 当前实现：

- `MDD`
- `ACD`
- `SD`
- `KD`
- `ED`
- `DTW`

Utility evaluation 当前实现：

- `real_only`
- `synthetic_only`
- `real_plus_synthetic`

分类器：

- `RandomForestClassifier`

输出包括：

- metrics CSV
- confusion matrix CSV / PNG
- summary plots

## 12. 后续答复格式要求

Codex 以后在本仓库中给运行方案时，必须按以下结构组织：

### A. 需要修改的仓库文件

列出本地要改的文件路径，并说明为什么改。

### B. 需要用户重新上传到 Triton 的文件

列出必须上传的文件。不要笼统说“上传项目”，除非确实需要全量同步。

### C. 需要用户在 Triton 上执行的命令

命令应从以下两种形式之一开始：

```bash
ssh yangs9@triton.aalto.fi
cd /scratch/work/yangs9/ChatTS
```

或明确写：

```text
假设已登录 Triton，并已 cd 到 /scratch/work/yangs9/ChatTS：
```

### D. 如何检查是否成功

必须包含状态、日志和结果文件检查命令。

### E. 该步骤使用 CPU 还是 GPU

必须明确说明。GPU 步骤要指出 Slurm 脚本中的 `#SBATCH --gpus=1`。

不要只给理论建议。不要脱离当前 Triton 环境。不要继续围绕 smoke 做大修。重点放在正式 GPT-2 baseline 落地。

## 13. 文件同步注意事项

如果 Codex 当前是在远程项目 `/scratch/work/yangs9/ChatTS` 中工作，修改会直接落到 Triton，不需要用户再上传。

如果 Codex 当前是在 Windows 本地仓库 `C:\Users\ysj\Desktop\ChatTS` 中工作，用户需要把相关文件上传到 Triton。常见需要上传：

- 修改过的 `scripts/*.py`
- 修改过的 `pamap2_forger/*.py`
- 修改过的 `config/*.yaml`
- 修改过的 `slurm/*.slurm`
- 修改过的文档，如 `AGENTS.md`

如果只改了 `AGENTS.md`，只需要上传：

```text
AGENTS.md
```

不要随意要求上传 `artifacts/`、`outputs/`、`gpt2baseline/`，除非明确需要同步数据或结果。这些目录通常体积较大。

## 14. Git 与文件安全

- 当前仓库可能有大量未跟踪或本地生成文件。
- 不要无差别执行 `git reset --hard`、`git clean` 或删除 `artifacts/`、`outputs/`、`gpt2baseline/`。
- 修改配置时优先新增或明确说明覆盖目的。
- 如果更换 split、活动类别、window size、embedding 维度，必须重新生成 dataset artifacts，并在结果说明中记录新配置。
- Gemma 相关失败时，先检查 Hugging Face 权限、token、license、cache 和 GPU 显存，不要直接大改训练代码。
