# AutoDL运行Arm C/D单种子PINN

本分支包含AutoDL/Linux运行所需的代码、冻结数据集和配置。不要上传训练中的checkpoint到GitHub；AutoDL实例应使用持久化数据盘保存`results/08_pinn_gap/04_heldout_pilots/`。

## 1. Clone与环境

```bash
git clone --branch codex/autodl-pinn-pilots \
  git@github.com:Ruixixu-hub/2026-surf-american-risk-surfaces.git
cd 2026-surf-american-risk-surfaces
```

建议选择已经包含PyTorch与CUDA的AutoDL镜像。确认Python环境后安装其余依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-pinn.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

如果镜像已经安装可用的CUDA版PyTorch，不要再安装另一个CPU版PyTorch覆盖它。

## 2. 使用tmux

```bash
tmux new -s surf-pinn
```

SSH断开后，使用`tmux attach -t surf-pinn`恢复终端。

## 3. 运行

Arm D、seed 101：

```bash
bash scripts/linux/run_autodl_pinn_pilot.sh D 101 all
```

Arm C、seed 101：

```bash
bash scripts/linux/run_autodl_pinn_pilot.sh C 101 all
```

意外中断后执行相同命令，训练会从checkpoint恢复。

### RTX 5090并行运行

单个PINN网络较小，RTX 5090可以先使用4个独立进程训练不同regime：

```bash
bash scripts/linux/run_autodl_pinn_parallel.sh C 101 4
```

脚本会依次完成：并行训练、生成67个高精度参考、一次性评分。每个分片有独立日志：

```text
results/08_pinn_gap/04_heldout_pilots/parallel_logs/
```

已有的顺序训练checkpoint会被`--resume`复用。不要让原来的顺序命令和并行命令同时运行。训练步数仍固定为冻结配置；并行脚本只把每个任务的墙钟安全上限自动设为`并行数×1小时`，避免共享GPU造成误判超时。

只运行一个指定分片的底层命令是：

```bash
bash scripts/linux/run_autodl_pinn_pilot.sh C 101 train 0 4 14400
```

若只训练而暂不打开held-out参考和评分：

```bash
bash scripts/linux/run_autodl_pinn_pilot.sh D 101 train
```

完成67个训练任务后，再执行：

```bash
bash scripts/linux/run_autodl_pinn_pilot.sh D 101 reference
bash scripts/linux/run_autodl_pinn_pilot.sh D 101 score
```

## 4. 结果

Arm D总结：

```text
results/08_pinn_gap/04_heldout_pilots/arm_d_seed101/single_seed_pilot_summary.json
```

Arm C总结：

```text
results/08_pinn_gap/04_heldout_pilots/arm_c_seed101/single_seed_pilot_summary.json
```

该实验是validation-selected single-seed pilot，不是原注册的五种子held-out实验。
