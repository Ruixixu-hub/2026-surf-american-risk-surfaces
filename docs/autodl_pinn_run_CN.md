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
