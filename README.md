
# 🚀 GEPO: Group Expectation Policy Optimization for Heterogeneous Reinforcement Learning

> **Paper**: [GEPO: Group Expectation Policy Optimization for Heterogeneous Reinforcement Learning](https://arxiv.org/abs/2508.17850)  
> **Codebase**: Built on [`trl`](https://github.com/huggingface/trl) & [`open-r1`](https://github.com/huggingface/open-r1)

![GEPO Architecture](./MainFig.png)

> **Figure 1**: GEPO improves upon GRPO and GSPO by employing **group-level importance weights** to enhance training stability. It demonstrates superior performance in both **zero-delay (online)** and **high-delay (up to 1800s)** heterogeneous RL scenarios.

---

## 📰 Latest Update: ∆L Normalization Integrated!

<details>
<summary>🎯 <strong>Added Implementation of ∆L Normalization — Unbiased & Minimum-Variance!</strong></summary>

<br>

<h2 align="center">✨ ∆L Normalization: Rethink Loss Aggregation in RLVR</h2>

📅 **Release Date**: September 9, 2025 (arXiv)  
📄 **Paper**: [∆L Normalization: Rethink Loss Aggregation in RLVR](https://arxiv.org/abs/2509.07558)  
🧑‍💻 **Authors**: Zhiyuan He, Xufang Luo (Microsoft Research), Yike Zhang (Tsinghua), et al.  
🔗 **Implementation**: Based on [Delta-L-Normalization](https://github.com/zerolllin/Delta-L-Normalization)

---

### ⚡ Why It Matters
In RLVR, response lengths vary dramatically — leading to **high gradient variance** and **biased updates** in existing methods (GRPO, DAPO, Dr. GRPO).  
∆L Normalization solves both:
✅ **Unbiased estimator** of true policy gradient  
✅ **Theoretically minimal variance** (when `α=1`)  
✅ **Plug-and-play** — <10 lines to integrate

> 💡 **Pro Tip**:  
> - Use `α=1` for **minimum variance** (default, stable training).  
> - Use `α=0.75` for **Math tasks** — better utilization of long, informative responses.

</details>

---

## 🧠 Importance Weight Computation (Policy Optimization Methods)

```python
# Token-level importance ratio (e.g., GRPO, Dr. GRPO, BNPO)
if self.loss_type in ["grpo", "dr_grpo", "bnpo"]:
    coef_1 = learner_token_p / sampler_token_p

# Sequence-level importance ratio (GSPO)
elif self.loss_type == "gspo":
    coef_1 = learner_seq_p / sampler_seq_p

# Group-level importance ratio (GEPO — Ours)
elif self.loss_type == "gepo":
    normalized_q = sampler_seq_p.detach() / sampler_seq_p.sum().detach()
    coef_1 = learner_seq_p / (normalized_q * sampler_seq_p).sum()
```

> 📌 **Note**: GEPO computes importance weights at the **group level**, stabilizing training under heterogeneous sampling delays.

---

## ⚙️ Heterogeneous Reinforcement Learning Setup

> 📁 Enter the project directory first. Adjust paths in scripts if your directory differs.

### 1️⃣ Launch the Learner (4×A100 80GB)

```bash
cd ./open-r1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash sh_dir/HeteroRL_Learner_4gpus.sh \
    learner_script_checkpoint \
    GEPO_think_1th \
    1 \
    v6b \
    gepo \
    1L2S_GEPO_diff32_think
```

### 2️⃣ Launch Samplers (Run in Sequence)

> 🔄 To resume from checkpoint: Set `model_name_or_path` to your checkpoint path.

```bash
# Launch 4 sampler processes in background
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 0 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 1 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 2 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 3 &
```

---

## 🌐 Online Reinforcement Learning (4×A100 80GB)

Supports multiple loss types:
- [`grpo`](https://arxiv.org/abs/2402.03300)
- [`bnpo`](https://arxiv.org/abs/2506.02864)
- [`dr_grpo`](https://arxiv.org/abs/2503.20783)
- [`gspo`](https://arxiv.org/abs/2507.18071)
- [`∆L Norm`](https://arxiv.org/abs/2509.07558)
- **`gepo` (ours)** 👈

```bash
cd /userhome/Research_HUB/GPG/open-r1
CUDA_VISIBLE_DEVICES="0,1,2,3" \
MASTER_PORT=29510 \
bash sh_dir/Online_gXpo_4gpus.sh gepo
```

---

## 📚 Citation

If you use **GEPO** or find this code helpful, please cite:

```bibtex
@misc{gepo2025,
  title     = {Group Expectation Policy Optimization for Heterogeneous Reinforcement Learning},
  author    = {Han Zhang and Ruibin Zheng and Zexuan Yi and Zhuo Zhang and Hanyang Peng and Hui Wang and Zike Yuan and Cai Ke and Shiwei Chen and Jiacheng Yang and Yangning Li and Xiang Li and Jiangyue Yan and Yaoqi Liu and Liwen Jing and Jiayin Qi and Ruifeng Xu and Binxing Fang and Yue Yu},
  year      = {2025},
  eprint    = {2508.17850},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url       = {https://arxiv.org/abs/2508.17850}
}
```

