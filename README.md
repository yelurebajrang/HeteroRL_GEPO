## [**GEPO: Group Expectation Policy Optimization for Heterogeneous Reinforcement Learning**](https://arxiv.org/abs/2508.17850) 

![GEPO](./MainFig.png)

Figure-1: GEPO improves upon GRPO and GSPO by employing group-level importance weights to enhance training stability. In both zero-delay (online) and high-delay (up to 1800 seconds) heterogeneous reinforcement learning scenarios, GEPO demonstrates superior stability and better evaluation performance.

The code is built on [trl](https://github.com/huggingface/trl)/[openR1](https://github.com/huggingface/open-r1).


<details>
<summary>🚀 <strong>News: Added implementation of ∆L Normalization — Unbiased & Minimum-Variance!</strong></summary>

<br>

<h2 align="center">🎯 ∆L Normalization: Rethink Loss Aggregation in RLVR</h2>

📅 **Date**: September 9, 2025 (arXiv)  
📄 **Paper**: [**∆L Normalization: Rethink Loss Aggregation in RLVR**](https://arxiv.org/abs/2509.07558)  
🧑‍💻 **Authors**: Zhiyuan He, Xufang Luo (Microsoft Research), Yike Zhang (Tsinghua), et al.  
🔗 **Code**: Code is based on [github.com/zerolllin/Delta-L-Normalization](https://github.com/zerolllin/Delta-L-Normalization)

---

### ⚡ **Why It Matters**
In Reinforcement Learning with Verifiable Rewards (**RLVR**), response lengths vary wildly — from dozens to thousands of tokens. This causes:
- ❌ **High gradient variance** → unstable training
- ❌ **Biased updates** in existing methods (GRPO, DAPO)

**∆L Normalization solves both problems**:
✅ **Unbiased Estimator** — Matches true policy gradient expectation.  
✅ **Minimum Variance** — Theoretically proven to minimize gradient noise (when α=1).  
✅ **Plug-and-Play** — <10 lines of code to implement.

---

### 📈 **Key Results**
Extensive experiments on **Qwen2.5-3B/7B** across **CountDown** & **Math** tasks show:
- 🥇 **Highest Accuracy**: Consistently outperforms GRPO, DAPO, Dr. GRPO.
- 📉 **Most Stable Training**: Near-perfect monotonicity score (>0.94 Pearson correlation).
- 🧠 **Better Utilization of Long Responses**: Adaptive α parameter (α=0.75 optimal for Math).

> _“∆L Normalization changes how we think about normalization in RLVR.”_ — Paper Authors

---

### 🆚 **Theoretical Advantage**
| Method       | Unbiased? | Gradient Variance | Coefficient of Variation (CV) |
|--------------|-----------|-------------------|-------------------------------|
| GRPO         | ❌ Biased | Medium            | Low                           |
| DAPO         | ❌ Biased | High              | High                          |
| Dr. GRPO     | ✅        | High              | High                          |
| **∆L Norm (Ours)** | ✅        | **Minimum**       | **Lowest**                    |

---

> 💡 **Pro Tip**: Set `α=1` for minimum variance (default). Use `α=0.75` for Math tasks to better leverage long, informative responses.

</details>


### Importance weight computation for different policy optimization methods
```python
# Token level
if self.loss_type in ["grpo","dr_grpo","bnpo"]: 
    coef_1 = learner_token_p / sampler_token_p
# Sequence level
elif self.loss_type == "gspo":  
    coef_1 = learner_seq_p / sampler_seq_p
# Group level
elif self.loss_type == "gepo": 
    normalized_q = sampler_seq_p.detach() / (sampler_seq_p.sum().detach())
    coef_1 = learner_seq_p / (normalized_q * sampler_seq_p).sum() 
```


### Heterogeneous Reinforcement Learning

Enter the current directory (if the directory is different, you need to replace the corresponding path variables in the script).

Launch the learner firstly（using 4 * 80GB Nvidia A100 by default）
```shell
cd ./open-r1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sh_dir/HeteroRL_Learner_4gpus.sh learner_script_checkpoint GEPO_think_1th 1 v6b gepo 1L2S_GEPO_diff32_think
```
Sampler: launch samplers one by one in sequence
resume from checkpoint: put the path of checkpoint into model_name_or_path
```shell
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 0 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 1 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 2 &
bash sh_dir/HeteroRL_Sampler_4gpus.sh sampler_script_checkpoint GEPO_think_1th v6b gepo 1L2S_GEPO_diff32_think 3 &
```

### Online Reinforcement Learning（using 4 * 80GB Nvidia A100 by default）:

We support [grpo](https://arxiv.org/abs/2402.03300)/[bnpo](https://arxiv.org/abs/2506.02864)/[dr_grpo](https://arxiv.org/abs/2503.20783)/[gspo](https://arxiv.org/abs/2507.18071)/[∆L Normalization](https://arxiv.org/abs/2509.07558)/[***gepo*** (ours)](https://arxiv.org/abs/2508.17850) loss currently.
```shell
cd /userhome/Research_HUB/GPG/open-r1
CUDA_VISIBLE_DEVICES="0,1,2,3" MASTER_PORT=29510 bash sh_dir/Online_gXpo_4gpus.sh gepo
```
### Citation

If you find GEPO or code useful, please cite
```bibtex
@misc{GEPO,
      title={Group Expectation Policy Optimization for Heterogeneous Reinforcement Learning}, 
      author={Han Zhang and Ruibin Zheng and Zexuan Yi and Zhuo Zhang and Hanyang Peng and Hui Wang and Zike Yuan and Cai Ke and Shiwei Chen and Jiacheng Yang and Yangning Li and Xiang Li and Jiangyue Yan and Yaoqi Liu and Liwen Jing and Jiayin Qi and Ruifeng Xu and Binxing Fang and Yue Yu},
      year={2025},
      eprint={2508.17850},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.17850}, 
}
```
