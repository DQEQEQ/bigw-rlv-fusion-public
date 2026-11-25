# BI-GW 与 RLV 实验过程与结果

本页汇总论文的正式实验，提供图表与逐图分数的对应关系。由 SwinFusion 的复现基线出发，BI-GW 按图像对分配结构损失权重，RLV 按每对图像的目标方差归一化局部方差匹配误差。使用共有的推理结构。

**主结果。** 种子 123 下，联合模型相对复现基线的六指标等权平均相对增益为 **1.676%**。五项指标在三个种子下都优于各自的同种子基线。联合模型的局部方差匹配误差比复现基线降低 **9.583%**。

- [复现说明](REPRODUCE.md)涵盖从数据准备、训练到推理和评估。
- [paper-v1 的 Assets](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/tag/paper-v1)提供 10 个正式 EMA 权重、权重清单、SHA-256 校验值及论文图 3–5 的作图材料。
- [paper-v2 的代码快照](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/tag/paper-v2)保存相应的训练和评估程序。
- [逐图分数](reference/scores)与[测试配对清单](reference/splits)可直接在线查看。公开仓库的页面和发布附件可直接查看和下载。

## 数据和训练过程

| 数据集 | 训练 | 测试 | 用途 |
| --- | --- | --- | --- |
| MSRS | 1,083 对 | 361 对 | 消融、机制检验、多种子、定性及方法比较 |
| TNO | — | 49 对 | 跨数据集评估、代表性方法比较 |
| RoadScene | — | 221 对 | 跨数据集评估 |
| LLVIP | — | 3,463 对 | 跨数据集评估 |

训练与推理使用已对齐的红外图像和可见光亮度图。训练时对每对图像同步裁取 128 × 128 图像块及增广。Adam 的初始学习率为 2 × 10⁻⁵，共训练 10,000 个优化器步。每个微批次包含 4 对，累积 3 个微批次后更新一次，有效批次大小 12。EMA 衰减为 0.999，评估统一使用第 10,000 步 EMA 权重。种子 123 比较六种监督配置，种子 456 和 789 各训练复现基线与联合模型。种子内使用相同初始化和样本顺序。

流程是准备图像对、选定监督配置和随机种子训练、读取相应 EMA 权重做推理，再按配对文件名逐一评估。强度、梯度和结构项的系数为 20、20、10。联合模型的 RLV 系数为 1。评估量包括 MI、NCIE、Qabf、Qp、Qy 和 VIF。各指标均以较大为优。

| 需要查看 | 仓库位置 |
| --- | --- |
| 完整操作说明 | [REPRODUCE.md](REPRODUCE.md) |
| 实验设定 | [configs/protocol.json](configs/protocol.json) |
| 数据准备 | [scripts/prepare_pairs.py](scripts/prepare_pairs.py) |
| 训练入口 | [scripts/train.py](scripts/train.py) |
| 损失实现 | [bigw_rlv/losses.py](bigw_rlv/losses.py) |
| 权重推理 | [scripts/infer.py](scripts/infer.py) |
| 六指标评估 | [scripts/evaluate_ivif.py](scripts/evaluate_ivif.py) |
| RLV 评估 | [scripts/evaluate_rlv.py](scripts/evaluate_rlv.py) |
| 表格重算 | [scripts/build_tables.py](scripts/build_tables.py) |
| 图 6 拼图重建 | [scripts/plot_fig6.py](scripts/plot_fig6.py) |

## RQ1 六组监督消融

| 配置 | 结构加权 | 使用 RLV |
| --- | --- | --- |
| 复现基线 | 整个微批共享一组梯度权重 | 无 |
| BI-EQ | 两个模态的 SSIM 得分各占 1/2 | 无 |
| BILS | 根据局部梯度能量随位置分配权重 | 无 |
| BI-GW | 每对图像各算一组梯度权重 | 无 |
| 仅 RLV | 整个微批共享一组梯度权重 | 有 |
| BI-GW + RLV | 每对图像各算一组梯度权重 | 有 |

**论文表 1。** 使用种子 123 和完整 MSRS 测试集。用原精度算出各指标相对增益后作等权平均。

| 方法 | MI | NCIE | Qabf | Qp | Qy | VIF | 平均增益 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 4.5241 | 0.8192 | 0.6576 | 0.5023 | 0.8996 | 0.9960 | 0.000 |
| BI-EQ | 4.5511 | 0.8185 | 0.6667 | 0.5054 | 0.9124 | 1.0083 | +0.864 |
| BILS | 4.5459 | 0.8182 | 0.6692 | 0.5079 | 0.9138 | 1.0148 | +1.119 |
| BI-GW | 4.5979 | 0.8190 | 0.6680 | 0.5114 | 0.9161 | 1.0132 | +1.428 |
| RLV-only | 4.5742 | 0.8196 | 0.6617 | 0.5065 | 0.9059 | 1.0045 | +0.700 |
| BI-GW+RLV | 4.6214 | 0.8193 | 0.6694 | 0.5127 | 0.9159 | 1.0179 | +1.676 |

前期在 108 对图像上筛选 RLV 的系数，正式实验采用系数 1.00。以下均为相对筛选基线的结果。

| RLV 系数 | 筛选图像对数 | 平均相对增益 | 提高指标数 |
| --- | --- | --- | --- |
| 0.25 | 108 | +1.083% | 5/6 |
| 0.50 | 108 | +1.151% | 5/6 |
| 1.00 | 108 | +1.487% | 6/6 |

## RQ2 梯度与局部方差机制

**BI-GW。** 固定锚点图像对、融合预测和微批次大小，替换同批其他图像对。归一化偏差为两组锚点梯度差的 L2 范数除以单独计算锚点时的梯度范数加稳定项。在红外主导和可见光主导批次的对照中，批次共享权重为 **2.619**，BI-GW 为 **4.446 × 10⁻⁷**。该梯度为结构损失对固定融合预测的梯度。保留了归档汇总，原始锚点及同批样本标识待核对。

| 批次对照 | 批次共享 | BI-GW |
| --- | --- | --- |
| 锚点单独 / 红外主导 | 0.662372 | 7.342778 × 10⁻⁷ |
| 锚点单独 / 可见光主导 | 1.956786 | 7.454358 × 10⁻⁷ |
| 红外主导 / 可见光主导 | 2.619158 | 4.446009 × 10⁻⁷ |

**RLV。** 在 3 × 3 邻域计算局部方差，逐位置取两种模态方差的较大值为目标，再按该图像对的目标方差均值加稳定项归一化。MSRS 的 361 对测试图像上，复现基线、BI-GW、仅 RLV 与联合模型的平均误差依次为 0.113801、0.109321、0.106211、0.102896。

| 对照 | 相对误差降幅 | 平均降幅的 95% CI | 误差较低图像对 |
| --- | --- | --- | --- |
| BI-GW 对复现基线 | 3.937% | [0.003718, 0.005295] | 250/361 |
| 仅 RLV 对复现基线 | 6.669% | [0.007214, 0.007989] | 361/361 |
| 联合模型对 BI-GW | 5.877% | [0.006142, 0.006709] | 361/361 |
| 联合模型对仅 RLV | 3.122% | [0.002678, 0.003964] | 216/361 |
| 联合模型对复现基线 | 9.583% | [0.009965, 0.011892] | 361/361 |

配对差值按参考方法误差减候选方法误差计算，使用 20,000 次 Bootstrap 重采样。五组双侧 Wilcoxon 检验经 Holm 校正均有 p < 0.001。

## RQ3 跨数据集与三个训练种子

**论文表 2。** 固定在 MSRS 上得到的种子 123 权重，直接评估 TNO、RoadScene 和 LLVIP。联合模型相对各自复现基线的六指标平均相对增益分别为 **1.368%**、**0.656%**、**1.444%**。

| 数据集 | 方法 | MI | NCIE | Qabf | Qp | Qy | VIF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TNO | Baseline | 3.2731 | 0.8086 | 0.5102 | 0.3534 | 0.8462 | 0.7201 |
| TNO | BI-GW | 3.3888 | 0.8090 | 0.5120 | 0.3607 | 0.8479 | 0.7298 |
| TNO | BI-GW+RLV | 3.3774 | 0.8090 | 0.5140 | 0.3605 | 0.8488 | 0.7339 |
| TNO | Relative change (%) | +3.187 | +0.051 | +0.745 | +2.008 | +0.303 | +1.915 |
| RoadScene | Baseline | 3.4080 | 0.8084 | 0.4703 | 0.3820 | 0.7865 | 0.6269 |
| RoadScene | BI-GW | 3.4881 | 0.8087 | 0.4746 | 0.3836 | 0.7907 | 0.6365 |
| RoadScene | BI-GW+RLV | 3.4261 | 0.8085 | 0.4764 | 0.3832 | 0.7925 | 0.6332 |
| RoadScene | Relative change (%) | +0.529 | +0.012 | +1.305 | +0.311 | +0.771 | +1.010 |
| LLVIP | Baseline | 3.9892 | 0.8121 | 0.6623 | 0.3942 | 0.8143 | 0.8861 |
| LLVIP | BI-GW | 4.1586 | 0.8132 | 0.6549 | 0.4029 | 0.8217 | 0.8947 |
| LLVIP | BI-GW+RLV | 4.1478 | 0.8131 | 0.6596 | 0.4036 | 0.8235 | 0.8991 |
| LLVIP | Relative change (%) | +3.976 | +0.128 | -0.410 | +2.386 | +1.125 | +1.459 |

**论文表 3。** 每个种子使用同种子复现基线。表中的误差项为样本标准差。三个种子的平均相对增益依次为 1.676%、1.426% 和 1.140%。

| 指标 | 复现基线 | 联合模型 | 相对增益 % | 提升种子 |
| --- | --- | --- | --- | --- |
| MI | 4.6115 ± 0.0761 | 4.6873 ± 0.0573 | +1.648 ± 0.522 | 3/3 |
| NCIE | 0.8200 ± 0.0007 | 0.8198 ± 0.0005 | -0.019 ± 0.029 | 1/3 |
| Qabf | 0.6614 ± 0.0035 | 0.6718 ± 0.0021 | +1.581 ± 0.237 | 3/3 |
| Qp | 0.5084 ± 0.0052 | 0.5161 ± 0.0029 | +1.529 ± 0.484 | 3/3 |
| Qy | 0.9049 ± 0.0054 | 0.9201 ± 0.0041 | +1.682 ± 0.159 | 3/3 |
| VIF | 1.0029 ± 0.0064 | 1.0236 ± 0.0049 | +2.063 ± 0.228 | 3/3 |

种子 123 的逐图像配对采用 10,000 次 Bootstrap 估计置信区间，使用双侧 Wilcoxon 检验并在六项指标内做 Holm 校正。下面是联合模型相对复现基线的检验结果。

| 指标 | 相对增益 | 逐图胜率 | Holm 校正 p 值 |
| --- | --- | --- | --- |
| MI | +2.149% | 84.5% | 1.61e-44 |
| NCIE | +0.014% | 56.5% | 0.096 |
| Qabf | +1.791% | 97.5% | 1.39e-58 |
| Qp | +2.073% | 79.2% | 7.68e-38 |
| Qy | +1.818% | 72.3% | 8.64e-28 |
| VIF | +2.209% | 98.3% | 5.68e-59 |

## RQ4 图像样例、复杂度与代表性方法

论文图 6 展示 MSRS 的 00959N、00111D、00345D，分别为夜间行人、白天车辆与白天行人。同一行各方法的橙框采用同位置与同比例放大。仓库提供 [SVG 布局](reference/fig6-layout.svg) 和 [图 6 生成脚本](scripts/plot_fig6.py)，模型图像可从对应 EMA 权重推理得到。

**论文表 4。** 计算量使用 256 × 256 输入。四种方法均使用同一个推理结构，计时结果随硬件变化。

| 方法 | 参数量 | GFLOPs | 新增推理分支 |
| --- | --- | --- | --- |
| Baseline | 973681 | 151.1048 | No |
| BI-GW | 973681 | 151.1048 | No |
| RLV-only | 973681 | 151.1048 | No |
| BI-GW+RLV | 973681 | 151.1048 | No |

**论文表 5。** 使用相同的测试图像对与指标实现。SwinFusion 行使用官方权重。表 1 中的复现基线是按本文实验设置重新训练的。

| 数据集 | 方法 | MI | NCIE | Qabf | Qp | Qy | VIF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSRS | CDDFuse | 4.7531 | 0.8201 | 0.6786 | 0.5174 | 0.9187 | 1.0339 |
| MSRS | LUT-Fuse | 3.6262 | 0.8111 | 0.6046 | 0.3987 | 0.8283 | 0.8823 |
| MSRS | PIAFusion | 4.6527 | 0.8204 | 0.6892 | 0.5162 | 0.9053 | 0.9899 |
| MSRS | SFDFusion | 4.7348 | 0.8205 | 0.6669 | 0.5131 | 0.9033 | 0.9979 |
| MSRS | FSATFusion | 5.6458 | 0.8254 | 0.6914 | 0.5184 | 0.9494 | 1.0066 |
| MSRS | SwinFusion | 4.5285 | 0.8198 | 0.6542 | 0.5009 | 0.8888 | 0.9900 |
| MSRS | BI-GW | 4.5979 | 0.8190 | 0.6680 | 0.5114 | 0.9161 | 1.0132 |
| MSRS | BI-GW+RLV | 4.6214 | 0.8193 | 0.6694 | 0.5127 | 0.9159 | 1.0179 |
| TNO | CDDFuse | 3.1735 | 0.8083 | 0.5210 | 0.3739 | 0.8399 | 0.7768 |
| TNO | LUT-Fuse | 3.7160 | 0.8101 | 0.4618 | 0.3238 | 0.8283 | 0.6903 |
| TNO | PIAFusion | 3.2235 | 0.8093 | 0.5627 | 0.3796 | 0.8489 | 0.7276 |
| TNO | SFDFusion | 4.2271 | 0.8127 | 0.5339 | 0.4086 | 0.8517 | 0.8551 |
| TNO | FSATFusion | 5.7550 | 0.8221 | 0.5701 | 0.4555 | 0.9418 | 0.8883 |
| TNO | SwinFusion | 3.2804 | 0.8087 | 0.5110 | 0.3509 | 0.8424 | 0.7312 |
| TNO | BI-GW | 3.3888 | 0.8090 | 0.5120 | 0.3607 | 0.8479 | 0.7298 |
| TNO | BI-GW+RLV | 3.3774 | 0.8090 | 0.5140 | 0.3605 | 0.8488 | 0.7339 |

## 在 GitHub 查看对应材料

| 实验 | 文件 | 内容 |
| --- | --- | --- |
| 六组消融 | [msrs-design-ablation.csv](reference/scores/msrs-design-ablation.csv)、[msrs-fourway.csv](reference/scores/msrs-fourway.csv) | 表 1 的逐图分数 |
| BI-GW 梯度诊断 | [anchor-context.csv](reference/scores/anchor-context.csv) | 三组批次对照的归档偏差 |
| RLV 机制 | [msrs-rlv.csv](reference/scores/msrs-rlv.csv)、[rlv-paired.csv](reference/scores/rlv-paired.csv) | 361 对的误差和五组配对统计 |
| MSRS 六指标配对检验 | [msrs-main-paired.csv](reference/scores/msrs-main-paired.csv) | 置信区间、胜率、Wilcoxon 与 Holm 校正 |
| 跨数据集 | [tno49.csv](reference/scores/tno49.csv)、[roadscene221.csv](reference/scores/roadscene221.csv)、[llvip3463.csv](reference/scores/llvip3463.csv) | 三种方法的逐图分数 |
| 三种子汇总 | [three-seed-perseed.csv](reference/scores/three-seed-perseed.csv)、[three-seed-gains.csv](reference/scores/three-seed-gains.csv)、[three-seed-aggregate.csv](reference/scores/three-seed-aggregate.csv) | 逐种子数据和跨种子汇总 |
| 其他两个种子逐图结果 | [msrs-seed456.csv](reference/scores/msrs-seed456.csv)、[msrs-seed789.csv](reference/scores/msrs-seed789.csv) | 同种子基线与联合模型 |
| 代表性方法及复杂度 | [comparison-msrs.csv](reference/scores/comparison-msrs.csv)、[comparison-tno.csv](reference/scores/comparison-tno.csv)、[complexity.csv](reference/scores/complexity.csv) | 表 4 与表 5 的原始分数和测量记录 |

- **权重。** 从 [paper-v1 的 Assets](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/tag/paper-v1) 下载 [种子 123 复现基线](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/download/paper-v1/seed123-baseline-step10000-ema.pth)和[种子 123 联合模型](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/download/paper-v1/seed123-bigw-rlv-step10000-ema.pth)即可开始主对照。其他变体与种子见 [weights-manifest.csv](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/download/paper-v1/weights-manifest.csv)，校验值见 [SHA256SUMS.txt](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/download/paper-v1/SHA256SUMS.txt)。
- **图 3–5。** 下载 [reproduction-materials.zip](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/download/paper-v1/reproduction-materials.zip)，查看包内 README.md。data/ 内的 fig3-rlv-mechanism.zip、fig4-cross-dataset.zip 和 fig5-ablation-and-stability.zip 对应论文图号，figures/ 内有绘图脚本。
- **实验数据。** [TNO 的 49 对清单](reference/splits/tno49.csv)与[原始压缩包映射](reference/splits/tno49-origin-matches.csv)可一并查看。其中 20 对由场景、模态和尺寸补配。逐图指标来自原实验归档。训练日志和部分原始输入保存在本机备份。
