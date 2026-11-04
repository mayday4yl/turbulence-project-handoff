# 第三方来源与许可记录

本文件依据仓库内 [SOURCE_SNAPSHOT_PLAN.json](provenance/SOURCE_SNAPSHOT_PLAN.json) 以及随源码保留的许可原文编写，记录这次私有交接实际包含和排除的内容。**本仓库未设置统一开放许可证；下列组件的许可不自动覆盖本项目研究代码、其他上游、数据、权重或整套仓库。** 私有存放也不替代各权利人的许可条件。后续公开发布或再分发时，需要逐项核实来源、衍生关系与授权。

## 已随包保留的许可原文

| 对应内容 | 实物许可与来源边界 |
|---|---|
| TurbulenceSim_P2S 上游快照 `5b743872bc66` 的许可文字 | [LICENSE](source_snapshot/output/v4_preparation/sources/TurbulenceSim_P2S_5b743872bc66/LICENSE)：CC BY-NC-SA 4.0。此处仅收录该许可文字，没有收录其官方源码树、权重或数组。它不自动覆盖 PiRN、Places365 或本研究仓库。 |
| NAFNet 候选 vendor | [LICENSE](source_snapshot/v4_nafnet_sidd116m_candidate_20261005/vendor/nafnet/LICENSE)：包含 MIT、Copyright (c) 2022 megvii-model，以及 BasicSR Authors 的 Apache License 2.0 许可段。保留完整文件，不将其简化为整个仓库 MIT。 |
| V5 Concertormer 所存上游 vendor | [LICENSE](source_snapshot/v5_concertormer_10k_20261021/vendor/upstream/LICENSE)：原上游文件包含 MIT、Copyright (c) 2022 megvii-model，及 BasicSR Apache License 2.0 段。对应源文件仍须结合其自己的署名与上游身份记录阅读。 |
| einops 0.8.1 | [LICENSE](source_snapshot/v5_concertormer_10k_20261021/vendor/python/einops-0.8.1.dist-info/licenses/LICENSE)：MIT，Copyright (c) 2018 Alex Rogozhnikov。 |
| DiffUIR-L stage1 的部分模型与 metrics 源码 | [LICENSE.txt](source_snapshot/v5_diffuir_l_stage1_20261026/upstream/LICENSE.txt)：MIT，Copyright (c) 2024 iSEE。计划中记录上游提交 `c71437ce86ad1b098aa9be4678dc82e8b99649df`；未复制模型权重及 NIQE 参数数组。 |

这些原许可文件与所选源码一同按原字节保存，SHA 位于源码计划中。此处表格仅作导航，适用条款以对应原文件及实际组件来源为准。

## 许可或来源未核实而排除的 vendor

以下是计划中明确排除的路径，仓库不包含这些路径的 vendor 字节：

| 原项目路径 | 本次记录 |
|---|---|
| `unidb_d0_20261102/vendor/unidb_gou/` | 本机官方镜像未找到可验证的上游许可。仅保存 [UPSTREAM_SOURCE.json](source_snapshot/unidb_d0_20261102/protocol/UPSTREAM_SOURCE.json) 等元数据，记录提交 `1bb63e318891b932c6e641b98729643e46b91e06` 及 blob/SHA。运行前须自行取得授权明确的依赖。 |
| `v4_swinir_l1_train500_2k_20261001/vendor/pirn/` | 未验证这一具体本地上游快照的许可。 |
| `v4_swinir_l1_train500_2k_20261001/vendor/timm_layers/` | 未验证这一裁剪子集的完整许可及来源绑定。 |
| `v4_restormer_train2000_npu_20k_20261003/vendor/restormer/d2turb/` | 本地 D²Turb 作者树未找到 LICENSE/COPYING；不能以另一标准 Restormer 仓库的许可证代替。 |

v1—v3 所依赖的 PiRN 提交 `12b9ef6a32c75143c23402a7ab8f843624ca99e8` 在本机官方副本中未找到仓库级许可；因此未收录其官方源码树、`upstream_simulator_new.py` 或 P2S/PSF/积分表资产。项目中的 `simulator.py` 是保留上游数学的设备适配，存在需进一步确认的衍生许可与署名边界。PiRN 内 ResizeRight 子目录的 MIT 许可仅覆盖其子组件，该子组件也未在本次相应依赖组中分发。

计划另记录部分 V4 SwinIR/PiRN、裁剪 timm 及 D²Turb 相关实现的许可或来源边界。保存现存项目研究代码不构成新增公开许可。源码身份相符只说明字节来源，不能当作再分发权利证明。

## 数据、权重和运行资产

Places365、LSDIR、各外部测试集、官方预训练模型、模拟器资产、训练 checkpoint、raw ξ、数组及全量预测未作为载荷进入本仓库。路径、SHA、来源和历史收据的保留不授予这些资产的使用或再分发权。取得资产时请同时保留其上游条件，详见[数据](handoff/registers/DATA_REGISTER.csv)、[权重](handoff/registers/WEIGHT_REGISTER.csv)与[资产](handoff/registers/ASSET_REGISTER.csv)登记。

未核实的依赖、缺失配置、历史冻结 SHA 不一致项及环境限制均以源码计划各组 `gaps` 为准。本文件不补写缺失的许可证，也不对现存研究代码作统一重新许可。
