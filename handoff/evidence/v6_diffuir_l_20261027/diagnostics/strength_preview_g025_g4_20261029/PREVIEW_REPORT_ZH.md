# 同批图六档物理退化预览

已完成32个独立Clean来源×6档=192张退化，另保留32张清晰原图。前8个来源和上一轮G1–G4对比一致，全部顺序冻结，不按结果挑图。

列：Clean GT / G0.25 / G0.5 / G1 / G2 / G3 / G4。每源一个固定原始ξ，六档独立从同一Clean生成，不对已退化图连续退化。复用当前已冻结模拟器、canonical buffers、数据及原始随机种子。原始256×256 RGB；PNG仅clip01及float64四舍五入到8bit用于显示，浮点数组未裁剪、未增强。

**G1–G4相对既有128张浮点退化图，最大绝对误差 0。** 原始ε_zernike/ε_tilt逐图SHA完全一致；192张退化有限、32个Clean与来源SHA吻合。单卡物理生成耗时 25.60 秒；进程总耗时 54.78 秒，包括启动和加载，不混作纯模拟吞吐。

本轮只有物理模拟，D0恢复调用0、训练更新0。仅使用当时空闲NPU4，原资源守护保留；生成进程正常退出、旧STOP与执行策略SHA保持。A/B未恢复。未加载或修改D0权重。

图库：[STRENGTH_GALLERY.html](<../../../../NOT_INCLUDED.md#ref-48462e88a905817a2492>)。前8张：[第一页](<../../../../NOT_INCLUDED.md#ref-4aea634bb21ff7f91c0e>)、[第二页](<../../../../NOT_INCLUDED.md#ref-72d48b368ed316f66cf7>)。单张与中央固定128×128局部在previews；原始PNG在images。

INPUT_AND_DISPLACEMENT.csv的PSNR只表示输入相对Clean的像素误差，位移为生成器实际场的像素单位均值/95分位，不把这些单独作为物理强弱的统一标准。32张预览不是全量Val1000统计，不是恢复性能测试。

选图：前次固定100可视化顺序中的前32个独立Clean，输入split=val；这是用户已观察过的可视化子集，用于确认强度。正式混训分布与预算尚未冻结，没有新的训练授权。

远端原始FP32：`/data2/yanglin/v6_diffuir_l_20261027/diagnostics/strength_preview_g025_g4_20261029/arrays`。本地PNG逐一SHA/尺寸/RGB校验通过，共224张。种子/原始ξ/来源/文件SHA见PAIR_PLAN.jsonl、GENERATED.jsonl、SOURCE_MANIFEST.json；生成命令见RUN_COMMAND.json。本地排版脚本build_gallery.py不调用模型。
