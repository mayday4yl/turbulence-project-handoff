# Heat-Chamber 扩大公开检索

日期：2026-10-29。范围：针对完整 Heat-Chamber RGB 退化帧与配对 GT 的公开搜索；延续原始官方入口核查，不运行恢复方法。

后续更新：文献检索又发现了 Figshare 公开的 200×90 张 PNG 测试结果包，已读取 ZIP 目录并抽检一张 RGB 图，但未确认原始输入和配对 GT。该新线索及十篇实际使用论文见 [论文核查报告](<../../../NOT_INCLUDED.md#ref-030ebbacc3117246a3d3>)。本文下方的灰度 demo 指另一份已下载演示文件，不能与新结果包混为一谈。

## 结论

**尚未找到公开可下载、来源可追溯且能核实为完整 Heat-Chamber RGB+GT 的副本。不能将这一有限范围的检索表述为“全网不存在”。**

扩大检索找到了可实际下载的后续论文演示包。检查发现它只有一个 100 帧灰度图像数组，没有 GT，也未在说明中标明属于 Heat-Chamber，因此不能用它冒充该 benchmark。原始 Heat-Chamber 的场景、帧数和 GT 数仍未知，固定真实测试集仍未建立。

## 检索及候选核对

| 来源 | 实际检查 | 与目标数据的关系 |
|---|---|---|
| 通用网页检索 | Heat Chamber、HeatChamber、heat_chamber、HeatChamber_Dataset；download/mirror/zip/网盘/百度等组合；追踪返回结果到作者仓库和数据说明 | 找到后续论文与竞赛线索，未获得可验证的完整原始包；不声称穷尽全部网页或网盘 |
| GitHub 代码检索 | 连写、下划线、带空格名称，加 turbulence/dataset/download，以及原始 Drive 文件 ID；结果 URL 已保存 | 原始 ID 的相关命中仍为 TurbNet/DATUM/MambaTM 的旧链接；其他命中包括仅含代码的复现实验及竞赛说明 |
| Hugging Face | 连接器不可用，改查公开 API；HeatChamber 和 Heat Chamber 名称查询均为空；另查 turbulence 并核对图像相关候选 | 未发现对应镜像；Strong_Turbulence 仓库元数据仅列 README/LICENSE/.gitattributes，ASU 名称候选未提供 Heat-Chamber 来源证明 |
| Kaggle | HeatChamber 查询为空；heat chamber 查询命中工业炉、火山、环境等其他数据 | 返回的候选与目标图像 benchmark 无关 |
| Zenodo | HeatChamber 查询为空；带引号 heat chamber 查询返回 7 条元数据记录，标题均为热处理、生理或农业等其他主题 | 未发现目标 benchmark 的可用副本；初始复合查询报 400，后用两个有效简单查询复核 |
| Internet Archive | 对原始 Drive 文件 URL 调用 Wayback available 接口 | HTTP 429，归档可用性未核实；不能据此声称无归档 |
| donglao/DiffeomorphicTemplate，CVPR 2024 作者演示仓库及 CVF 官方补充 ZIP | 实际下载 MAT 和 ZIP；校验 MAT 的 Git blob SHA1，ZIP 全部 CRC；比对 ZIP 内 MAT 与 GitHub MAT 的字节 | 两份来源保存的是同一示例，100 幅 271×271 单通道图像，无 GT，无原始 Heat-Chamber 场景身份说明 |
| whuluojia/ImTurb，后续论文作者仓库 | README、完整 Git 树及 main.m | README 的 Heat-Chamber 下载仍指向旧失效 ID；示例图与算法资源不能证明完整原始包已公开 |
| nedimbalihodzic/TurbulenceMitigation_Datasets，硕士论文相关实现 | README 与仓库目录 | README 明确仅保存修改后的 Python 文件，没有数据包 |
| ConVRT | 作者仓库 README 和项目网页 | 仓库标注 coming soon，网页仍在建设，没有发现数据下载入口 |
| UG2 2023/2024 相关数据页和解法仓库 | 竞赛官方数据说明、CodaLab 公开页面及仓库迁移后的 README | 指向文本识别/编码目标挑战数据，不能认作原始自然图像 Heat-Chamber 的完整镜像；未下载这些不同任务的数据 |

GitHub 搜索是有限结果集合，不覆盖未索引文件、私有仓库、历史分支和未公布的网盘。平台名称查询也不是对平台所有未标注内容的穷举证明。

## 已下载演示文件验收

作者代码仓库：[DiffeomorphicTemplate](https://github.com/donglao/DiffeomorphicTemplate)。固定源码 commit：`39297cc798239da5361125a43873051bec74cb6f`。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `downloads/diffeomorphic_sample_data.mat` | 5,619,287 | `40332d083ca1c481a6eb7fd43c75e637ed3ef85bc5296e4708d472995ef096da` |
| `downloads/diffeomorphic_cvpr2024_demo.zip` | 11,234,881 | `5b616f9a8a7e9a00b51576957694417b4a5d9cec95aa599f6a4c24f5754c0246` |

MAT 仅有变量 `image_stack`：形状 `[271,271,100]`，`uint8`，数值范围 `[16,255]`。按作者 demo.m 的读取方式，第三维为帧维；不是 RGB 通道。文件内没有清晰 GT 变量。首帧已检查并保存在 `evidence/extended_search/DEMO_FRAME0.png`。

Git blob SHA1 实算为 `9235936c2aadc994711d584fcb2ae3c549015d39`，与作者仓库树一致。CVF ZIP 共 251 个条目，CRC 检查通过；其中 `suppmat/sample_data.mat` 与 GitHub MAT 字节完全一致。包中其余 JPG 是光流工具自带示例，未作为该图像序列的 GT。

这些文件可用于独立研究代码或灰度无参考演示，不能作为我们单帧 RGB、PSNR/SSIM/LPIPS 配对真实测试集，也不能据此估计完整 Heat-Chamber 的规模。

## UG2 线索的范围差异

[UG2 2024 官方数据说明](https://cvpr2024ug2challenge.github.io/dataset24_t2.html)描述的是文本识别及编码目标恢复任务。文本 dry-run 为 50 个序列、每序列 100 帧，提供的是任务标签；final 为 500 个序列，参与者不获得 GT 标签。不能把这些文字标签等同于逐像素清晰 RGB GT，也不能将该挑战数据改称原始 Heat-Chamber。

## 剩余缺口与下一步

缺少的仍是可追溯的原始 RGB 退化帧、配对清晰 GT、场景目录及完整性依据。最直接的补充来源是原作者或实际使用该数据的后续研究团队保存的原包。本轮没有发邮件、评论或其他消息；没有调用恢复模型、训练服务器或加速设备。

所有公开平台响应、候选 README、搜索结果 URL、MAT 检查结果与 ZIP 清单位于 `evidence/extended_search/`。原始获取结果 JSON 继续将完整数据数量记为 null，而非 0。
