# 接手老师先读这里

更新：2026年11月4日。正文聚焦现有模块与接手运行；历史演进按需查附录。资料准备完成不等于接收人已经签收。

## 当前要接手什么

任务是单帧大气湍流图像恢复。当前交接包含两个已有模块：

|模块|现在的作用|接手入口|
|---|---|---|
|V5 Simulator|可复用的物理退化生成器，解释模糊、位移及强度路径|[原理与对应变量](SIMULATOR_GUIDE.md)|
|最新UniDB D0|现有像素扩散恢复基线，默认示例使用原路线U67180 EMA|[当前版本、权重与状态](CURRENT_VERSION.md)|
|会议提出的physics + consistency|下一步适配目标，尚待接入接手方原代码|[方法与接口](METHODS_AND_NEXT_STEP.md)|

V5 Simulator与最新UniDB仍是独立模块；不能将二者写成一个已经完成的新方法。

## 建议按这个顺序读和跑

1. [当前版本说明](CURRENT_VERSION.md)：先了解交接对象、模型身份、现状和入口。
2. [模拟器图解](visual_guide.html#simulator)：看两条支路；切换固定ξ的强度图，对比blur与warp。
3. [代码导航](CODE_MAP.md)：从`simulator_v5.py`及`unidb_d0_20261102/core.py`开始，按函数查实现。
4. [NPU运行步骤](NPU_RUNBOOK.md)：登录、核对环境/资产，跑CPU模拟器，再在分配到的NPU上运行。
5. [当前方法与下一步接口](METHODS_AND_NEXT_STEP.md)：确认接手方consistency定义后再开发。
6. [论文阅读顺序](PAPERS.md)：优先P2S、UniDB与接手方真实consistency论文，其他论文按需查阅。

[本次运行验证](VALIDATION.md) · [齐备范围与待对接项](COMPLETENESS_REVIEW.md) · [接收清单](MEETING_CHECKLIST.md)。

## 历史背景只需先知道这些

研究曾检验物理递归、多种恢复骨干和扩散基线。V5 B有约0.293dB的局部收益，但末步仍弱于首步；最新UniDB最终五档PSNR均低于相应Input。已有实验已经收尾，多步与真实域稳定优势仍未确立。

需要追查某项设计或失败时再打开：[历史方法](METHODS_HISTORY.md)、[完整演进与失败记录](../DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md)、原[主交接PDF](../../handoff/documents/Turbulence_Project_Handoff_FINAL.pdf)/[Word](../../handoff/documents/Turbulence_Project_Handoff_FINAL.docx)。它们是保留证据，不是首次接手的阅读前置条件。

论文草稿和汇报按需查[材料索引](PAPER_MATERIALS.md)。数据、大权重、完整官方vendor按[资产登记](../../handoff/registers/ASSET_REGISTER.csv)取得；本ZIP不是完整服务器镜像。
