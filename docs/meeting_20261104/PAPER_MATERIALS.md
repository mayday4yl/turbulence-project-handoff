# 论文、汇报与结果素材

## 已放入本补充的素材

|材料|入口|实际状态|
|---|---|---|
|现有论文主文件|[main.tex](paper_snapshot/main.tex)|原2026-10-28/29草稿字节副本；题目和author占位保留|
|摘要|[0_abstract.tex](paper_snapshot/sec/0_abstract.tex)|已有草稿；描述的方法与最终研究方案仍需对齐|
|Introduction / Related Work|[1_intro.tex](paper_snapshot/sec/1_intro.tex)|同一个文件内包含两个章节|
|原参考文献|[main.bib](paper_snapshot/main.bib)|原稿引用集合，未自动与本次新增阅读列表合并|
|模板依赖|[preamble.tex](paper_snapshot/preamble.tex)、[cvpr.sty](paper_snapshot/cvpr.sty)、[bst](paper_snapshot/ieeenat_fullname.bst)|当前main的必要小文件依赖；沿用原官方布局|
|11月3日19点汇报|[11页可编辑PPTX](paper_snapshot/UniDB_20261103_1900_Briefing.pptx)、[讲稿](paper_snapshot/UniDB_20261103_1900_Talk_Track_ZH.md)|这是19点中途快照，**不是午夜最终结果**；里面的仍在训练等措辞只描述当时|
|新增方法图|[Simulator](figures/SIMULATOR.drawio)、[方法](figures/METHODS.drawio)、[环境](figures/NPU_DEPENDENCIES.drawio)|可编辑draw.io及PNG；新consistency框明确为待实现|
|新增论文阅读列表|[PAPERS.md](PAPERS.md)、[references.bib](references.bib)|来源已核对；CGDB候选身份待接手方确认|

复制记录及SHA：[MATERIAL_COPIES.json](evidence/MATERIAL_COPIES.json)。未复制原输出目录中已经不被main引用的旧`paper/`草案支线，不将其重新变成正式论文结构。原项目`main_original_20261028.tex`等历史备份保持原处。

## 论文目前缺什么

当前main只导入摘要和Introduction/Related Work；**Method、Experiments、Conclusion未完成**。模板格式示例不是已写的方法或实验。旧草稿的递归扩散叙述也不意味着最新UniDB已经接入物理轨迹；后续应围绕最终真正实现的方法重写，并从冻结结果取表。

现有稿可在独立目录按原模板编译。本次是素材复制与依赖检查，不重编论文、不承诺新增投稿模板/政策合规性，也不确认转录中的“CPR”代表的准确投稿目标或截止。

## 结果表与图片从哪里取

- [V5 D0/A/B正式表](../../handoff/documents/MAIN_HANDOFF_FINAL_v2.md#5-v5的三阶段正式结果)：同Val1000、refinement OFF，注明末步与自身第一步。
- [UniDB最终实验说明](../../handoff/experiment/README_ZH.md)：两线best/latest、输入基线、五档四指标和真实终点。
- [历史版本与未决问题](../DECISIONS_FAILURES_AND_OPEN_ISSUES_ZH.md)：不得从跨版本表中推导单因素骨干优劣。
- [原始结果登记](../../handoff/registers/EVIDENCE_REGISTER.csv)：找对应CSV、SUMMARY、固定样例与配置身份。

后续论文分工、作者顺序与贡献声明应由相关人员确认；本材料不代作约定。
