# 会议补充材料：实际验证范围

日期：2026-11-04；这是本次接手材料/运行示例的验证，不是新的正式恢复实验。共0次optimizer更新，没有重启旧controller、没有修改冻结模拟器/训练代码、没有停止外部服务。

## 在de36实际执行

|检查|结果|证据与边界|
|---|---|---|
|解释器、依赖与资产路径|PASS|[环境快照](evidence/ENVIRONMENT_20261104.json)、[路径与实际示例SHA](evidence/REMOTE_HELPER_AND_PATHS.txt)；主入口Python3.10.12，torch实际来自physical_bridge；三条.pth已记录|
|模拟器CPU|PASS，19.4秒|[JSON](evidence/SIMULATOR_CPU.json) · [原日志](evidence/SIMULATOR_CPU.log)；六档前向，g0端点，g2的C/raw-ξ一阶反向|
|模拟器NPU初次|FAIL，原因已修复|私有Ascend cache目录不存在；[失败日志](evidence/SIMULATOR_NPU_INITIAL_CACHE_FAILURE.log)。只修复新增demo的目录创建，没有修改原模型|
|模拟器NPU重跑|PASS，450.1秒|[JSON](evidence/SIMULATOR_NPU.json) · [日志](evidence/SIMULATOR_NPU.log)；物理卡0，同六档与一阶反向；首次算子编译等待包含在耗时内|
|UniDB CPU EMA严格加载＋1次前向|PASS，34.1秒|[JSON](evidence/UNIDB_CPU.json) · [日志](evidence/UNIDB_CPU.log)；不是恢复图|
|UniDB NPU原生100步恢复|PASS，22.9秒|[JSON](evidence/UNIDB_NPU_RESTORE.json) · [日志](evidence/UNIDB_NPU_RESTORE.log)；U67180/EMA、val_s0040_r00、G4、B1、实际NFE100|
|运行后资源|本次计算进程已退出|[运行前](evidence/NPU_RESOURCE_BEFORE.txt) · [运行后](evidence/NPU_RESOURCE_AFTER.txt)；物理卡1原服务仍在|

上述耗时仅为本次实测，CPU与NPU包含的初始化/编译开销不同，不构成性能比较。输出写入新建`/tmp/yanglin_handoff_20261104_1733/`，只是演示临时目录，不是新正式数据或永久资产位置。接收者重跑请用runbook给出的自有新目录。

一阶梯度检查仅证明一个合成点上C和两块raw-ξ的梯度有限且非零，不覆盖原Gate的FAIL、有限差分、高阶导数、关于g的导数或完整四步训练。B1恢复的随机流不同于正式B4/source评价，没有为它计算或添加新benchmark结论。

## 本次复核补正

针对图中裸下划线的反馈，公式已改成真正的上下标，并补充图中符号与代码变量的对照。源码、命令和真实路径中的下划线保留。此前将普通字符下标视为可接受排版的判断已撤销，修正记录附在图稿检查日志。

`inspect`原先仅查9项基础路径，现已扩大为实际依赖/小型配置/首样本检查，实机包含选定checkpoint时共38项通过，见[记录](evidence/PREFLIGHT_V2.json)。新增检查只用stdlib，不占卡。缺文件、缺嵌套字段、非法JSON三个失败夹具均清晰返回FAIL；有效结构夹具返回PASS。

原CPU/NPU运行对应的[脚本原件](evidence/DEMO_EXECUTED_20261104.py)已保存。当前脚本只改inspect分支；从输出目录/设备初始化开始的全部计算分支逐字节未变，见[身份与差异核对](evidence/DEMO_REVISION_IDENTITY.json)。本轮未重跑模型、未新增训练。

## 文档、源码与图解

- 新代码导航核对13个原文件和仓库快照的SHA均一致：[CODE_MAP.json](evidence/CODE_MAP.json)。每个关键函数提供快照行号和原位置。
- 原交接包561项和源码快照754项经`tools/verify_repository.py`校验；没有改写原冻结包。机器记录见[仓库核验](evidence/REPOSITORY_VERIFICATION.json)。
- 论文草稿、19点汇报与历史图按原字节复制：[素材复制记录](evidence/MATERIAL_COPIES.json)；补带原Simulator Gate：[原件身份](evidence/ORIGINAL_GATE_COPY.json)。原Gate内部历史相对链接仍以原目录为基准。
- 三张图均保留可编辑draw.io和PNG，XML结构/视觉预检、三轮渲染检查及评分见[图稿核验](figures/defect-log.md)。
- 离线HTML的强度滑块、blur/warp按钮、章节跳转展开已在Chrome实际检查；静态本地资源/文档链接由补充校验脚本检查。网页载荷不依赖外部脚本，不使用在线推理。
- 新增Python脚本做语法检查；运行脚本检查了缺少卡分配标记及重复输出目录拒绝。没有新增单元测试去重复低风险排版实现。

完整性入口：

```bash
python3 tools/verify_repository.py
python3 tools/receiver/verify_supplement.py
```

完整性校验只验证随包文件身份和本地链接，不证明远端资产永久在线或接手者已经获得访问权。资料包不包含训练数据、完整模型载荷或完整官方vendor；在其他机器运行需要按登记补齐。

## 仍不应写成已经完成

接手老师的独立登录/重跑、具体consistency代码接入、JVP/高阶梯度支持、新训练恢复适配、真实数据恢复改进、论文Method/Experiments/Conclusion写完，均未由本次资料补充完成。
