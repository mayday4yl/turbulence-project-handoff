# V6 A 阶段完成报告

A 已完成累计6000次正式更新、259280次轨迹呈现。原六卡U1–2410与新五卡U2411–6000日志位置完整连续；Train10000/40K均覆盖。最后raw/EMA/Adam有限、计数6000、游标259280、五卡RNG与完整checkpoint实际SHA通过，全部A worker正常退出。

固定Val1000、4→3→2→1→0四步最终PSNR最佳为U1000：19.86120922 dB；Step1为20.99742496 dB，Step4−Step1为−1.13621573 dB。终点U6000四步为19.79950879 dB。四步递归效果仍低于Step1，属于负研究结果，未据此降低预算或改Loss/LR。B将按既定规则从A U1000 EMA初始化，不继承Pilot或终点较差权重。

最后checkpoint：/data2/yanglin/v6_diffuir_l_20261027/distributed/A_five_npu_formal6000_01/checkpoints/U006000.pt

SHA256：8ccedfa47a769c9cc6ac79f072f354bf26aaa2868d7ac671adcb0c9c6f1a4e45

best checkpoint：/data2/yanglin/v6_diffuir_l_20261027/distributed/A_authorized_formal6000_01/checkpoints/U001000.pt

SHA256：c3ae378bf96d66d531bc30db646fd8a621e85680474a6c324fe6aaa841b0668a

逐图固定验证、独立终审：execution_20261027_five_npu/A_COMPLETION_INDEPENDENT_AUDIT.json。B需通过完整工程/梯度/Benchmark/恢复/Pilot门槛后才正式训练；本报告不将尚未执行的B门槛标PASS。Test未参与。
