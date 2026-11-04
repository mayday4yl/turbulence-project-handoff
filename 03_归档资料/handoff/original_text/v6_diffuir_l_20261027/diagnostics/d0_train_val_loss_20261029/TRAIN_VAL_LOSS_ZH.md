# D0 U95544 当前 Train / Val Loss

|记录|Loss|定义与范围|
|---|---:|---|
|Train最后一个更新|0.090749830|在线raw网络，四卡effective60；原生随机时间残差L1|
|Train最近100更新均值|0.089156272|同上|
|Train最近1000更新均值|0.089182109|同上|
|当前EMA Val残差L1|0.087682003|本次新测完整Val1000，250 Clean×4 realization；每图固定一次随机tau/noise|
|当前EMA Val三步Clean MAE|0.055116700|此前已完成的完整固定Val1000；三步输出与Clean的RGB[0,1] L1|

训练原生残差L1为mean(abs(pred_res_raw−(y−c)))，y=2Y−1、c=2C−1，系数1，无额外Clamp。Val残差评估调用同一个官方p_losses；真实每图1NFE，共1000NFE，0参数更新。每图tau、seed、损失和输入身份见VAL_RESIDUAL_PER_IMAGE.csv，源码、配置和完整命令另附。

原有验证历史记录PSNR/SSIM/LPIPS与三步Clean MAE，没有记录同目标Val残差Loss。图左只画真实在线Train历史和当前U95544 Val残差点，不伪造Val残差历史曲线；图右画既有完整Val1000的三步Clean MAE曲线。Train在线raw与Val终点EMA的权重及随机采样口径不同，不能仅凭损失差判断过拟合。

有效训练链U14928–95544，共80617更新；420次未保存尾重放不重复计入。前1000更新Train均值0.093141340，最近1000为0.089182109，下降4.25%。这是旧联合Loss checkpoint切换原生Loss后的阶段，不与U14927之前联合Loss绝对值拼接比较。

当前Val residual单图标准差0.040736693，tau范围3–997。本次单个固定seed realization的平均值，不是对每张图遍历1000个tau后的精确期望。原checkpoint SHA 1fd16df15c64cc496bda89ccb32fd6a9413c9de8e59942fc13cbb6592e757cc3 完整保持；四卡NPU4–7 worker均退出，原STOP保留，D0/A/B没有恢复训练。新前向评估进程墙钟57.92秒（包含加载与校验）。
