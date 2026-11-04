# 单源图局部 C–ξ 补偿诊断

36 段固定预算搜索已完成，保存数组已逐个通过原 simulator 独立重放和约束复核，并通过本报告的数据、哈希与候选选择审计。每段 600 次诊断变量更新，总计 21,600 次；网络训练更新为 0。

只覆盖 Train source 01、原真实 raw ξ、G=1/2/3/4 和三个预先固定初始化。C 的距离固定为 RMS(C′−C_GT)=0.010001，RGB 限制 [0,1]；ξ 使用登记的局部变化边界和统计筛查。目标只含观测 MSE/1e−8，没有 clean loss；GT 用于初始化、可行域与最终评价。这不是盲恢复算法的测试集性能。

fixed_xi 只更新 C；xi_only 固定初始 C′、只更新两块 raw ξ；joint 同时更新 C 和两块 ξ。固定 ξ 达到阈值只记作对照，不能计为 blind witness。每段只从初始化及每 50 次更新的 13 个审核候选中取最小观测 MSE；不是 601 个状态或全局最优。

优先使用有合法且 ξ 变化达到门槛的 best；若无，则显示 best_legal 并明确 NO_ELIGIBLE。xi_only 与 joint 都可独立成为 blind witness，但二者回答的问题不同。

## 全部 36 段

| G | seed | mode | selected step | finding | obs RMS | clean RMS | PSNR | SSIM | LPIPS | ξ combined/Z/T RMS | prior |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 2026092901 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000152409 | 0.010001 | 39.9991 | 0.991613 | 0.00350726 | 0/0/0 | True |
| 1 | 2026092901 | xi_only | 500 | NOT_FOUND | 0.000367543 | 0.010001 | 39.9991 | 0.993762 | 0.00688117 | 0.0512065/0.0526073/0.00213388 | True |
| 1 | 2026092901 | joint | 600 | BLIND_WITNESS | 8.66405e-05 | 0.010001 | 39.9991 | 0.99358 | 0.00631287 | 0.0197161/0.020256/0.00050439 | True |
| 1 | 2026092902 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000131084 | 0.010001 | 39.9991 | 0.992989 | 0.00397633 | 0/0/0 | True |
| 1 | 2026092902 | xi_only | 600 | NOT_FOUND | 0.000267173 | 0.010001 | 39.9991 | 0.995064 | 0.00720287 | 0.0452369/0.0464733/0.00231739 | True |
| 1 | 2026092902 | joint | 550 | BLIND_WITNESS | 8.53178e-05 | 0.010001 | 39.9991 | 0.995001 | 0.00549298 | 0.0169396/0.0174035/0.000449569 | True |
| 1 | 2026092903 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000176199 | 0.010001 | 39.9991 | 0.993256 | 0.00358323 | 0/0/0 | True |
| 1 | 2026092903 | xi_only | 550 | NOT_FOUND | 0.000314592 | 0.010001 | 39.9991 | 0.995008 | 0.00659814 | 0.0463846/0.0476545/0.0014491 | True |
| 1 | 2026092903 | joint | 550 | BLIND_WITNESS | 8.48911e-05 | 0.010001 | 39.9991 | 0.995367 | 0.00517027 | 0.0177212/0.0182067/0.000313359 | True |
| 2 | 2026092901 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000140116 | 0.010001 | 39.9991 | 0.99171 | 0.00758135 | 0/0/0 | True |
| 2 | 2026092901 | xi_only | 500 | NOT_FOUND | 0.000364498 | 0.010001 | 39.9991 | 0.993762 | 0.00688117 | 0.0260319/0.0267436/0.00126223 | True |
| 2 | 2026092901 | joint | 450 | NOT_FOUND | 0.000115874 | 0.010001 | 39.9991 | 0.993976 | 0.00682797 | 0.0139527/0.0143349/0.000221381 | True |
| 2 | 2026092902 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000109416 | 0.010001 | 39.9991 | 0.993255 | 0.00781981 | 0/0/0 | True |
| 2 | 2026092902 | xi_only | 600 | NOT_FOUND | 0.000231497 | 0.010001 | 39.9991 | 0.995064 | 0.00720287 | 0.0243576/0.0250225/0.00152086 | True |
| 2 | 2026092902 | joint | 350 | NOT_FOUND | 0.00012259 | 0.010001 | 39.9991 | 0.995128 | 0.00658705 | 0.0127586/0.013108/0.000288718 | True |
| 2 | 2026092903 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000147188 | 0.010001 | 39.9991 | 0.993514 | 0.00758357 | 0/0/0 | True |
| 2 | 2026092903 | xi_only | 600 | NOT_FOUND | 0.000256097 | 0.010001 | 39.9991 | 0.995008 | 0.00659814 | 0.0263328/0.0270528/0.00125157 | True |
| 2 | 2026092903 | joint | 550 | BLIND_WITNESS | 8.77852e-05 | 0.010001 | 39.9991 | 0.995554 | 0.00551062 | 0.014335/0.0147276/0.000260929 | True |
| 3 | 2026092901 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000122791 | 0.010001 | 39.9991 | 0.991756 | 0.00948289 | 0/0/0 | True |
| 3 | 2026092901 | xi_only | 550 | NOT_FOUND | 0.000237447 | 0.010001 | 39.9991 | 0.993762 | 0.00688117 | 0.0190666/0.0195872/0.00116049 | True |
| 3 | 2026092901 | joint | 500 | NOT_FOUND | 0.000117421 | 0.010001 | 39.9991 | 0.993884 | 0.00670591 | 0.0132133/0.0135751/0.000362192 | True |
| 3 | 2026092902 | fixed_xi | 600 | FIXED_XI_THRESHOLD_MET | 9.53758e-05 | 0.010001 | 39.9991 | 0.993542 | 0.00977634 | 0/0/0 | True |
| 3 | 2026092902 | xi_only | 600 | NOT_FOUND | 0.000221258 | 0.010001 | 39.9991 | 0.995064 | 0.00720287 | 0.0188671/0.019382/0.00119902 | True |
| 3 | 2026092902 | joint | 350 | NOT_FOUND | 0.000137211 | 0.010001 | 39.9991 | 0.99491 | 0.00683486 | 0.0114067/0.0117192/0.000210752 | True |
| 3 | 2026092903 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000123534 | 0.010001 | 39.9991 | 0.993671 | 0.00979934 | 0/0/0 | True |
| 3 | 2026092903 | xi_only | 600 | NOT_FOUND | 0.000243657 | 0.010001 | 39.9991 | 0.995008 | 0.00659814 | 0.0202122/0.0207651/0.000874606 | True |
| 3 | 2026092903 | joint | 450 | NOT_FOUND | 0.000109462 | 0.010001 | 39.9991 | 0.995373 | 0.00607545 | 0.0126347/0.0129807/0.000275142 | True |
| 4 | 2026092901 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000111199 | 0.010001 | 39.9991 | 0.990818 | 0.00991228 | 0/0/0 | True |
| 4 | 2026092901 | xi_only | 550 | NOT_FOUND | 0.000232477 | 0.010001 | 39.9991 | 0.993762 | 0.00688117 | 0.0163328/0.0167792/0.000861687 | True |
| 4 | 2026092901 | joint | 600 | NOT_FOUND | 0.000178778 | 0.010001 | 39.9991 | 0.993617 | 0.006421 | 0.0134891/0.0138584/0.00039169 | True |
| 4 | 2026092902 | fixed_xi | 600 | FIXED_XI_THRESHOLD_MET | 8.74007e-05 | 0.010001 | 39.9991 | 0.993189 | 0.0100183 | 0/0/0 | True |
| 4 | 2026092902 | xi_only | 600 | NOT_FOUND | 0.000218699 | 0.010001 | 39.9991 | 0.995064 | 0.00720287 | 0.0164232/0.0168719/0.000908108 | True |
| 4 | 2026092902 | joint | 600 | NOT_FOUND | 0.000116373 | 0.010001 | 39.9991 | 0.994641 | 0.00701401 | 0.0126449/0.0129911/0.00033671 | True |
| 4 | 2026092903 | fixed_xi | 600 | FIXED_XI_THRESHOLD_NOT_MET | 0.000109051 | 0.010001 | 39.9991 | 0.993428 | 0.00995363 | 0/0/0 | True |
| 4 | 2026092903 | xi_only | 600 | NOT_FOUND | 0.000236611 | 0.010001 | 39.9991 | 0.995008 | 0.00659814 | 0.0170819/0.0175491/0.000743074 | True |
| 4 | 2026092903 | joint | 450 | NOT_FOUND | 0.000177058 | 0.010001 | 39.9991 | 0.995052 | 0.00595546 | 0.0127388/0.0130876/0.000311153 | True |

## 每 G 的 selected-best joint（另列，非固定展示 seed）

每段先取 eligible best；无 eligible 才取 best_legal。跨 seed 同样先在 eligible 选中解里取最低残差；全部无 eligible 时取合法 fallback。

| G | selected seed | step | finding | obs RMS | clean RMS | PSNR | SSIM | LPIPS | same-C′ fixed ξ RMS | improvement |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2026092903 | 550 | BLIND_WITNESS | 8.48911e-05 | 0.010001 | 39.9991 | 0.995367 | 0.00517027 | 0.0028822 | 33.9517 |
| 2 | 2026092903 | 550 | BLIND_WITNESS | 8.77852e-05 | 0.010001 | 39.9991 | 0.995554 | 0.00551062 | 0.00222171 | 25.3085 |
| 3 | 2026092903 | 450 | NOT_FOUND | 0.000109462 | 0.010001 | 39.9991 | 0.995373 | 0.00607545 | 0.00189675 | 17.3279 |
| 4 | 2026092902 | 600 | NOT_FOUND | 0.000116373 | 0.010001 | 39.9991 | 0.994641 | 0.00701401 | 0.00163511 | 14.0506 |

ξ RMS 为两块元素平方差求和后除合计元素数再开方，不能把两块 mean 相加。完整相对 L2、每通道统计和候选详情见 SUMMARY.json / candidate_details.json。统计门槛通过只表示这些检查没有拒绝；不能证明经过观测选择的 ξ 仍是独立同分布 Gaussian，也不能证明它是正确物理 realization。

## 同一个 C′ 的 ξ 补偿归因

| G | seed | mode | fixed ξ same-C′ RMS | changed ξ RMS | improvement | cancellation cos | ξ/clean contribution norm | effective Z RMS | displacement RMS | strong compensation |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2026092901 | fixed_xi | 0.000152409 | 0.000152409 | 1 | — | 0 | 0 | 0 | False |
| 1 | 2026092901 | xi_only | 0.00422781 | 0.000367543 | 11.5029 | -0.996226 | 0.991333 | 0.258051 | 0.0555338 | False |
| 1 | 2026092901 | joint | 0.00277283 | 8.66405e-05 | 32.0039 | -0.999512 | 0.999081 | 0.0861345 | 0.0177581 | True |
| 1 | 2026092902 | fixed_xi | 0.000131084 | 0.000131084 | 1 | — | 0 | 0 | 0 | False |
| 1 | 2026092902 | xi_only | 0.00392998 | 0.000267173 | 14.7095 | -0.99769 | 0.99502 | 0.246436 | 0.0274173 | False |
| 1 | 2026092902 | joint | 0.00261121 | 8.53178e-05 | 30.6057 | -0.999466 | 0.999426 | 0.122878 | 0.00919879 | True |
| 1 | 2026092903 | fixed_xi | 0.000176199 | 0.000176199 | 1 | — | 0 | 0 | 0 | False |
| 1 | 2026092903 | xi_only | 0.00402882 | 0.000314592 | 12.8065 | -0.996953 | 0.993361 | 0.257043 | 0.0216451 | False |
| 1 | 2026092903 | joint | 0.0028822 | 8.48911e-05 | 33.9517 | -0.999566 | 0.999741 | 0.109405 | 0.00551818 | True |
| 2 | 2026092901 | fixed_xi | 0.000140116 | 0.000140116 | 1 | — | 0 | 0 | 0 | False |
| 2 | 2026092901 | xi_only | 0.00344242 | 0.000364498 | 9.44427 | -0.994378 | 0.994541 | 0.17157 | 0.0147854 | False |
| 2 | 2026092901 | joint | 0.00210604 | 0.000115874 | 18.1753 | -0.998485 | 0.99838 | 0.0883827 | 0.00288785 | False |
| 2 | 2026092902 | fixed_xi | 0.000109416 | 0.000109416 | 1 | — | 0 | 0 | 0 | False |
| 2 | 2026092902 | xi_only | 0.00317805 | 0.000231497 | 13.7282 | -0.997344 | 0.99601 | 0.199511 | 0.0162482 | False |
| 2 | 2026092902 | joint | 0.00205214 | 0.00012259 | 16.7398 | -0.998214 | 0.99825 | 0.121099 | 0.00564053 | False |
| 2 | 2026092903 | fixed_xi | 0.000147188 | 0.000147188 | 1 | — | 0 | 0 | 0 | False |
| 2 | 2026092903 | xi_only | 0.00320613 | 0.000256097 | 12.5192 | -0.996807 | 0.99465 | 0.22212 | 0.0111806 | False |
| 2 | 2026092903 | joint | 0.00222171 | 8.77852e-05 | 25.3085 | -0.999219 | 0.999382 | 0.106958 | 0.00223703 | True |
| 3 | 2026092901 | fixed_xi | 0.000122791 | 0.000122791 | 1 | — | 0 | 0 | 0 | False |
| 3 | 2026092901 | xi_only | 0.00294842 | 0.000237447 | 12.4172 | -0.996754 | 0.994513 | 0.179384 | 0.0111884 | False |
| 3 | 2026092901 | joint | 0.00192538 | 0.000117421 | 16.3973 | -0.998139 | 0.998944 | 0.13193 | 0.00381803 | False |
| 3 | 2026092902 | fixed_xi | 9.53758e-05 | 9.53758e-05 | 1 | — | 0 | 0 | 0 | False |
| 3 | 2026092902 | xi_only | 0.00275514 | 0.000221258 | 12.4522 | -0.99677 | 0.996437 | 0.207146 | 0.0104223 | False |
| 3 | 2026092902 | joint | 0.00172734 | 0.000137211 | 12.589 | -0.996846 | 1.00032 | 0.133893 | 0.00379635 | False |
| 3 | 2026092903 | fixed_xi | 0.000123534 | 0.000123534 | 1 | — | 0 | 0 | 0 | False |
| 3 | 2026092903 | xi_only | 0.00273253 | 0.000243657 | 11.2146 | -0.996018 | 0.994506 | 0.239846 | 0.00837223 | False |
| 3 | 2026092903 | joint | 0.00189675 | 0.000109462 | 17.3279 | -0.998334 | 0.999402 | 0.108444 | 0.00172364 | False |
| 4 | 2026092901 | fixed_xi | 0.000111199 | 0.000111199 | 1 | — | 0 | 0 | 0 | False |
| 4 | 2026092901 | xi_only | 0.002616 | 0.000232477 | 11.2527 | -0.996046 | 0.993852 | 0.177826 | 0.00660384 | False |
| 4 | 2026092901 | joint | 0.00180208 | 0.000178778 | 10.08 | -0.995085 | 1.00112 | 0.133625 | 0.00245928 | False |
| 4 | 2026092902 | fixed_xi | 8.74007e-05 | 8.74007e-05 | 1 | — | 0 | 0 | 0 | False |
| 4 | 2026092902 | xi_only | 0.0024978 | 0.000218699 | 11.4212 | -0.99616 | 0.99623 | 0.179028 | 0.00760625 | False |
| 4 | 2026092902 | joint | 0.00163511 | 0.000116373 | 14.0506 | -0.997464 | 0.998219 | 0.112899 | 0.00308318 | False |
| 4 | 2026092903 | fixed_xi | 0.000109051 | 0.000109051 | 1 | — | 0 | 0 | 0 | False |
| 4 | 2026092903 | xi_only | 0.00245224 | 0.000236611 | 10.364 | -0.995337 | 0.993008 | 0.223957 | 0.00644761 | False |
| 4 | 2026092903 | joint | 0.00180284 | 0.000177058 | 10.1822 | -0.995175 | 0.999474 | 0.186465 | 0.00168855 | False |

cos 定义：a=F(C′,ξ_GT)−F(C_GT,ξ_GT)，b=F(C′,ξ′)−F(C′,ξ_GT)。接近 −1 表示两项方向相反，但需要同时检查绝对残差和幅度比。strong compensation 使用原计划门槛：同 C′ 固定 ξ 残差 > 0.001、blind 残差 ≤ 0.0001、改进 ≥ 10 倍，并且候选本身通过 blind witness 约束。

## 旧固定 ξ 方向：原幅度与本轮同幅度

| G | kind | clean RMS | single-sided obs RMS | PSNR | SSIM | LPIPS | current single-side threshold | old double-sided label |
|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | old_original_amplitude | 0.0496901 | 0.00014889 | 26.0746 | 0.738017 | 0.0490453 | False | NOT_FOUND (unchanged) |
| 1 | scaled_same_amplitude | 0.010001 | 2.9967e-05 | 39.9991 | 0.978484 | 0.000604542 | True | NOT_FOUND (unchanged) |
| 2 | old_original_amplitude | 0.0562029 | 0.000157956 | 25.0048 | 0.698915 | 0.104252 | False | NOT_FOUND (unchanged) |
| 2 | scaled_same_amplitude | 0.010001 | 2.8108e-05 | 39.9991 | 0.978641 | 0.00113981 | True | NOT_FOUND (unchanged) |
| 3 | old_original_amplitude | 0.0561299 | 0.000148401 | 25.0161 | 0.698263 | 0.125972 | False | NOT_FOUND (unchanged) |
| 3 | scaled_same_amplitude | 0.010001 | 2.64418e-05 | 39.9991 | 0.978684 | 0.00154705 | True | NOT_FOUND (unchanged) |
| 4 | old_original_amplitude | 0.0586831 | 0.000151892 | 24.6297 | 0.681135 | 0.144515 | False | NOT_FOUND (unchanged) |
| 4 | scaled_same_amplitude | 0.010001 | 2.58862e-05 | 39.9991 | 0.97842 | 0.00152006 | True | NOT_FOUND (unchanged) |

旧报告较大扰动使用两侧 F(C+)−F(C−) 差异；这里另用原 simulator 实测单侧 F(C′)−Y，并缩到相同 clean 幅度。旧 NOT_FOUND 不追溯改写。缩幅带来的变化不能归因于 ξ 补偿。

## Joint 三个初始化的所有两两比较

| G | seeds | clean pair RMS | pair PSNR | pair SSIM | pair LPIPS | render pair RMS | residual ratio | within 2× | C separation ≥.005 | classification |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 2026092901 / 2026092902 | 0.0110353 | 39.1443 | 0.993445 | 0.00503528 | 0.000107671 | 1.0155 | True | True | SEPARATED_COMPARABLE; BOTH_BLIND_WITNESSES |
| 1 | 2026092901 / 2026092903 | 0.0094146 | 40.524 | 0.994518 | 0.00432437 | 9.68095e-05 | 1.02061 | True | True | SEPARATED_COMPARABLE; BOTH_BLIND_WITNESSES |
| 1 | 2026092902 / 2026092903 | 0.0117711 | 38.5837 | 0.994255 | 0.00334534 | 0.000111478 | 1.00503 | True | True | SEPARATED_COMPARABLE; BOTH_BLIND_WITNESSES |
| 2 | 2026092901 / 2026092902 | 0.0109966 | 39.1748 | 0.994091 | 0.00576577 | 0.000138638 | 1.05796 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 2 | 2026092901 / 2026092903 | 0.00953031 | 40.4179 | 0.994987 | 0.00496367 | 0.000113324 | 1.31997 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 2 | 2026092902 / 2026092903 | 0.0115441 | 38.7528 | 0.99481 | 0.00373315 | 0.000127961 | 1.39648 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 3 | 2026092901 / 2026092902 | 0.0108742 | 39.272 | 0.993938 | 0.00609272 | 0.000148017 | 1.16854 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 3 | 2026092901 / 2026092903 | 0.00956446 | 40.3868 | 0.994868 | 0.00520434 | 0.000125763 | 1.0727 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 3 | 2026092902 / 2026092903 | 0.0114243 | 38.8434 | 0.994616 | 0.0043601 | 0.000148706 | 1.2535 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 4 | 2026092901 / 2026092902 | 0.0107373 | 39.3821 | 0.993909 | 0.00635307 | 0.000168562 | 1.53625 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 4 | 2026092901 / 2026092903 | 0.00952907 | 40.419 | 0.994703 | 0.00527173 | 0.000229032 | 1.00971 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |
| 4 | 2026092902 / 2026092903 | 0.0112918 | 38.9447 | 0.994436 | 0.00441794 | 0.000194733 | 1.52147 | True | True | SEPARATED_COMPARABLE; NOT_BOTH_WITNESSES |

两解残差在 2 倍以内仅表示相对可比；还要看绝对残差是否达到门槛。只有两端都为合格 blind witness 时，才额外标注 BOTH_BLIND_WITNESSES。不同初始化落在不同 C′，不自动构成两张自然照片的歧义证据。

## 固定图与选优图

主图预定 joint seed 2026092901，各 G 均使用本段的规定候选选择，避免按画面挑 seed。

[图像原位置：固定 joint seed 的纵向对照](<../../../../NOT_INCLUDED.md#ref-34764ac93493980c93a0>)

[图像原位置：固定范围差值图](<../../../../NOT_INCLUDED.md#ref-f1f5a5ba6d06630b24d8>)

另图明确按每 G 的 joint 三个 seed 中合法选中解的观测残差最低者展示；若存在 eligible，优先 eligible，否则显示合法 fallback 并标 NO_ELIGIBLE。

[图像原位置：按观测残差选择的 joint 对照](<../../../../NOT_INCLUDED.md#ref-d8d8c51d4b2c5184af28>)

## 证据与复现

PSNR/SSIM/LPIPS 从完整 FP32 NPY 计算，无 clamp、resize、量化或逐图归一化。PNG 仅展示时 clip 到 [0,1]。固定 seed 不承诺 CUDA grid_sample 反向优化轨迹逐位一致；本次 witness 结论由保存候选的真实 simulator forward 重放独立确认。指标实现、版本和资产在 METRIC_SPEC.json；所有浮点数组 SHA256 在 ARRAY_HASH_MANIFEST.json。SEARCH/CONFIG/SOURCE_RECORD、36 段代码身份、每段初始数组、600 条更新和 13 候选均核验；同 seed 的 C 初值跨 G/mode 保持相同。

```sh
# Run only in a fresh independent copy; existing results refuse overwrite. Do not duplicate an active run.
cd /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_blind_ambiguity_20260929
CUDA_VISIBLE_DEVICES=GPU-83edf451-9cae-c2a0-c9a6-f344b7bb2673 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 flock -n /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train_20260929/protocol/gpu5.lock /home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python -u /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_blind_ambiguity_20260929/scripts/search.py --phase prior
CUDA_VISIBLE_DEVICES=GPU-83edf451-9cae-c2a0-c9a6-f344b7bb2673 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 flock -n /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train_20260929/protocol/gpu5.lock /home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python -u /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_blind_ambiguity_20260929/scripts/baseline.py
CUDA_VISIBLE_DEVICES=GPU-83edf451-9cae-c2a0-c9a6-f344b7bb2673 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 flock -n /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train_20260929/protocol/gpu5.lock /home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python -u /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_blind_ambiguity_20260929/scripts/run_search_all.py
CUDA_VISIBLE_DEVICES=GPU-83edf451-9cae-c2a0-c9a6-f344b7bb2673 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 flock -n /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train_20260929/protocol/gpu5.lock /home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python -u /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_blind_ambiguity_20260929/scripts/post_validate.py
CUDA_VISIBLE_DEVICES=GPU-83edf451-9cae-c2a0-c9a6-f344b7bb2673 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 flock -n /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_train_20260929/protocol/gpu5.lock /home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python -u /home/zhuzhiyu/yanglin/turbulence_ir/versions/v4_noise_bridge/implementations/v4_blind_ambiguity_20260929/scripts/build_report.py
```

结论边界：这是指定约束、初始化和预算内的局部连续 RGB/ξ 搜索。未找到表示本轮搜索未找到，不证明唯一性；找到表示指定数值容差下的候选，不是精确碰撞、自然图像分布上的非唯一性证明，也不能单独解释网络拟合或泛化。所有原始数据、历史结果、simulator 和 V4 源码保持不变。
