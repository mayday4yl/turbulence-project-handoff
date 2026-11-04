# G1 数据构建与独立验收

已通过真实文件验收：Train10000独立Clean，40000个G4配对；原2000源和8000配对只读复用，新增8000源及32000个G4文件。固定Val250×4和Test250×4身份不变。

官方归档：26103685120 bytes，SHA256 `0ade5d7d38c682aa0203014a2b99ad3bf608bf1de0f151dc0d80efba54660a0a`。这是本次完整文件校验值，未冒称官方公开checksum。归档实际图像成员1803460；固定排名选择后排除5个候选，详见SOURCE_REJECTIONS.jsonl。

| 类别ID | 类别 | 原Train | 新Train | 总Clean | realization |
|---:|---|---:|---:|---:|---:|
| 1 | /a/airplane_cabin | 40 | 160 | 200 | 800 |
| 3 | /a/alcove | 40 | 160 | 200 | 800 |
| 5 | /a/amphitheater | 40 | 160 | 200 | 800 |
| 7 | /a/amusement_park | 40 | 160 | 200 | 800 |
| 11 | /a/arcade | 40 | 160 | 200 | 800 |
| 13 | /a/archaelogical_excavation | 40 | 160 | 200 | 800 |
| 21 | /a/art_studio | 40 | 160 | 200 | 800 |
| 31 | /b/bakery/shop | 40 | 160 | 200 | 800 |
| 38 | /b/banquet_hall | 40 | 160 | 200 | 800 |
| 40 | /b/barn | 40 | 160 | 200 | 800 |
| 49 | /b/beach_house | 40 | 160 | 200 | 800 |
| 57 | /b/boardwalk | 40 | 160 | 200 | 800 |
| 61 | /b/booth/indoor | 40 | 160 | 200 | 800 |
| 67 | /b/building_facade | 40 | 160 | 200 | 800 |
| 101 | /c/conference_center | 40 | 160 | 200 | 800 |
| 107 | /c/cottage | 40 | 160 | 200 | 800 |
| 118 | /d/desert_road | 40 | 160 | 200 | 800 |
| 123 | /d/doorway/outdoor | 40 | 160 | 200 | 800 |
| 132 | /e/embassy | 40 | 160 | 200 | 800 |
| 134 | /e/entrance_hall | 40 | 160 | 200 | 800 |
| 135 | /e/escalator/indoor | 40 | 160 | 200 | 800 |
| 137 | /f/fabric_store | 40 | 160 | 200 | 800 |
| 140 | /f/field/cultivated | 40 | 160 | 200 | 800 |
| 142 | /f/field_road | 40 | 160 | 200 | 800 |
| 154 | /f/fountain | 40 | 160 | 200 | 800 |
| 156 | /g/garage/indoor | 40 | 160 | 200 | 800 |
| 159 | /g/gazebo/exterior | 40 | 160 | 200 | 800 |
| 162 | /g/gift_shop | 40 | 160 | 200 | 800 |
| 166 | /g/greenhouse/outdoor | 40 | 160 | 200 | 800 |
| 167 | /g/grotto | 40 | 160 | 200 | 800 |
| 173 | /h/hayfield | 40 | 160 | 200 | 800 |
| 175 | /h/highway | 40 | 160 | 200 | 800 |
| 176 | /h/home_office | 40 | 160 | 200 | 800 |
| 180 | /h/hot_spring | 40 | 160 | 200 | 800 |
| 185 | /i/ice_cream_parlor | 40 | 160 | 200 | 800 |
| 203 | /k/kitchen | 40 | 160 | 200 | 800 |
| 215 | /l/living_room | 40 | 160 | 200 | 800 |
| 234 | /m/mountain_snowy | 40 | 160 | 200 | 800 |
| 239 | /n/natural_history_museum | 40 | 160 | 200 | 800 |
| 241 | /n/nursing_home | 40 | 160 | 200 | 800 |
| 254 | /p/park | 40 | 160 | 200 | 800 |
| 300 | /s/shoe_shop | 40 | 160 | 200 | 800 |
| 312 | /s/stadium/baseball | 40 | 160 | 200 | 800 |
| 314 | /s/stadium/soccer | 40 | 160 | 200 | 800 |
| 341 | /t/tundra | 40 | 160 | 200 | 800 |
| 343 | /u/utility_room | 40 | 160 | 200 | 800 |
| 344 | /v/valley | 40 | 160 | 200 | 800 |
| 356 | /w/watering_hole | 40 | 160 | 200 | 800 |
| 361 | /w/windmill | 40 | 160 | 200 | 800 |
| 363 | /y/youth_hostel | 40 | 160 | 200 | 800 |

原2500源图的前处理像素/文件/感知hash均已独立重放。新增原图保留官方train路径、类别与哈希，原source/Val/Test无重命名或替换。新旧源的精确及固定联合pHash≤4、dHash≤4候选排除均记录；未用模型效果挑数据。

原生ξ以CPU Torch 2.7.1 FP32、每种噪声独立seed及版本身份复现，避免持久化全量约1.45TiB的raw张量。新旧各8条代表性训练轨迹完成独立进程重放，原始ξ哈希、g0 identity、g2有限及g4重放均有证据。此项不冒充A/B尚未实现的训练图验收。

| 来源 | 亮度 mean/std | 对比度 mean/std | Laplacian variance mean/std | G4输入PSNR mean/std |
|---|---:|---:|---:|---:|
| original_v5 | 0.445004 / 0.107501 | 0.225853 / 0.053974 | 0.022781 / 0.017467 | 18.713817 / 2.880180 |
| new_official_train | 0.445163 / 0.110995 | 0.226439 / 0.051989 | 0.023764 / 0.018415 | 18.639259 / 2.874067 |

旧源历史来自官方validation池，新源来自官方train-standard；类别配额一致不等于图像分布完全一致。完整RGB均值、分位数与逐图统计见 DOMAIN_SHIFT_STATISTICS.json / SOURCE_DISTRIBUTION_PER_IMAGE.jsonl，不据此调整固定Val/Test。

Test只参与源身份、前处理与泄漏审计；没有Test恢复推理、恢复指标或模型选择。最终40K的每个文件均实际SHA/形状/dtype/finite核验，计划项未算作生成项。

数据身份 SHA256：`0eb85bfd1d783b25ec3145d65de02da71176bce7c880dde7e183c8aa140b53d0`。四份清单为 V6_TRAIN10000_SOURCE.jsonl、V6_TRAJECTORY40000.jsonl、V6_G4_PAIRS40000.jsonl、V6_DATA_FINGERPRINT.json；所有数据为独立V6目录或只读原文件引用。
