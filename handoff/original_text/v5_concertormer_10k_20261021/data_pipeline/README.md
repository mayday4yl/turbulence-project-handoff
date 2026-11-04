# V5-v1 数据链

CPU source 准备已完成；新 Y4 是否完成以数据根 `DATA_READY.json` 为准。

数据根：`/data2/yanglin/v5_concertormer_10k_20261021/data/v5_places_c50_s2000_v250_t250_g4_r4_20261021`。

## 固定协议

- Places 固定50类，clean train/val/test = 2000/250/250，每张4个独立原生 Gaussian raw-ξ，G=4，共8000/1000/1000个观测。
- 保留旧 canonical 的2450张clean及split；新增50张test每类1张，来自现存官方Places val archive中未进入旧扩展9200来源选择的图像。这里train/val/test是研究分割，不是Places官方分割。
- 所有来源ID、原图字节和预处理像素SHA唯一；联合pHash/dHash近重复筛查作为附加启发式，不能证明不存在全部语义相似或重裁切图片。
- 预处理代码逐字节复用旧已冻结版本：EXIF transpose、RGB、短边256 bicubic、中心crop、FP32 CHW除255；本次旧2450张全部重新解码预处理并逐像素核对一致。无在线增强。
- 新Y4全部通过已研究接收的V5 Simulator生成，保持FP32和既有物理语义；不读、不复用旧Y4，不clamp、不量化为PNG。
- raw-ξ不用于训练监督或网络输入，不落盘大数组。每样本/每kind保存独立CPU torch seed、固定形状、torch版本、raw字节SHA；生成采用原生 `torch.Generator(device='cpu')` + `torch.randn(dtype=float32)`。改变torch版本须显式重放核查，不能仅相信seed。

## 执行顺序

以下均在实验根运行。NPU卡号由主控实时分配；不得因代码示例擅自占卡。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TASK_QUEUE_ENABLE=0
PY=/data0/yanglin/envs/physical_bridge/bin/python
DATA=/data2/yanglin/v5_concertormer_10k_20261021/data/v5_places_c50_s2000_v250_t250_g4_r4_20261021
$PY data_pipeline/freeze_sources.py --root "$DATA"
$PY data_pipeline/generate.py --root "$DATA" --device npu:1
$PY data_pipeline/finalize.py --root "$DATA" --device npu:1
```

多个generator使用不同已分配NPU并发执行相同命令，按source `flock`动态认领；每clean的4个ξ复用 `prepare_image`。每张Y4和完成record原子写入，恢复时校验样本计划、生成身份及输出SHA。worker结束不等于整个数据READY；必须全部workers结束、完整finalize通过。

完整finalize检查2500 clean、10000 Y4、20000个raw哈希唯一；每split首尾共6样本重放raw与同backend渲染，并用首个train样本做CPU32/NPU32参考，沿用原冻结Gate的输出max-abs≤2e-5。test只进行身份、完整性与Simulator重放，不运行恢复网络、不计算选模指标。

## Loader

```python
from data_pipeline.dataset import V5Dataset
dataset = V5Dataset(DATA, 'train')
batch_item = dataset[0]
```

字段为 `clean`、`observation`（FP32 `[3,256,256]`）；`G`/`strength`=4；`sample_id`/`realization_id`；`source_id`；`realization_index`=0..3。不返回真实raw ξ。默认必须存在匹配的DATA_READY；`test`默认锁定。相同index无随机变换，便于确定性训练流和续训。

7项有界CPU回归覆盖原生RNG逐字节一致、重复调用确定性、未裁剪FP32保留、原子冻结防覆盖、source/manifest篡改拒绝、READY缺失拒绝、test锁定。已在本机torch2.9.1及de36 torch2.7.1+cpu分别通过；这是CPU接口验证，不代表恢复模型NPU已验证。

## 空间与耗时

新增clean+Y4数组理论约9.16GiB（另有原JPEG和元数据），当前source资产约1.87GiB；避免落盘约371.09GiB raw数组。de36本次检查/data2可用约4.7TiB。

旧Simulator合成fixture在NPU上prepared render约22–27ms/张，首次含冷启动约0.635s；该数字不包含CPU raw采样/hash、host-device传输、clean prepare或存盘。正式生成预计时间应使用worker `elapsed_seconds/made` 的实测，不用该局部计时推算训练耗时。
