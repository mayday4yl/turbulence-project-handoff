# V0.3-half 数据验收

固定 seed：2026091504。Places365 类别 ID 与原 source metadata 一致，采用确定性分层抽样，按 source 精确减半。

| Split | Clean sources | Trajectories | Six-level degraded states |
|---|---:|---:|---:|
| Train | 1000 | 4000 | 24000 |
| Val | 100 | 200 | 1200 |
| Test | 250 | 500 | 3000 |

- 每个保留 train source 完整4条轨迹；val/test各2条。
- Source overlap=0，重复trajectory key=0。
- 47000成员均读取并解码，42300份既有SHA256全部匹配；clean/xi/六档state完整。
- 审计读入198.90GB，用时453.51秒，未使用NPU。
- 只创建引用原始tar的manifest，原始manifest/index哈希和tar文件身份前后相同，未重合成或删除数据。未对整个tar重新hash；读取可能改变atime。
- 原val每类仅一个source，因此半量val保留100类，这是精确减半的结果。
- Test只做完整性读取，未运行模型或计算恢复评价。

完整摘要、抽样source/trajectory名单、manifest哈希见 `artifacts/dataset_audit/`。远端数据视图：`/data2/yanglin/v03_half_recursive_20260915/data/places_p2s_bridge_v03_half`。

训练仍从4000条轨迹均匀采样，独立均匀选六档G，忽略旧terminal_G标签。
