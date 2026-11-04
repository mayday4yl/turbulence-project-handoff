from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
FILES = [
 'v5_v1_simulator_20261020/simulator_v5.py',
 'v5_v1_simulator_20261020/vendor/v4_simulator.py',
 'v5_v1_simulator_20261020/method_interface_20261021/schedules.py',
 'v5_concertormer_10k_20261021/models/clean_concertormer.py',
 'v5_concertormer_10k_20261021/models/v5_concertormer.py',
 'v5_concertormer_10k_20261021/training/core.py',
 'v5_concertormer_10k_20261021/alignment_training/core.py',
 'v5_concertormer_10k_20261021/alignment_training/runner.py',
 'unidb_d0_20261102/core.py', 'unidb_d0_20261102/data.py',
 'unidb_dual_20261103_until_noon/improved_objective.py',
 'unidb_dual_20261104_until00/training_worker.py',
 'unidb_dual_20261104_until00/validation.py',
]
WANT = set('build_simulator bind_canonical_buffers effective_state prepare_image render_prepared sampling_grid warp sample_realization render forward sample_training_paths inference_paths build_clean_model unroll_training infer main train save_training_checkpoint initialize configure_npu process make_ema load_pool validation_groups save_checkpoint load_resume evaluate sources_ok source_checks'.split())
lines = ['# 逐函数代码导航', '',
 '行号由当前源码AST提取，SHA见[evidence/CODE_MAP.json](evidence/CODE_MAP.json)。链接指向Git源码快照；服务器根为`/data2/yanglin/<分支>`。', '',
 '当前必读顺序：simulator_v5 → UniDB core → data → 午夜worker/validation。V5 training/core与alignment_training/core仅在追查旧递归实现时阅读。', '',
 '运行用[独立示例](../../tools/receiver/demo.py)；原worker绑定旧root、截止、卡数与源SHA，不能直接当今天的通用launcher。', '',
 '## 读代码时的重点', '',
 '- `effective_state`把raw变成场；`render_prepared`先Smooth再warp；`flow`是归一化grid，位移在`effective_state.displacement`。',
 '- `build_simulator`绑定原canonical buffers；`_P2S.forward`按上游复用fc2，不能当手误改fc3。',
 '- V5 A回投当前g/Yg，B回投G4/Y4；下一状态来自预测，正式refinement OFF。',
 '- UniDB `NativeObjective.forward`是原生一步均值L1；NPU初始化含精确GELU适配。',
 '- 完整续训需要RAW、optimizer、EMA、RNG、loader、cursor和scheduler；加载EMA仅用于推理/新初始化。', '',
 '## 函数索引']
evidence = []
for rel in FILES:
    path = ROOT / 'source_snapshot' / rel
    if not path.exists():
        lines += ['', f'`{rel}` 未包含在当前快照；从远端同名目录读取。']
        continue
    raw = path.read_bytes()
    entries = sorted((n.lineno, n.name) for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.FunctionDef) and n.name in WANT)
    lines += ['', f'### {rel}', '', f'[打开文件](../../source_snapshot/{rel})', '', '|函数|源码入口|', '|---|---|']
    for line, name in entries:
        lines.append(f'|`{name}`|[L{line}](../../source_snapshot/{rel}#L{line})|')
    original = ROOT.parent / rel
    evidence.append({'file': rel, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                     'original_matches': original.exists() and original.read_bytes() == raw,
                     'functions': [{'name': name, 'line': line} for line, name in entries]})
lines += ['', '## 官方UniDB源码', '',
 'Git快照未包含全部官方vendor。de36实际目录是`/data2/yanglin/unidb_d0_20261102/vendor/unidb_gou/`。权威映射在该root的`protocol/UPSTREAM_SOURCE.json`，固定commit为`1bb63e318891b932c6e641b98729643e46b91e06`。', '',
 '|文件|查看内容|', '|---|---|',
 '|`DenoisingUNet_arch.py`|`ConditionalUNet.forward(xt,cond,time)`，xt−cond与cond拼接|',
 '|`sde_utils.py`|随机桥状态、reverse_optimum_step、reverse_sde_step_mean、100步reverse_sde|',
 '|`loss.py`|MatchingLoss L1归约|',
 '|`module_util.py`|时间嵌入、残差块、注意力|', '',
 '[官方固定版本](https://github.com/UniDB-SOC/UniDB/tree/1bb63e318891b932c6e641b98729643e46b91e06/UniDB-GOU)。项目wrapper与官方模块分开登记，不用上游最新main替代冻结版本。']
(ROOT / 'docs/meeting_20261104/CODE_MAP.md').write_text('\n'.join(lines)+'\n')
(ROOT / 'docs/meeting_20261104/evidence/CODE_MAP.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'files':len(evidence), 'all_original_matches':all(x['original_matches'] for x in evidence)}))
