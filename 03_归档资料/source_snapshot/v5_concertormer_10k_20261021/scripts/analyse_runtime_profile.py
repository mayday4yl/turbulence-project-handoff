"""Compact, source-bound summary of the one completed NPU profile."""
import collections
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
directory = root/'reports/profile_R0_b2_highres_profile'
kernel = next(directory.glob('*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
trace = kernel.with_name('trace_view.json')
totals, counts = collections.Counter(), collections.Counter()
with kernel.open() as stream:
    for row in csv.DictReader(stream):
        totals[row['Name']] += float(row['Duration(us)']) / 1e6
        counts[row['Name']] += 1
values = json.loads(trace.read_text())
events = values.get('traceEvents', []) if isinstance(values, dict) else values
matches = []
for event in events:
    dims = event.get('args', {}).get('Input Dims', '')
    if event.get('name') == 'aten::convolution_backward' and ',33,33' in dims:
        matches.append({'name': event['name'], 'timestamp_us': event.get('ts'),
                        'input_dims': dims, 'input_type': event.get('args', {}).get('Input type')})
del values, events
names = ['Conv2DBackpropInput21', 'Conv2DBackpropInput22']
slow = sum(totals[name] for name in names)
report = {'scope': 'one instrumented R0 microbatch2, highest two resolution levels checkpointed',
          'not_uninstrumented_throughput': True,
          'device_kernel_count': sum(counts.values()), 'sum_device_kernel_seconds': sum(totals.values()),
          'two_slow_input_gradient_kernel_seconds': slow,
          'fraction_of_summed_device_kernel_time': slow/sum(totals.values()),
          'top_kernels': [{'name': name, 'seconds': duration, 'count': counts[name]} for name, duration in totals.most_common(20)],
          'simulator_33x33_backward_calls': matches,
          'attribution': 'The frozen Simulator alone uses these 100x1x33x33 and 1x1x33x33 weights; four calls each match its four image preparations.',
          'profiler_warning': 'Stop-in-RECORD warning retained in raw log; parsing completed. Timing is scoped to captured kernels, not a convergence/stability claim.',
          'source_files': []}
for path in (kernel, trace):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8<<20), b''):
            sha.update(chunk)
    report['source_files'].append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()})
output = root/'reports/RUNTIME_PROFILE_SUMMARY.json'
output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ['device_kernel_count', 'sum_device_kernel_seconds',
      'two_slow_input_gradient_kernel_seconds', 'fraction_of_summed_device_kernel_time']}))
