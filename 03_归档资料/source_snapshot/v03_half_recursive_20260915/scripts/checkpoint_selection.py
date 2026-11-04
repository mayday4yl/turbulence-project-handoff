"""Pure-CPU application of the frozen validation selection and readiness rules."""
import math


def choose_checkpoint(summaries, policy, require_full=True):
    if not summaries:
        raise ValueError('No validation evidence')
    identities = {(s['identity']['selected_keys_sha256'], s['identity']['weight_key']) for s in summaries}
    if len(identities) != 1:
        raise ValueError('Never rank different validation subsets or weight types together')
    for summary in summaries:
        if (summary['status'] != 'complete' or summary['split'] != 'val'
                or summary['held_out_test_opened'] or not summary['inference_unchanged']):
            raise ValueError('Complete unchanged-inference validation required')
        if require_full and summary['identity']['evaluation_scope'] != 'full-val':
            raise ValueError('Final recommendation requires full validation')
        if not all(math.isfinite(float(v)) for k in ['0','1','3','5']
                   for v in summary['per_k'][k]['macro_g'].values()):
            raise ValueError('Nonfinite validation quality')
    rules = policy['checkpoint_selection']
    best_k0 = max(s['per_k']['0']['macro_g']['psnr'] for s in summaries)
    preserved = [s for s in summaries if s['per_k']['0']['macro_g']['psnr'] >= best_k0-rules['k0_preservation_db']]
    best_psnr = max(s['macro_g_and_k']['psnr'] for s in preserved)
    tied_psnr = [s for s in preserved if s['macro_g_and_k']['psnr'] >= best_psnr-rules['psnr_tie_db']]
    best_lpips = min(s['macro_g_and_k']['lpips'] for s in tied_psnr)
    tied_lpips = [s for s in tied_psnr if s['macro_g_and_k']['lpips'] <= best_lpips+rules['lpips_tie']]
    chosen = min(tied_lpips, key=lambda s:s['checkpoint_update'])
    return chosen, {'best_comparable_K0_PSNR': best_k0,
                    'K0_preserved': chosen['per_k']['0']['macro_g']['psnr'] >= best_k0-rules['k0_preservation_db'],
                    'comparable_checkpoints': len(summaries), 'final_recommendation': require_full}


def progressive_gate(summary, policy, k0_preserved):
    rule = policy['progressive_gain']
    per_k = summary['per_k']
    delta = {metric: sum(per_k[str(k)]['macro_g'][metric] for k in [1,3,5])/3-per_k['0']['macro_g'][metric]
             for metric in ['psnr','ssim','lpips','l1']}
    per_g = {g: sum(per_k[str(k)]['per_g'][g]['quality']['psnr'] for k in [1,3,5])/3
                - per_k['0']['per_g'][g]['quality']['psnr'] for g in per_k['0']['per_g']}
    checks = {'K0_preserved': k0_preserved,
              'mean_progressive_PSNR': delta['psnr'] >= rule['minimum_psnr_delta_db'],
              'positive_G_count': sum(value > 0 for value in per_g.values()) >= rule['minimum_positive_G_count'],
              'no_large_G_regression': min(per_g.values()) >= rule['minimum_per_G_psnr_delta_db'],
              'LPIPS_preserved': delta['lpips'] <= rule['maximum_lpips_delta'],
              'SSIM_preserved': delta['ssim'] >= rule['minimum_ssim_delta']}
    return {'passed': all(checks.values()), 'checks': checks,
            'mean_progressive_minus_K0': delta, 'per_G_PSNR_delta': per_g,
            'interpretation_scope': 'Full V0.3 protocol; not isolated causal proof of recursive training.'}
