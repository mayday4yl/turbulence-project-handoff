import copy
import unittest
import torch
from train_worker import compare_resume_reexecution


def snapshot():
    return {'network': {'weight': torch.tensor([.125, -.25, .01])},
            'ema': {'network': {'weight': torch.tensor([.12, -.24, .01])}, 'updates': 2},
            'optimizer': {'state': {0: {'exp_avg': torch.tensor([.01, -.02]),
                                        'exp_avg_sq': torch.tensor([1e-5, 2e-5]), 'step': torch.tensor(2.)}},
                          'param_groups': [{'lr': 4e-6}]},
            'scheduler': {'next_lr': 6e-6, 'completed_updates': 2},
            'sampler': {'position': 64, 'rng': torch.tensor([1, 2], dtype=torch.uint8)},
            'rng': {'cpu': torch.tensor([1, 2, 3], dtype=torch.uint8)}, 'counts': {'paths': 64}}


class ResumeReexecutionTests(unittest.TestCase):
    def test_exact_pass_preserved(self):
        a=snapshot();result=compare_resume_reexecution(a,copy.deepcopy(a))
        self.assertTrue(result['passed']);self.assertTrue(result['exact'])
    def test_finite_fp32_roundoff_passes_but_is_never_called_exact(self):
        a=snapshot();b=copy.deepcopy(a)
        for value in [b['network']['weight'],b['ema']['network']['weight'],
                      b['optimizer']['state'][0]['exp_avg'],b['optimizer']['state'][0]['exp_avg_sq']]:
            value[0]=torch.nextafter(value[0],torch.tensor(float('inf')))
        result=compare_resume_reexecution(a,b)
        self.assertTrue(result['passed']);self.assertFalse(result['exact'])
    def test_meaningful_weight_error_fails(self):
        a=snapshot();b=copy.deepcopy(a);b['network']['weight'][0]+=1e-4
        self.assertFalse(compare_resume_reexecution(a,b)['passed'])
    def test_small_second_moment_drift_cannot_hide_behind_larger_first_moment(self):
        a=snapshot();b=copy.deepcopy(a);b['optimizer']['state'][0]['exp_avg_sq'][0]+=1e-8
        self.assertFalse(compare_resume_reexecution(a,b)['passed'])
    def test_lr_schedule_rng_sampler_counter_are_exact(self):
        mutators=[lambda b:b['optimizer']['param_groups'][0].update(lr=4e-6+1e-16),
                  lambda b:b['scheduler'].update(next_lr=6e-6+1e-16),
                  lambda b:b['sampler'].update(position=65),
                  lambda b:b['rng']['cpu'].__setitem__(0,2),
                  lambda b:b['ema'].update(updates=3),
                  lambda b:b['counts'].update(paths=65)]
        for mutate in mutators:
            a=snapshot();b=copy.deepcopy(a);mutate(b)
            self.assertFalse(compare_resume_reexecution(a,b)['passed'])
    def test_optimizer_step_is_exact_even_when_tensor_is_floating(self):
        a=snapshot();b=copy.deepcopy(a);x=b['optimizer']['state'][0]['step'];b['optimizer']['state'][0]['step']=torch.nextafter(x,torch.tensor(float('inf')))
        self.assertFalse(compare_resume_reexecution(a,b)['passed'])
    def test_dtype_shape_keys_and_nonfinite_fail(self):
        mutators=[lambda b:b['network'].update(weight=torch.ones(4)),
                  lambda b:b['network'].update(weight=b['network']['weight'].double()),
                  lambda b:b['network'].update(unexpected=torch.zeros(1)),
                  lambda b:b['network']['weight'].__setitem__(0,float('nan'))]
        for mutate in mutators:
            a=snapshot();b=copy.deepcopy(a);mutate(b)
            self.assertFalse(compare_resume_reexecution(a,b)['passed'])
    def test_zero_reference_cannot_accept_new_nonzero_state(self):
        a=snapshot();a['optimizer']['state'][0]['exp_avg_sq'].zero_();b=copy.deepcopy(a);b['optimizer']['state'][0]['exp_avg_sq'][0]=1e-20
        self.assertFalse(compare_resume_reexecution(a,b)['passed'])

if __name__=='__main__':unittest.main()
