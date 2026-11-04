"""Same deterministic source/class traversal generalized from 4 to 16 entries."""
from collections import defaultdict
import random
from method_impl.training_io import PositionDataset
from method_impl.d0 import derive_seed
class MixedOrder:
    def __init__(self,pool,seed):
        self.pool=pool;self.seed=seed;self._cycle=None;self._order=None
        self.groups=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(pool):self.groups[r['category_id']][r['source_id']].append(i)
        self.groups={c:dict(v) for c,v in self.groups.items()}
        assert len(self.groups)==50 and len({len(v) for v in self.groups.values()})==1
        for group in self.groups.values():
            for ids in group.values():
                assert len(ids)==16 and {(pool[i]['realization_index'],pool[i]['g']) for i in ids}=={(r,g) for r in range(4) for g in [1,2,3,4]}
    def cycle(self,cycle):
        if self._cycle==cycle:return self._order
        rng=random.Random(derive_seed(self.seed,'balanced_cycle',cycle));cats=sorted(self.groups);byclass={}
        for c in cats:
            names=sorted(self.groups[c]);rng.shuffle(names);byclass[c]=[]
            for name in names:
                ids=list(self.groups[c][name]);rng.shuffle(ids);byclass[c].append(ids)
        result=[];count=len(next(iter(byclass.values())))
        for realization_pass in range(16):
            for source_position in range(count):
                order=list(cats);rng.shuffle(order)
                result.extend(byclass[c][source_position][realization_pass] for c in order)
        assert len(result)==len(self.pool)==len(set(result))
        self._cycle=cycle;self._order=result;return result
    def at(self,position):
        cycle,offset=divmod(position,len(self.pool));return self.cycle(cycle)[offset]
class MixedPositionDataset(PositionDataset):
    def __init__(self,pool,seed,cursor,count,rank,world,microbatch):
        self.pool=pool;self.order=MixedOrder(pool,seed);self.cursor=cursor;self.count=count
        self.rank=rank;self.world=world;self.microbatch=microbatch;self.verified=set()
