from typing import Dict, Sequence, TypeVar, Union
import torch

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def zip_chunk(size, *seqs):
    for i in range(0, max([0] + [len(s) for s in seqs]), size):
        yield [seq[i:i+size] for seq in seqs]

def sized_chunk(seq, sizes: Union[int, Sequence[int]]):
    st = 0
    if isinstance(sizes, int):
        sizes = [sizes] * ((len(seq) + sizes - 1) // sizes)
        sizes[-1] = len(seq) - sum(sizes[:-1])

    for size in sizes:
        yield seq[st:st+size]
        st += size

def dictseq2seqdict(dict_of_seq: Dict[K, Sequence[V]]):
    '''{k0: [v00, v01, ...], ...} -> [{k0: v00, k1: v01, ...}, ...]'''
    keys, values = dict_of_seq.keys(), dict_of_seq.values()
    return [{k: v for k, v in zip(keys, values_t)} for values_t in zip(*values)]

# batchify
def pad_to_same(xs, max_len=None):
    lens = [len(i) for i in xs]    
    if max_len is None:
        max_len = max(lens)
    
    # fall back to concat case
    if all(i == max_len for i in lens):
        return torch.stack(xs), lens

    content_shape = xs[0].shape[1:]
    res_size = [len(xs), max_len, *content_shape]
    if xs[0].dtype is torch.int32 or \
        xs[0].dtype is torch.int64:
        fill_val = -1
    else:
        fill_val = torch.nan
    res = xs[0].new_full(size=res_size, fill_value=fill_val, )

    for i in range(len(xs)):
        res[i, :lens[i]] = xs[i]

    return res, lens