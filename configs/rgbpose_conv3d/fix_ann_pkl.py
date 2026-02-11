import pickle
import random

src_pkl = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/ntu60_hrnet.pkl'

with open(src_pkl, 'rb') as f:
    data = pickle.load(f)

all_anns = data['annotations']

s001_anns = [
    ann for ann in all_anns
    if ann['frame_dir'].startswith('S001')
]

print(f'S001 samples: {len(s001_anns)}')