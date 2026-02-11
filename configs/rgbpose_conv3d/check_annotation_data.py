import pickle
import os

ann_path = 'ntu60_xsub_train.pkl'
out_path = 'ntu60_xsub_train_fall.pkl'

rgb_root = '/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/nturgbd_raw/fall_rgb'
valid_ids = set(os.listdir(rgb_root))

with open(ann_path, 'rb') as f:
    data = pickle.load(f)

new_data = []
for item in data:
    if item['frame_dir'] in valid_ids:
        new_data.append(item)

print(f'Before: {len(data)}, After: {len(new_data)}')

with open(out_path, 'wb') as f:
    pickle.dump(new_data, f)
