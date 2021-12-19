from .builder import DATASETS
import os
import numpy as np


def find_folders(root):
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx):
    samples = []
    root = os.path.expanduser(root)
    print(f"{folder_to_idx}")
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = os.path.join(root, folder_name)
        for _, _, fns in sorted(os.walk(_dir)):
            # idx = 0
            for fn in sorted(fns):            
                # if idx >10000: break
                path = os.path.join(folder_name, fn)
                item = (path,folder_to_idx[folder_name])
                samples.append(item)
                # idx +=1
    return samples


@DATASETS.register_module()
class EncryptTrafficPic(BaseDataset):
    CLASSES = ['Beanbot','Pletor','Selfmite','zsone']
    def load_annotations(self):
        folder_to_idx = find_folders(self.data_prefix)
        samples = get_samples(self.data_prefix, folder_to_idx)
        self.samples = samples
        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos