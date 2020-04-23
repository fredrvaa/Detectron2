class Dataset(object):
    def __init__(self):
        self.val_images = '/media/fredrik/HDD/Master/data/dataset/val'
        self.val_annos = '/media/fredrik/HDD/Master/data/dataset/val.json'

        self.test_images = '/media/fredrik/HDD/Master/data/dataset/test'
        self.test_annos = '/media/fredrik/HDD/Master/data/dataset/test.json'

        self.aug_images = '/media/fredrik/HDD/Master/data/dataset/aug'
        self.aug_annos = '/media/fredrik/HDD/Master/data/dataset/aug.json'

class Base(Dataset):
    def __init__(self):
        super().__init__()

        self.annotation_dir = '/media/fredrik/HDD/Master/data/dataset/train/base/annotations'

        self.train_images = '/media/fredrik/HDD/Master/data/dataset/train/base/train'
        self.train_annos = '/media/fredrik/HDD/Master/data/dataset/train/base/annotations/train.json'


class BaseNew(Dataset):
    def __init__(self):
        super().__init__()

        self.annotation_dir = '/media/fredrik/HDD/Master/data/dataset/train/base+new/annotations'

        self.train_images = '/media/fredrik/HDD/Master/data/dataset/train/base+new/train'
        self.train_annos = '/media/fredrik/HDD/Master/data/dataset/train/base+new/annotations/train.json'
