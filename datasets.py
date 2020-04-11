class Base(object):
    def __init__(self):
        self.annotation_dir = '/media/fredrik/HDD/Master/data/base/annotations'

        self.train_images = '/media/fredrik/HDD/Master/data/base/train'
        self.train_annos = '/media/fredrik/HDD/Master/data/base/annotations/train.json'

        self.val_images = '/media/fredrik/HDD/Master/data/base/val'
        self.val_annos = '/media/fredrik/HDD/Master/data/base/annotations/val.json'

        self.test_images = '/media/fredrik/HDD/Master/data/base/test'
        self.test_annos = '/media/fredrik/HDD/Master/data/base/annotations/test.json'

class BaseNew(object):
    def __init__(self):
        self.annotation_dir = '/media/fredrik/HDD/Master/data/base+new/annotations'

        self.train_images = '/media/fredrik/HDD/Master/data/base+new/train'
        self.train_annos = '/media/fredrik/HDD/Master/data/base+new/annotations/train.json'

        self.val_images = '/media/fredrik/HDD/Master/data/base+new/val'
        self.val_annos = '/media/fredrik/HDD/Master/data/base+new/annotations/val.json'

        self.test_images = '/media/fredrik/HDD/Master/data/base+new/test'
        self.test_annos = '/media/fredrik/HDD/Master/data/base+new/annotations/test.json'
