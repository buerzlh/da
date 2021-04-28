import os.path as osp

from da.utils import listdir_nohidden

from .build import DATASET_REGISTRY
from .base_dataset import Datum, DatasetBase

from random import shuffle


@DATASET_REGISTRY.register()
class OfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """
    dataset_dir = 'office_home'
    domains = ['art', 'clipart', 'product', 'real_world']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_s = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        test_s = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        if cfg.DATALOADER.TRAIN_S.VAL_SOURCE:
            p = 1 - cfg.DATASET.VAL_PERCENT
            shuffle(train_s)
            dsize = len(train_s)
            tsize = int(p * dsize)
            train_s, test_s = train_s[0:tsize], train_s[tsize:dsize]
        train_t = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_s=train_s, test_s=test_s, train_t=train_t, test=test)

    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()
            id = 0
            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(impath=impath, label=label, domain=domain, classname=class_name, domainname=dname, id=id)
                    id = id + 1
                    items.append(item)

        return items
