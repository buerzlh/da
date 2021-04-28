import torch
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset

from da.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import build_transform


def build_data_loader(cfg, sampler_type='SequentialSampler', data_source=None, batch_size=64, n_domain=0, tfm=None,
    is_train=True, dataset_wrapper=None):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )

    return data_loader


class DataManager:
    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print('* Using custom transform for training')
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print('* Using custom transform for testing')
            tfm_test = custom_tfm_test




        # Build train_loader_s
        train_loader_s = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_S.SAMPLER,
            data_source=dataset.train_s,
            batch_size=cfg.DATALOADER.TRAIN_S.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_S.N_DOMAIN,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        test_loader_s = None
        if dataset.test_s:
            test_loader_s = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.test_s,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_S.N_DOMAIN,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )


        # Build train_loader_t
        train_loader_t = None
        if dataset.train_t:
            sampler_type_ = cfg.DATALOADER.TRAIN_T.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_T.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_T.N_DOMAIN

            if cfg.DATALOADER.TRAIN_T.SAME_AS_S:
                sampler_type_ = cfg.DATALOADER.TRAIN_S.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_S.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_S.N_DOMAIN

            train_loader_t = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_t,
                batch_size=batch_size_,
                n_domain=n_domain_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_s = train_loader_s
        self.train_loader_t = train_loader_t
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_loader_s = test_loader_s

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        print('***** Dataset statistics *****')

        print('  Dataset: {}'.format(cfg.DATASET.NAME))

        if cfg.DATASET.SOURCE_DOMAINS:
            print('  Source domains: {}'.format(cfg.DATASET.SOURCE_DOMAINS))
        if cfg.DATASET.TARGET_DOMAINS:
            print('  Target domains: {}'.format(cfg.DATASET.TARGET_DOMAINS))

        print('  # classes: {}'.format(self.num_classes))

        print('  # train_s: {:,}'.format(len(self.dataset.train_s)))

        if self.dataset.test_s:
            print('  # test_s: {:,}'.format(len(self.dataset.test_s)))


        if self.dataset.train_t:
            print('  # train_t: {:,}'.format(len(self.dataset.train_t)))

        if self.dataset.val:
            print('  # val: {:,}'.format(len(self.dataset.val)))

        print('  # test: {:,}'.format(len(self.dataset.test)))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        # transform accepts list (tuple) as input
        self.transform = transform
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform without any data augmentation
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE)]
        to_tensor += [T.ToTensor()]
        if 'normalize' in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'cname:': item.classname,
            'dname': item.domainname,
            'domain': item.domain,
            'impath': item.impath,
            'id': item.id
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        img0 = self.to_tensor(img0)
        output['img0'] = img0

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
