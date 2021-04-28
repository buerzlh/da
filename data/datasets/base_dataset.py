import os
import os.path as osp
import tarfile
import zipfile
import gdown

from da.utils import check_isfile


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
        domainname (str): domain name
        idx (int): the id in dataset
    """

    def __init__(self, impath='', label=0, domain=-1, classname='', domainname='', id=-1):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)
        assert isinstance(domainname, str)
        assert isinstance(id, int)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname
        self._domainname = domainname
        self._id = id

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

    @property
    def domainname(self):
        return self._domainname

    @property
    def id(self):
        return self._id


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # directory which contains the dataset
    domains = [] # string names of all domains

    def __init__(self, train_s=None, test_s=None, train_t=None, val=None, test=None):
        self._train_s = train_s                     # labeled training data
        self._test_s = test_s                       # labeled test source data(use for free source, optional)
        self._train_t = train_t                     # unlabeled training data (optional)
        self._val = val                             # validation data (optional)
        self._test = test                           # test data

        self._num_classes = self.get_num_classes(train_s)
        self._lab2cname = self.get_label_classname_mapping(train_s)

    @property
    def train_s(self):
        return self._train_s

    @property
    def test_s(self):
        return self._test_s

    @property
    def train_t(self):
        return self._train_t

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_label_classname_mapping(self, data_source):
        tmp = set()
        for item in data_source:
            tmp.add((item.label, item.classname))
        mapping = {label: classname for label, classname in tmp}
        return mapping


    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))
