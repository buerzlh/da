import time
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from da.data import DataManager
from da.optim import build_optimizer, build_lr_scheduler
from da.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, resume_from_checkpoint, load_pretrained_weights
)
from da.modeling import build_head, build_backbone
from da.evaluation import build_evaluator


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs
        )
        fdim = self.backbone.out_features

        # self.head = None
        # if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
        #     self.head = build_head(
        #         model_cfg.HEAD.NAME,
        #         verbose=cfg.VERBOSE,
        #         in_features=fdim,
        #         hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
        #         activation=model_cfg.HEAD.ACTIVATION,
        #         bn=model_cfg.HEAD.BN,
        #         dropout=model_cfg.HEAD.DROPOUT,
        #         **kwargs
        #     )
        #     fdim = self.head.out_features
        #
        # self.classifier = None
        # if num_classes > 0:
        #     self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        return f
        # if self.head is not None:
        #     f = self.head(f)
        #
        # if self.classifier is None:
        #     return f
        #
        # y = self.classifier(f)
        #
        # if return_feature:
        #     return y, f
        #
        # return y


class TrainerBase:
    """Base class for unsupervised DA and multi-source DA trainer."""

    def __init__(self, cfg):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.dm.lab2cname)

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        self.dm = DataManager(self.cfg)
        self.train_loader_s = self.dm.train_loader_s
        self.train_loader_t = self.dm.train_loader_t
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.num_source_domains = self.dm.num_source_domains

    def build_model(self):
        raise NotImplementedError

    def register_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False):
        names = self.get_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models[name].state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': self._optims[name].state_dict(),
                    'scheduler': self._scheds[name].state_dict()
                },
                osp.join(directory, name),
                is_best=is_best
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print('No checkpoint found, train from scratch')
            return 0

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        names = self.get_model_names()
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode='train', names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Loss is infinite or NaN!')

    def init_writer(self, log_dir):
        if self.__dict__.get('_writer') is None or self._writer is None:
            print(
                'Initializing summary writer for tensorboard '
                'with log_dir={}'.format(log_dir)
            )
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self):
        """Generic training loops."""
        self.max_acc = 0
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        self.init_writer(self.output_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print('Finished training')

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        if self.temp_acc > self.max_acc:
            self.max_acc = self.temp_acc
            self.save_model(self.epoch, self.output_dir, is_best=True)

        print("max_acc: " + str(self.max_acc) + "%")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        # Close writer
        self.close_writer()

    def before_epoch(self):
        pass

    def after_epoch(self):
        not_last_epoch = (self.epoch + 1) != self.max_epoch
        do_test = self.cfg.TEST.EVAL_FREQ > 0 and not self.cfg.TEST.NO_TEST
        meet_test_freq = (
                                 self.epoch + 1
                         ) % self.cfg.TEST.EVAL_FREQ == 0 if do_test else False
        meet_checkpoint_freq = (
                                       self.epoch + 1
                               ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        if not_last_epoch and do_test and meet_test_freq:
            self.test()

        if not_last_epoch and meet_checkpoint_freq and self.temp_acc > self.max_acc:
            self.max_acc = self.temp_acc
            self.save_model(self.epoch, self.output_dir, is_best=True)

        print("max_acc: " + str(self.max_acc) + "%")

    def run_epoch(self):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_s = len(self.train_loader_s)
        len_train_loader_t = len(self.train_loader_t)
        if self.cfg.TRAIN.COUNT_ITER == 'train_s':
            self.num_batches = len_train_loader_s
        elif self.cfg.TRAIN.COUNT_ITER == 'train_t':
            self.num_batches = len_train_loader_t
        elif self.cfg.TRAIN.COUNT_ITER == 'smaller_one':
            self.num_batches = min(len_train_loader_s, len_train_loader_t)
        elif self.cfg.TRAIN.COUNT_ITER == 'bigger_one':
            self.num_batches = max(len_train_loader_s, len_train_loader_t)
        else:
            raise ValueError

        train_loader_s_iter = iter(self.train_loader_s)
        train_loader_t_iter = iter(self.train_loader_t)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_s = next(train_loader_s_iter)
            except StopIteration:
                train_loader_s_iter = iter(self.train_loader_s)
                batch_s = next(train_loader_s_iter)

            try:
                batch_t = next(train_loader_t_iter)
            except StopIteration:
                train_loader_t_iter = iter(self.train_loader_t)
                batch_t = next(train_loader_t_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_s, batch_t)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        self.temp_acc = results['accuracy']


        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)


    def parse_batch_train(self, batch_s, batch_t):
        input_s = batch_s['img']
        label_s = batch_s['label']
        input_t = batch_t['img']

        input_s = input_s.to(self.device)
        label_s = label_s.to(self.device)
        input_t = input_t.to(self.device)

        return input_s, label_s, input_t

    def parse_batch_test(self, batch):
        input = batch['img']
        label = batch['label']

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def forward_backward(self, batch_s, batch_t):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError


    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']


class TrainerFreeSource:
    """Base class for source free trainer."""
    def __init__(self, cfg):
        self._models_s = OrderedDict()
        self._optims_s = OrderedDict()
        self._scheds_s = OrderedDict()

        self._models_t = OrderedDict()
        self._optims_t = OrderedDict()
        self._scheds_t = OrderedDict()

        self._writer = None

        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Save as attributes some frequently used variables
        self.start_epoch_s = self.start_epoch_t = self.epoch_s = self.epoch_t = 0
        self.source_max_epoch = cfg.OPTIM.SOURCE_MAX_EPOCH
        self.target_max_epoch = cfg.OPTIM.TARGET_MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        if self.cfg.TARGET_TRAIN:
            self.build_target_model()
        else:
            self.build_source_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.dm.lab2cname)

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        What must be done in the re-implementation
        of this method:
        1) initialize data manager
        2) assign as attributes the data loaders
        3) assign as attribute the number of classes
        """
        self.dm = DataManager(self.cfg)
        self.train_loader_s = self.dm.train_loader_s                     ### use for train source model
        self.test_loader_s = self.dm.test_loader_s                       ### use for test source model
        self.train_loader_t = self.dm.train_loader_t
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.num_source_domains = self.dm.num_source_domains

    def build_source_model(self):
        raise NotImplementedError

    def build_target_model(self):
        raise NotImplementedError

    def register_source_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models_s') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims_s') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds_s') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models_s[name] = model
        self._optims_s[name] = optim
        self._scheds_s[name] = sched

    def register_target_model(self, name='model', model=None, optim=None, sched=None):
        if self.__dict__.get('_models_t') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('_optims_t') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('_scheds_t') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self._models_t[name] = model
        self._optims_t[name] = optim
        self._scheds_t[name] = sched

    def get_source_model_names(self, names=None):
        names_real = list(self._models_s.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def get_target_model_names(self, names=None):
        names_real = list(self._models_t.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_source_model(self, epoch, directory, is_best=False):
        names = self.get_source_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models_s[name].state_dict(),
                    'epoch': epoch + 1,
                    'optimizer': self._optims_s[name].state_dict(),
                    'scheduler': self._scheds_s[name].state_dict()
                },
                osp.join(directory, name),
                is_best=is_best
            )

    def save_target_model(self, epoch, directory, is_best=False):
        names = self.get_target_model_names()

        for name in names:
            save_checkpoint(
                {
                    'state_dict': self._models_t[name].state_dict(),
                    'epoch': epoch + 1,
                    # 'optimizer': self._optims_t[name].state_dict(),
                    # 'scheduler': self._scheds_t[name].state_dict()
                },
                osp.join(directory, name),
                is_best=is_best
            )


    def resume_source_model_if_exist(self, directory):
        names = self.get_source_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print('No checkpoint found, train from scratch')
            return 0

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models_s[name], self._optims_s[name],
                self._scheds_s[name]
            )

        return start_epoch

    def resume_target_model_if_exist(self, directory):
        names = self.get_target_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print('No checkpoint found, train from scratch')
            return 0

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models_t[name], self._optims_t[name],
                self._scheds_t[name]
            )

        return start_epoch

    def load_source_model(self, directory, epoch=None):
        names = self.get_source_model_names()
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models_s[name].load_state_dict(state_dict)

    def load_target_model(self, directory, epoch=None):
        names = self.get_target_model_names()
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models_t[name].load_state_dict(state_dict)

    def get_source_model(self, model, name, directory):
        # name = self.get_source_model_names(name)
        model_file = 'model-best.pth.tar'
        model_path = osp.join(directory, name, model_file)

        if not osp.exists(model_path):
            raise FileNotFoundError(
                'Model not found at "{}"'.format(model_path)
            )

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict)






    def set_source_model_mode(self, mode='train', names=None):
        names = self.get_source_model_names(names)

        for name in names:
            if mode == 'train':
                self._models_s[name].train()
            else:
                self._models_s[name].eval()

    def set_target_model_mode(self, mode='train', names=None):
        names = self.get_target_model_names(names)

        for name in names:
            if mode == 'train':
                self._models_t[name].train()
            else:
                self._models_t[name].eval()

    def update_source_lr(self, names=None):
        names = self.get_source_model_names(names)

        for name in names:
            if self._scheds_s[name] is not None:
                self._scheds_s[name].step()

    def update_target_lr(self, names=None):
        names = self.get_target_model_names(names)

        for name in names:
            if self._scheds_t[name] is not None:
                self._scheds_t[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Loss is infinite or NaN!')

    def init_writer(self, log_dir):
        if self.__dict__.get('_writer') is None or self._writer is None:
            print(
                'Initializing summary writer for tensorboard '
                'with log_dir={}'.format(log_dir)
            )
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train_s(self):
        """Generic training loops."""
        self.max_acc_s = 0
        self.before_train_s()
        for self.epoch_s in range(self.start_epoch_s, self.source_max_epoch):
            self.before_epoch_s()
            self.run_epoch_s()
            self.after_epoch_s()
        self.after_train_s()

    def train_t(self):
        self.max_acc_t = 0
        self.before_train_t()
        for self.epoch_t in range(self.start_epoch_t, self.target_max_epoch):
            self.before_epoch_t()
            self.run_epoch_t()
            self.after_epoch_t()
        self.after_train_t()

    def before_train_s(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch_s = self.resume_source_model_if_exist(directory)

        # Initialize summary writer
        self.init_writer(self.output_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def before_train_t(self):
        directory = self.cfg.OUTPUT_DIR
        self.start_epoch_t = self.resume_target_model_if_exist(directory)

        # Initialize summary writer
        self.init_writer(self.output_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train_s(self):
        print('Finished source training')

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test_s()

        # Save model
        if self.temp_acc_s > self.max_acc_s:
            self.max_acc_s = self.temp_acc_s
            self.save_source_model(self.epoch_s, self.output_dir, is_best=True)

        print("max_acc_source: " + str(self.max_acc_s) + "%")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        # Close writer
        self.close_writer()

    def after_train_t(self):
        print('Finished target training')

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test_t()

        # Save model
        if self.temp_acc_t > self.max_acc_t:
            self.max_acc_t = self.temp_acc_t
            self.save_target_model(self.epoch_t, self.output_dir, is_best=True)

        print("max_acc_target: " + str(self.max_acc_t) + "%")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

        # Close writer
        self.close_writer()

    def before_epoch_s(self):
        pass

    def before_epoch_t(self):
        pass

    def after_epoch_s(self):
        not_last_epoch = (self.epoch_s + 1) != self.source_max_epoch
        do_test = self.cfg.TEST.EVAL_FREQ > 0 and not self.cfg.TEST.NO_TEST
        meet_test_freq = (
                                 self.epoch_s + 1
                         ) % self.cfg.TEST.EVAL_FREQ == 0 if do_test else False
        meet_checkpoint_freq = (
                                       self.epoch_s + 1
                               ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        if not_last_epoch and do_test and meet_test_freq:
            self.test_s()

        if not_last_epoch and meet_checkpoint_freq and self.temp_acc_s > self.max_acc_s:
            self.max_acc_s = self.temp_acc_s
            self.save_source_model(self.epoch_s, self.output_dir, is_best=True)

        print("max_acc: " + str(self.max_acc_s) + "%")

    def after_epoch_t(self):
        not_last_epoch = (self.epoch_t + 1) != self.target_max_epoch
        do_test = self.cfg.TEST.EVAL_FREQ > 0 and not self.cfg.TEST.NO_TEST
        meet_test_freq = (
                                 self.epoch_t + 1
                         ) % self.cfg.TEST.EVAL_FREQ == 0 if do_test else False
        meet_checkpoint_freq = (
                                       self.epoch_t + 1
                               ) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False

        if not_last_epoch and do_test and meet_test_freq:
            self.test_t()

        if not_last_epoch and meet_checkpoint_freq and self.temp_acc_t > self.max_acc_t:
            self.max_acc_t = self.temp_acc_t
            self.save_target_model(self.epoch_t, self.output_dir, is_best=True)

        print("max_acc: " + str(self.max_acc_t) + "%")

    def run_epoch_s(self):
        self.set_source_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_s = len(self.train_loader_s)
        self.num_batches_s = len_train_loader_s

        train_loader_s_iter = iter(self.train_loader_s)

        end = time.time()
        for self.batch_idx_s in range(self.num_batches_s):
            try:
                batch_s = next(train_loader_s_iter)
            except StopIteration:
                train_loader_s_iter = iter(self.train_loader_s)
                batch_s = next(train_loader_s_iter)


            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_s(batch_s)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx_s + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches_s - (self.batch_idx_s + 1)
                nb_future_epochs = (
                    self.source_max_epoch - (self.epoch_s + 1)
                ) * self.num_batches_s
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch_s + 1,
                        self.source_max_epoch,
                        self.batch_idx_s + 1,
                        self.num_batches_s,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr_s()
                    )
                )

            n_iter = self.epoch_s * self.num_batches_s + self.batch_idx_s
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr_s(), n_iter)

            end = time.time()


    def run_epoch_t(self):
        self.set_target_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_t = len(self.train_loader_t)
        self.num_batches_t = len_train_loader_t


        train_loader_t_iter = iter(self.train_loader_t)

        end = time.time()
        for self.batch_idx_t in range(self.num_batches_t):
            try:
                batch_t = next(train_loader_t_iter)
            except StopIteration:
                train_loader_t_iter = iter(self.train_loader_t)
                batch_t = next(train_loader_t_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward_t(batch_t)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx_t + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches_t - (self.batch_idx_t + 1)
                nb_future_epochs = (
                    self.target_max_epoch - (self.epoch_t + 1)
                ) * self.num_batches_t
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch_t + 1,
                        self.target_max_epoch,
                        self.batch_idx_t + 1,
                        self.num_batches_t,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr_t()
                    )
                )

            n_iter = self.epoch_t * self.num_batches_t + self.batch_idx_t
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
            self.write_scalar('train/lr', self.get_current_lr_t(), n_iter)

            end = time.time()


    @torch.no_grad()
    def test_s(self):
        """A generic testing pipeline."""
        self.set_source_model_mode('eval')
        self.evaluator.reset()


        print('Do evaluation on Source Val set')
        data_loader = self.test_loader_s
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference_s(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        self.temp_acc_s = results['accuracy']


        for k, v in results.items():
            tag = '{}/{}'.format('Source Val', k)
            self.write_scalar(tag, v, self.epoch_s)

    @torch.no_grad()
    def test_t(self):
        """A generic testing pipeline."""
        self.set_target_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference_t(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        self.temp_acc_t = results['accuracy']

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch_t)


    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def parse_batch_test(self, batch):
        input = batch['img']
        label = batch['label']

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def forward_backward_s(self, batch_s):
        raise NotImplementedError

    def forward_backward_t(self, batch_t):
        raise NotImplementedError

    def model_inference_s(self, input):
        raise NotImplementedError

    def model_inference_t(self, input):
        raise NotImplementedError

    def model_zero_grad_s(self, names=None):
        names = self.get_source_model_names(names)
        for name in names:
            self._optims_s[name].zero_grad()

    def model_zero_grad_t(self, names=None):
        names = self.get_target_model_names(names)
        for name in names:
            self._optims_t[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update_s(self, names=None):
        names = self.get_source_model_names(names)
        for name in names:
            self._optims_s[name].step()

    def model_update_t(self, names=None):
        names = self.get_target_model_names(names)
        for name in names:
            self._optims_t[name].step()

    def model_backward_and_update_s(self, loss, names=None):
        self.model_zero_grad_s(names)
        self.model_backward(loss)
        self.model_update_s(names)

    def model_backward_and_update_t(self, loss, names=None):
        self.model_zero_grad_t(names)
        self.model_backward(loss)
        self.model_update_t(names)

    def get_current_lr_s(self, names=None):
        names = self.get_source_model_names(names)
        name = names[0]
        return self._optims_s[name].param_groups[0]['lr']

    def get_current_lr_t(self, names=None):
        names = self.get_target_model_names(names)
        name = names[0]
        return self._optims_t[name].param_groups[0]['lr']


