import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

from .build import TRAINER_REGISTRY
from .trainer import TrainerBase, SimpleNet

from da.optim import build_optimizer, build_lr_scheduler
from da.utils import count_num_param


###this part is referd from CDAN
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class Discriminator(nn.Module):
    def __init__(self, in_feature, hidden_size, out_size):
        super(Discriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, out_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.apply(init_weights)

    def forward(self, x, alpha):
        x = x * 1.0
        x.register_hook(grl_hook(alpha))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y


@TRAINER_REGISTRY.register()
class MSDTR_CDAN(TrainerBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        batch_size = cfg.DATALOADER.TRAIN_S.BATCH_SIZE
        self.r_i = cfg.TRAINER.MSDTR.RECONSTRUCTION
        self.split_batch = batch_size // self.num_source_domains
        self.n_domain = self.num_source_domains
        self.cls_d = cfg.TRAINER.MSDTR.cls_d
        self.adv_begin = cfg.TRAINER.MSDTR.adv_begin
        self.mask_b = cfg.TRAINER.MSDTR.mask_b
        self.mask_k = cfg.TRAINER.MSDTR.mask_k
        self.g = cfg.TRAINER.MSDTR.g


    def build_model(self):
        cfg = self.cfg
        print('Building G')
        self.G = SimpleNet(cfg, cfg.MODEL)
        self.G.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.G)))
        print(self.G.fdim)

        self.optim_G = build_optimizer(self.G, cfg.OPTIM, cfg.TRAINER.MSDTR.lr_muti_g)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model('G', self.G, self.optim_G, self.sched_G)


        print('Building D_di')
        self.D_di = nn.Sequential(
            nn.Linear(self.G.fdim, cfg.TRAINER.MSDTR.HIDDEN[0]),
            nn.BatchNorm1d(cfg.TRAINER.MSDTR.HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(cfg.TRAINER.MSDTR.HIDDEN[0], cfg.TRAINER.MSDTR.HIDDEN[1]),
            nn.BatchNorm1d(cfg.TRAINER.MSDTR.HIDDEN[1]),
            nn.ReLU(),
        )
        self.D_di.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.D_di)))

        self.optim_D_di = build_optimizer(self.D_di, cfg.OPTIM, cfg.TRAINER.MSDTR.lr_muti_ddi)
        self.sched_D_di = build_lr_scheduler(self.optim_D_di, cfg.OPTIM)
        self.register_model('D_di', self.D_di, self.optim_D_di, self.sched_D_di)

        print('Building C_di')
        self.C_di = nn.Linear(cfg.TRAINER.MSDTR.HIDDEN[1], self.num_classes, bias=False)
        self.C_di.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C_di)))

        self.optim_C_di = build_optimizer(self.C_di, cfg.OPTIM, cfg.TRAINER.MSDTR.lr_muti_cdi)
        self.sched_C_di = build_lr_scheduler(self.optim_C_di, cfg.OPTIM)
        self.register_model('C_di', self.C_di, self.optim_C_di, self.sched_C_di)

        print('Building D_ds')
        self.D_ds = nn.Sequential(
            nn.Linear(self.G.fdim, cfg.TRAINER.MSDTR.HIDDEN[0]),
            nn.BatchNorm1d(cfg.TRAINER.MSDTR.HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(cfg.TRAINER.MSDTR.HIDDEN[0], cfg.TRAINER.MSDTR.HIDDEN[1]),
            nn.BatchNorm1d(cfg.TRAINER.MSDTR.HIDDEN[1]),
            nn.ReLU(),
        )
        self.D_ds.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.D_ds)))

        self.optim_D_ds = build_optimizer(self.D_ds, cfg.OPTIM, cfg.TRAINER.MSDTR.lr_muti_dds)
        self.sched_D_ds = build_lr_scheduler(self.optim_D_ds, cfg.OPTIM)
        self.register_model('D_ds', self.D_ds, self.optim_D_ds, self.sched_D_ds)

        print('Building C_ds')
        self.C_ds = nn.Linear(cfg.TRAINER.MSDTR.HIDDEN[1], self.num_source_domains+1, bias=False)
        self.C_ds.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C_ds)))

        self.optim_C_ds = build_optimizer(self.C_ds, cfg.OPTIM, cfg.TRAINER.MSDTR.lr_muti_cds)
        self.sched_C_ds = build_lr_scheduler(self.optim_C_ds, cfg.OPTIM)
        self.register_model('C_ds', self.C_ds, self.optim_C_ds, self.sched_C_ds)

        print('Building R')
        self.R = nn.Sequential(
            nn.Linear(2*cfg.TRAINER.MSDTR.HIDDEN[1], self.G.fdim),
            nn.BatchNorm1d(self.G.fdim),
            nn.ReLU(),
            nn.Linear(self.G.fdim, self.G.fdim)
        )
        self.R.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.R)))

        self.optim_R = build_optimizer(self.R, cfg.OPTIM, cfg.TRAINER.MSDTR.lr_muti_r)
        self.sched_R = build_lr_scheduler(self.optim_R, cfg.OPTIM)
        self.register_model('R', self.R, self.optim_R, self.sched_R)

        print('Building Dis')
        self.Dis = Discriminator(cfg.TRAINER.MSDTR.HIDDEN[1]*self.num_classes,
                                 cfg.TRAINER.MSDTR.DIS_SIZE, self.num_source_domains+1)
        self.Dis.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.Dis)))
        self.optim_Dis = build_optimizer(self.Dis, cfg.OPTIM, cfg.TRAINER.MSDTR.lr_muti_dis)
        self.sched_Dis = build_lr_scheduler(self.optim_Dis, cfg.OPTIM)
        self.register_model('Dis', self.Dis, self.optim_Dis, self.sched_Dis)

        print('Building Cs and Ct')
        self.Cs = nn.ModuleList(
            [nn.Linear(self.G.fdim, self.num_classes, bias=False) for _ in range(self.num_source_domains)]
        )
        self.Ct = nn.Linear(self.G.fdim, self.num_classes, bias=False)
        self.Cs.to(self.device)
        self.Ct.to(self.device)

    def Entropy(self, input_):
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def reconstruct_loss(self, src, tgt):
        # return torch.mean(torch.abs(src-tgt))
        return torch.sum((src-tgt)**2) / (src.shape[0]*src.shape[1])

    def forward_backward(self, batch_s, batch_t):
        parsed = self.parse_batch_train(batch_s, batch_t)
        input_s, label_s, domain_s, input_t, domain_t = parsed
        domain_label = torch.cat((domain_s, domain_t), 0)

        fea_G_s = self.G(input_s)
        fea_G_t = self.G(input_t)

        fea_di_s = self.D_di(fea_G_s)
        fea_di_t = self.D_di(fea_G_t)

        p_di_s = self.C_di(fea_di_s)
        p_di_t = self.C_di(fea_di_t)

        loss_cls_s = nn.CrossEntropyLoss()(p_di_s, label_s)

        fea_di = torch.cat((fea_di_s, fea_di_t), 0)
        p_di = nn.Softmax(dim=1)(torch.cat((p_di_s, p_di_t), 0))

        loss_adv = torch.Tensor([0]).cuda()
        if self.epoch > self.adv_begin:
            p = float(self.num_batches * (self.epoch - self.start_epoch) + self.batch_idx) / \
                self.max_epoch / self.num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            entropy = self.Entropy(p_di)
            # entropy = None
            loss_adv =self.cdan_loss(fea_di, p_di, domain_label, alpha,entropy)

        fea_ds_s = self.D_ds(fea_G_s)
        fea_ds_t = self.D_ds(fea_G_t)

        fea_ds = torch.cat((fea_ds_s, fea_ds_t), 0)
        p_ds = self.C_ds(fea_ds)

        loss_cls_d = nn.CrossEntropyLoss()(p_ds, domain_label)

        loss_1 = loss_cls_s + loss_adv + self.cls_d*loss_cls_d
        self.model_zero_grad(names=['G', 'D_di', 'C_di', 'D_ds', 'C_ds', 'Dis'])
        loss_1.backward(retain_graph=True)
        self.model_update(names=['G', 'D_di', 'C_di', 'D_ds', 'C_ds', 'Dis'])

        fea = torch.cat((fea_di, fea_ds), 1)
        fea_G = torch.cat((fea_G_s, fea_G_t), 0)
        fea_R = self.R(fea)

        loss_R = self.reconstruct_loss(fea_R, fea_G)

        self.model_zero_grad(names=['R'])
        loss_R.backward(retain_graph=True)
        self.model_update(names=['R'])

        if self.batch_idx % self.r_i == 0:
            domain_invariant = self.C_di.weight
            domain_specific = self.C_ds.weight.t()

            source_domain_specific = []
            for i in range(self.num_source_domains):
                source_domain_specific.append(domain_specific[:, i:i+1].t().expand(self.num_classes, -1))

            target_domain_specific = domain_specific[:, self.num_source_domains:self.num_source_domains+1]\
                .t().expand(self.num_classes, -1)

            for i in range(self.num_source_domains):
                source_fea_i = torch.cat((domain_invariant, source_domain_specific[i]), 1)
                source_r = self.R(source_fea_i)
                self.Cs[i].weight = nn.Parameter(source_r, requires_grad=False)

            target_fea = torch.cat((domain_invariant, target_domain_specific), 1)
            target_r = self.R(target_fea)
            self.Ct.weight = nn.Parameter(target_r, requires_grad=False)

        loss_s = 0
        for i in range(self.num_source_domains):
            mask = (domain_s==i)
            fea_G_s_i = fea_G_s[mask]
            label_s_i = label_s[mask]
            p_Cs_i = self.Cs[i](fea_G_s_i)
            loss_s = loss_s + nn.CrossEntropyLoss()(p_Cs_i, label_s_i)

        p_Ct = self.Ct(fea_G_t)
        pre_t = nn.Softmax(dim=1)(p_di_t)
        p_T, prelabel = torch.max(pre_t, 1)
        p = float(self.batch_idx + 1 + self.epoch * self.num_batches)/(self.max_epoch * self.num_batches)
        mask = (p_T > (self.mask_b + p/self.mask_k))
        prelabel = prelabel[mask]
        p_Ct = p_Ct[mask]

        loss_t = torch.Tensor([0]).cuda()
        if p_Ct.size(0) > 0:
            loss_t = nn.CrossEntropyLoss()(p_Ct, prelabel)

        loss_G = self.g*(loss_s + loss_t)
        self.model_backward_and_update(loss_G, 'G')

        loss_summary = {
            'loss_cls_s': loss_cls_s.item(),
            'loss_cls_d': loss_cls_d.item(),
            'loss_adv': loss_adv.item(),
            'loss_R': loss_R.item(),
            'loss_G':loss_G.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_s, batch_t):
        input_s = batch_s['img']
        label_s = batch_s['label']
        domain_s = batch_s['domain']
        input_t = batch_t['img']
        domain_t = batch_t['domain'] + self.num_source_domains

        input_s = input_s.to(self.device)
        label_s = label_s.to(self.device)
        domain_s = domain_s.to(self.device)
        input_t = input_t.to(self.device)
        domain_t = domain_t.to(self.device)

        return input_s, label_s, domain_s, input_t, domain_t

    def cdan_loss(self, fea_di, soft_out, domain_label, alpha, entropy):
        softmax_output = soft_out.detach()
        feature = fea_di
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = self.Dis(op_out.view(-1, softmax_output.size(1) * feature.size(1)), alpha)
        dc_target = domain_label
        if entropy is not None:
            entropy.register_hook(grl_hook(alpha))
            entropy = 1.0 + torch.exp(-entropy)
            weight = torch.zeros_like(entropy)
            for i in range(self.num_source_domains):
                mask_i = (domain_label == i).float()
                source_weight_i = entropy * mask_i
                weight = weight + source_weight_i/torch.sum(source_weight_i).detach().item()

            mask_t = (domain_label == self.num_source_domains).float()
            target_weight = entropy * mask_t
            weight = weight + target_weight/torch.sum(target_weight).detach().item()
            # source_mask = torch.ones_like(entropy)
            # source_mask[feature.size(0) // 2:] = 0
            # source_weight = entropy * source_mask
            # target_mask = torch.ones_like(entropy)
            # target_mask[0:feature.size(0) // 2] = 0
            # target_weight = entropy * target_mask
            # weight = source_weight / torch.sum(source_weight).detach().item() + \
            #          target_weight / torch.sum(target_weight).detach().item()
            return torch.sum(weight * nn.CrossEntropyLoss(reduction='none')(ad_out, dc_target)) / (torch.sum(weight).detach().item())
        else:
            return nn.CrossEntropyLoss()(ad_out, dc_target)

    def model_inference(self, input):
        f = self.G(input)
        p = self.C_di(self.D_di(f))
        # p = self.Ct(self.G(input))
        return p

    @torch.no_grad()
    def test_cdi_vs_ct(self):
        self.set_model_mode('eval')
        domain_invariant = self.C_di.weight
        domain_specific = self.C_ds.weight.t()


        target_domain_specific = domain_specific[:, self.num_source_domains:self.num_source_domains + 1] \
            .t().expand(self.num_classes, -1)


        target_fea = torch.cat((domain_invariant, target_domain_specific), 1)
        target_r = self.R(target_fea)
        self.Ct.weight = nn.Parameter(target_r)


        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.Ct(self.G(input))
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

    @torch.no_grad()
    def test_cs1_vs_ct(self):
        self.set_model_mode('eval')
        domain_invariant = self.C_di.weight
        domain_specific = self.C_ds.weight.t()

        source_domain_specific = domain_specific[:, 0:1].t().expand(self.num_classes, -1)

        source_fea = torch.cat((domain_invariant, source_domain_specific), 1)
        source_r = self.R(source_fea)
        self.Cs[0].weight = nn.Parameter(source_r)

        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.Cs[0](self.G(input))
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

    @torch.no_grad()
    def test_cs2_vs_ct(self):
        self.set_model_mode('eval')
        domain_invariant = self.C_di.weight
        domain_specific = self.C_ds.weight.t()

        source_domain_specific = domain_specific[:, 1:2].t().expand(self.num_classes, -1)

        source_fea = torch.cat((domain_invariant, source_domain_specific), 1)
        source_r = self.R(source_fea)
        self.Cs[1].weight = nn.Parameter(source_r)

        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.Cs[1](self.G(input))
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

    @torch.no_grad()
    def tsne(self, model_dir):
        '''
        please note that when using this function,
        we should regard all domains (both source and target domains) as source domain
        '''

        from sklearn.manifold import TSNE
        import matplotlib.pylab as plt
        import os.path as osp
        from da.utils import load_checkpoint

        model_file = 'model-best.pth.tar'

        model_G_path = osp.join(model_dir, 'G', model_file)
        model_D_path = osp.join(model_dir, 'D_di', model_file)

        state_dict_G = load_checkpoint(model_G_path)
        state_dict_D = load_checkpoint(model_D_path)

        state_dict_G = state_dict_G['state_dict']
        state_dict_D = state_dict_D['state_dict']

        self._models['G'].load_state_dict(state_dict_G)
        self._models['D_di'].load_state_dict(state_dict_D)


        self.set_model_mode('eval')

        train_loader_s_iter = iter(self.train_loader_s)
        # train_loader_t_iter = iter(self.train_loader_t)

        len_train_loader_s = len(self.train_loader_s)
        # len_train_loader_t = len(self.train_loader_t)

        feature_s = torch.Tensor().float().cuda()
        label_s = torch.Tensor().float().cuda()
        tmp = set()
        for batch_s in self.train_loader_s:
            input_s = batch_s['img'].cuda()
            domain_s = batch_s['domain'].float().cuda()
            dname = batch_s['dname']
            for i in range(input_s.size(0)):
                tmp.add((domain_s[i].cpu().item(), dname[i]))
            feature_di = self.D_di(self.G(input_s))
            feature_s = torch.cat((feature_s, feature_di), 0)
            label_s = torch.cat((label_s, domain_s), 0)

        # feature_t = torch.Tensor().float()
        # label_t = torch.Tensor().float()
        #
        # for i in range(len_train_loader_t):
        #     batch_t = next(train_loader_t_iter)
        #     input_t = batch_t['img']
        #     domain_t = batch_t['domain']
        #     feature_di = self.D_di(self.G(input_t))
        #     feature_t = torch.cat((feature_t, feature_di), 0)
        #     label_t = torch.cat((label_t, domain_t), 0)

        # feature = torch.cat((feature_s, feature_t), 0)
        # label = torch.cat((label_s, label_t), 0)

        tsne = TSNE(n_components=2)

        embs = tsne.fit_transform(feature_s.cpu().numpy())
        label_s = label_s.cpu().numpy()
        X, Y = embs[:, 0], embs[:, 1]
        mapd= {domain:dname for domain,dname in tmp}

        # for x, y, l in zip(X, Y, label_s):
        #     c = cm.rainbow(int(255/n_domain)*l)
        #     plt.text(x, y, l, backgroundcolor=c, fontsize=9)

        # plt.xlim(X.min(), X.max())
        # plt.ylim(Y.min(), Y.max())
        # plt.

        # for domain, dname in tmp:
        #     print(domain, dname)
        map = {float(0): 'r',
               float(1): 'lime',
               float(2): 'b'}


        # ax = fig.add_subplot(111)
        print(X.size)
        # for domain, dname in tmp:
        #     index = (label_s == domain)
        #     X_i = X[index]
        #     Y_i = Y[index]
        #     print(domain, dname, X_i.size)
        #     ax.scatter(X_i, Y_i, c=map[domain], label=dname)
        index_0 = (label_s == float(0))
        X_0 = X[index_0]
        Y_0 = Y[index_0]
        c_0 = map[float(0)]
        d_0 = mapd[float(0)]
        print(float(0), d_0, X_0.size)

        index_1 = (label_s == float(1))
        X_1 = X[index_1]
        Y_1 = Y[index_1]
        c_1 = map[float(1)]
        d_1 = mapd[float(1)]
        print(float(1), d_1, X_1.size)

        index_2 = (label_s == float(2))
        X_2 = X[index_2]
        Y_2 = Y[index_2]
        c_2 = map[float(2)]
        d_2 = mapd[float(2)]
        print(float(2), d_2, X_2.size)

        # fig = plt.figure()
        # ax =fig.add_subplot(111)
        # ax.scatter(X_0, Y_0, c=c_0, label=d_0)
        # ax.scatter(X_1, Y_1, c=c_1, label=d_1)
        # plt.axis('off')
        # plt.legend(loc='upper right')
        # plt.show()
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(X_0, Y_0, c=c_0, label=d_0)
        # ax.scatter(X_2, Y_2, c=c_2, label=d_2)
        # plt.axis('off')
        # plt.legend(loc='upper right')
        # plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X_0, Y_0, c=c_0, label=d_0, marker='.')
        ax.scatter(X_1, Y_1, c=c_1, label=d_1, marker='.')
        ax.scatter(X_2, Y_2, c=c_2, label=d_2, marker='.')




        # ax.scatter(X, Y, c=map[label_s], marker="*")



        plt.axis('off')

        plt.legend(loc='upper right')
        plt.show()

        #
        # plt.savefig("/home/buerzlh/Desktop/tsne.png")










        











