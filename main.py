
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
import pickle

from Contrastive.ContrastiveTransformations import ContrastiveTransformations
from metric import metric
import configs
import backbones
import dataloader


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = 'cuda:1'

gpus = [1]
default_opt = getattr(configs,'Defaultargs')().get_args()
save_dirpath = 'check_points/'

class ModelTrainer(pl.LightningModule):

    def __init__(self, n_node):
        super().__init__()
        self.best_res = [0, 0, 0]
        self.default_opt = default_opt
        self.model_opt = getattr(configs,self.default_opt.model+'args')().get_args()
        self.model = getattr(backbones, self.default_opt.model)(self.default_opt,self.model_opt,n_node)
        self.loss_function = nn.CrossEntropyLoss()
        self.loss = nn.Parameter(torch.Tensor(1))
        dict_association_rules = None
        self.transform = ContrastiveTransformations(dict_association_rules)


    def forward(self, *args):
        return self.model(*args)

    def sigmoid(self,tensor, temp=1.0):
        """ temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def sequence_augmentation(self,session,mul_len):
        """
        The code will be publicly available after the paper is accepted.
        :return: scores_ext
        """

    def computing_LCL(self,scores,scores_ext,sigmoid_temp):
        """
        The code will be publicly available after the paper is accepted.
        :return: lcl_loss
        """
        pass

    def computing_Lrec(self,scores,scores_ext,targets,sigmoid_temp):
        """
        The code will be publicly available after the paper is accepted.
        :return: rec_loss
        """
        pass


    def training_step(self, batch, batch_idx):
        sigma = 0.01
        sigmoid_temp = 100
        mul_len = int(1 * self.current_epoch)

        targets, session, lens = batch

        if default_opt.model == 'CORE':
            feat_session, feat_test_items,_ = self.model(session, lens)
            scores = torch.matmul(feat_session, feat_test_items)
            scores = scores / self.model_opt.temperature
        else:
            feat_session, feat_test_items = self.model(session, lens)
            scores = torch.matmul(feat_session, feat_test_items)
        # rec_loss = self.loss_function(scores, targets)

        # 1. Sequence Augmentation
        scores_ext = self.sequence_augmentation(session,mul_len)

        ### 2. Listwise Contrastive Leaning
        lcl_loss = self.computing_LCL(scores,scores_ext,sigmoid_temp)

        ### 3. Training with Listwise Regularization
        rec_loss = self.computing_Lrec(scores,scores_ext,targets,sigmoid_temp)

        loss = rec_loss + sigma * lcl_loss
        return loss

    def validation_step(self, batch, batch_idx):
        targets, session,lens = batch
        if default_opt.model == 'CORE':
            feat_session, feat_test_items,_ = self.model(session, lens)
            scores = torch.matmul(feat_session, feat_test_items)
            scores = scores / self.model_opt.temperature
        else:
            feat_session, feat_test_items = self.model(session, lens)
            scores = torch.matmul(feat_session, feat_test_items)
        res_list = []
        for k in [5,10,20]:
            res_list.append(torch.tensor(metric(scores,targets,k)))
        res = torch.cat(res_list,dim=1)
        return res

    def validation_epoch_end(self, validation_step_outputs):

        output = torch.cat(validation_step_outputs, dim=0)
        hit5 = torch.mean(output[:, 0]) * 100
        mrr5 = torch.mean(output[:, 1]) * 100
        ndcg5 = torch.mean(output[:, 2]) * 100
        hit10 = torch.mean(output[:, 3]) * 100
        mrr10 = torch.mean(output[:, 4]) * 100
        ndcg10 = torch.mean(output[:, 5]) * 100
        hit20 = torch.mean(output[:, 6]) * 100
        mrr20 = torch.mean(output[:, 7]) * 100
        ndcg20 = torch.mean(output[:, 8]) * 100
        if hit20 > self.best_res[0]:
            self.best_res[0] = hit20
        if mrr20 > self.best_res[1]:
            self.best_res[1] = mrr20
        if ndcg20 > self.best_res[2]:
            self.best_res[2] = ndcg20
        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        self.log('ndcg@20', self.best_res[2])
        msg = ' \n Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(5, hit5, mrr5, ndcg5)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(10, hit10, mrr10, ndcg10)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(20, hit20, mrr20, ndcg20)
        print("EPOCH:",self.current_epoch)
        self.print(msg)
        return mrr20

    def test_step(self, batch, idx):
        targets, session,lens = batch
        if default_opt.model == 'CORE':
            feat_session, feat_test_items,_ = self.model(session, lens)
            scores = torch.matmul(feat_session, feat_test_items)
            scores = scores / self.model_opt.temperature
        else:
            feat_session, feat_test_items = self.model(session, lens)
            scores = torch.matmul(feat_session, feat_test_items)
        res_list = []
        for k in [5,10,20]:
            res_list.append(torch.tensor(metric(scores,targets,k)))
        res = torch.cat(res_list,dim=1)
        return res

    def test_epoch_end(self, test_step_outputs):
        output = torch.cat(test_step_outputs, dim=0)
        hit5 = torch.mean(output[:, 0]) * 100
        mrr5 = torch.mean(output[:, 1]) * 100
        ndcg5 = torch.mean(output[:, 2]) * 100
        hit10 = torch.mean(output[:, 3]) * 100
        mrr10 = torch.mean(output[:, 4]) * 100
        ndcg10 = torch.mean(output[:, 5]) * 100
        hit20 = torch.mean(output[:, 6]) * 100
        mrr20 = torch.mean(output[:, 7]) * 100
        ndcg20 = torch.mean(output[:, 8]) * 100
        msg = ' \n Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(5, hit5, mrr5, ndcg5)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(10, hit10, mrr10, ndcg10)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(20, hit20, mrr20, ndcg20)
        self.print(msg)

        return mrr20


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.default_opt.lr, weight_decay=self.default_opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.default_opt.lr_dc_step, gamma=self.default_opt.lr_dc)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

if __name__ == "__main__":
    pl.seed_everything(default_opt.seed)
    if default_opt.dataset == 'Diginetica':
        n_node = 42596 + 1
    elif default_opt.dataset == 'Nowplaying':
        n_node = 60416 + 1
    elif default_opt.dataset == 'Tmall':
        n_node = 40727 + 1


    early_stop_callback = EarlyStopping(
        monitor='mrr@20',
        min_delta=0.00,
        patience=default_opt.patience,
        verbose=False,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='mrr@20',
        dirpath=save_dirpath,
        filename=default_opt.dataset + '-' + default_opt.model,
        save_top_k=1,
        mode='max')
    trainer = pl.Trainer(gpus=gpus, deterministic=True, max_epochs=default_opt.epoch, num_sanity_val_steps=2,
                         replace_sampler_ddp=False,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         progress_bar_refresh_rate=0)
    if default_opt.opt_train:
        train_loader, valid_loader, test_loader = getattr(dataloader, default_opt.model + 'Data')(
            dataset=default_opt.dataset, batch_size=default_opt.batch_size).get_loader(default_opt.opt_train)
        model = ModelTrainer(n_node=n_node)
        trainer.fit(model, train_loader, valid_loader)
    else:
        test_loader = getattr(dataloader, default_opt.model + 'Data')(
            dataset=default_opt.dataset, batch_size=default_opt.batch_size).get_loader(default_opt.opt_train)
        model = ModelTrainer.load_from_checkpoint(
            save_dirpath + default_opt.dataset + '-' + default_opt.model + '.ckpt', n_node=n_node)
        model.eval()
    print('Testing')
    trainer.test(model, test_loader)
