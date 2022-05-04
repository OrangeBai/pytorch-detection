import torch.nn as nn
from core.utils import *


class BaseModel(nn.Module):
    def __init__(self, args, logger):
        super(BaseModel, self).__init__()
        self.args = args
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer, self.lr_scheduler = None, None

        self.logger = logger
        self.result = np.zeros([self.args.num_epoch, 4])

    def save_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        torch.save(self.model.state_dict(), model_path)
        return

    def load_model(self, path, name=None):
        if not name:
            model_path = os.path.join(path, 'weights.pth')
        else:
            model_path = os.path.join(path, 'weights_{}.pth'.format(name))
        self.model.load_state_dict(torch.load(model_path), strict=False)
        return

    def save_result(self, path, name=None):
        if not name:
            res_path = os.path.join(path, 'result')
        else:
            res_path = os.path.join(path, 'result_{}'.format(name))
        np.save(res_path, self.result)

    def init_scheduler(self):
        if self.args.lr_scheduler == 'milestones':
            milestones = [milestone * self.args.num_epoch for milestone in self.args.milestones]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        elif self.args.lr_scheduler == 'static':
            def lambda_rule(t):
                return 1.0

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.args.lr_scheduler == 'exp':
            gamma = math.pow(1 / 100, 1 / self.args.num_epoch)
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
        elif self.args.lr_scheduler == 'cycle':
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.1,
                                                                  step_size_up=self.args.total_steps / 3,
                                                                  step_size_down=2 * self.args.total_steps / 3)
        elif self.args.lr_scheduler == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0001, max_lr=0.1,
                                                                  step_size_up=1,
                                                                  step_size_down=self.args.total_steps)
        return

    def train_model(self, train_loader, test_loader):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        self.init_scheduler()
        for cur_epoch in range(self.args.num_epoch):
            train_time, train_loss, train_acc = self.train_epoch(cur_epoch, train_loader)
            test_time, test_loss, test_acc = self.validate(cur_epoch, test_loader)
            self.result[cur_epoch] = [train_loss, train_acc, test_loss, test_acc]
            self.lr_scheduler.step()
        self.save_model(self.args.model_dir, name=str(self.args.exp_id))
        self.save_result(self.args.model_dir, name=str(self.args.exp_id))
        return

    def train_epoch(self, cur_epoch, train_loader):
        self.model.train()
        epoch_start = time.time()
        loss_sum = 0
        correct = 0
        total = 0
        num_iter = len(train_loader)
        msg = 'Epoch {}/{}, iter {}/{}, avg time: {:.4f}, eta: {:.4f}, lr: {:.4f}, average loss:{:4f}, accuracy: {:.4f}'
        for batch_index, (images, labels) in enumerate(train_loader):
            batch_start = time.time()
            labels = labels.cuda()
            images = images.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, preds = outputs.max(1)
            loss_sum += loss.detach().cpu()
            correct += preds.eq(labels).sum().cpu()
            total += images.shape[0]
            lr = self.optimizer.param_groups[0]['lr']
            batch_time = time.time() - batch_start

            if batch_index % self.args.print_every == 0:
                out_msg = msg.format(cur_epoch, self.args.num_epoch, batch_index, num_iter,
                                     batch_time, batch_time * num_iter, lr, loss / total, correct / total)
                print(out_msg)
                self.logger.info(out_msg)

        train_time = time.time() - epoch_start
        train_loss = loss_sum / total
        train_acc = correct / total

        msg = 'Train epoch {}/{}, Time: {:.4f}, Loss:{:.4f}, Accuracy:{:.4f}. Validation start'
        msg2 = msg.format(cur_epoch, self.args.num_epoch, train_time, train_loss, train_acc)
        print(msg2)
        self.logger.info(msg2)

        return train_time, train_loss, train_acc

    @torch.no_grad()
    def validate(self, cur_epoch, test_loader):
        eval_start = time.time()
        self.model.eval()

        loss_sum = 0
        correct = 0
        total = 0

        for (images, labels) in test_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)

            loss_sum += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().cpu()
            total += images.shape[0]

        test_time = time.time() - eval_start
        test_loss = loss_sum / total
        test_acc = correct / total

        msg = 'Test epoch {}/{}, Time: {:.4f}, Loss:{:.4f}, Accuracy:{:.4f}.'
        msg2 = msg.format(cur_epoch, self.args.num_epoch, test_time, test_loss, test_acc)
        print(msg2)
        self.logger.info(msg2)
        return test_time, test_loss, test_acc