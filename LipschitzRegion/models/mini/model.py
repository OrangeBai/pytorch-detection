from core.utils import *
from models.base_model import BaseModel
from models.mini.vgg import *
from models.dnn.model import DNN


class Model(BaseModel):
    def __init__(self, args, logger):
        super(Model, self).__init__(args, logger)
        self.args = args
        self.model = self.build_up().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        self.lr_scheduler = self.init_scheduler()
        self.loss_function = nn.CrossEntropyLoss()
        self.metrics = self.set_metrics()

    @staticmethod
    def set_metrics():
        metrics = MetricLogger()
        metrics.add_meter('top1', SmoothedValue(window_size=20))
        metrics.add_meter('loss', SmoothedValue(window_size=20))
        metrics.add_meter('lr', SmoothedValue(window_size=1, fmt="{value:.4f}"))
        return metrics

    def build_up(self):
        return DNN(self.args, None)

    def forward(self, x):
        return self.model.features(x)

    def train_step(self, images, labels):
        labels = labels.cuda()
        images = images.cuda()

        # train the model
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.loss_function(outputs, labels)
        loss.backward(retain_graph=True)

        self.optimizer.step()
        # Record Metrics
        top1, top5 = accuracy(outputs, labels)
        self.metrics.update(top1=(top1, len(images)), loss=(loss, len(images)),
                            lr=(self.optimizer.param_groups[0]['lr'], 1))

    def get_variable(self, var_type):
        """
        Get current weights of the model
        :return:
        """
        cur_variable = {}
        for name, module in self.model.named_modules():
            if 'Linear' in name:
                if var_type == 'weight':
                    cur_variable[name] = module.weight.cpu().detach().numpy()
                elif var_type == 'grad':
                    cur_variable[name] = module.weight.grad.cpu().detach().numpy()
        return cur_variable
