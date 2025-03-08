from libero.lifelong.algos.base import Sequential
from diffuser.datasets.libero_dataset import LiberoDataset
from diffuser.utils.training import Trainer
from diffuser.utils.lr_scheduler import get_scheduler

### All lifelong learning algorithm should inherit the Sequential algorithm super class

class NavieLifelongAlgo:
    """
    The experience replay policy.
    """
    def __init__(self,
                 dataset: LiberoDataset,
                 trainer: Trainer,
                 args,
                 **kwargs):
        # define the learning policy
        self.dataset = dataset
        self.trainer = trainer
        self.args = args

    def learn_one_task(self, task_id):
        self.dataset.task_id = task_id
        total_steps = self.args.n_epochs * len(self.dataset)
        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            self.trainer.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=total_steps,
         )
        self.trainer.lr_scheduler = lr_scheduler

        for i in range(self.args.n_epochs):
            print(f'Epoch {i} / {self.args.n_epochs} | {self.args.savepath}')
            self.trainer.train()


    # def start_task(self, task):
    #     # what to do at the beginning of a new task
    #     super().start_task(task)

    # def end_task(self, dataset, task_id, benchmark):
    #     # what to do when finish learning a new task
    #     self.datasets.append(dataset)

    # def observe(self, data):
    #     # how the algorithm observes a data and returns a loss to be optimized
    #     loss = super().observe(data)
    #     return loss