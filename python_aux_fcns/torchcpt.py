import os
import torch

class ModelCheckpoint:
    def __init__(self, save_path='local_checkpoints', monitor='accuracy', mode='max', log_file='performance_log.txt'):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_model = None
        self.log_file = log_file

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def update(self, model, epoch, score):
        self._save_model(model, f'checkpoint_epoch_{epoch}.pth')
        is_best = False

        if (self.mode == 'max' and (self.best_score is None or score > self.best_score)) or \
           (self.mode == 'min' and (self.best_score is None or score < self.best_score)):
            self.best_score = score
            self.best_model = model
            self._save_model(model, f'best_model.pth')
            is_best = True

        self._log_performance(epoch, score, is_best)

    def _save_model(self, model, filename):
        torch.save(model.state_dict(), os.path.join(self.save_path, filename))

    def _log_performance(self, epoch, score, is_best):
        with open(os.path.join(self.save_path, self.log_file), 'a') as log_file:
            log_file.write(f'Epoch {epoch}: {self.monitor} = {score}, Best so far: {is_best}\n')
