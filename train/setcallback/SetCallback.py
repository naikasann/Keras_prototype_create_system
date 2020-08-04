import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

class SetCallback:
    def __init__(self, yml, processingresult, learningrateshceduler):
        self.yml = yml
        self.modelcheckpoint = self.set_modelcheckpoint(processingresult)
        self.TensorBoard = self.set_tensorborad()
        self.lr_decay = self.set_lr_scheduler(learningrateshceduler)

    def getcall_back(self):
        return [self.modelcheckpoint, self.TensorBoard, self.lr_decay]

    def set_modelcheckpoint(self, processingresult):
        # callback function(tensorboard, modelcheckpoint) setting.
        # first modelcheckpoint setting
        execute_time = processingresult.get_executetime()
        modelCheckpoint = ModelCheckpoint(
                                    filepath = "./result/"+ execute_time + "/model/" + self.yml["Trainingresult"]["model_name"] +"_{epoch:02d}.h5",
                                    monitor=self.yml["callback"]["monitor"],
                                    verbose=self.yml["callback"]["verbose"],
                                    save_best_only=self.yml["callback"]["save_best_only"],
                                    save_weights_only=self.yml["callback"]["save_weights_only"],
                                    mode=self.yml["callback"]["mode"],
                                    period=self.yml["callback"]["period"])
        return modelCheckpoint

    def set_tensorborad(self):
        # tensorboard setting.
        tensorboard = TensorBoard(log_dir = self.yml["callback"]["tensorboard"], histogram_freq=self.yml["callback"]["tb_epoch"])
        return tensorboard

    def set_lr_scheduler(self, lr_decay_func):
        # learning rate shceduler setting.
        lr_decay = LearningRateScheduler(lr_decay_func, verbose=1)
        return lr_decay