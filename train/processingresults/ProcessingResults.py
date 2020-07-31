import os
import shutil
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

class ProcessingResults:
    def __init__(self):
        self.execute_time = dt.datetime.now().strftime("%m_%d_%H_%M")

    def makedir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def get_executetime(self):
        return self.execute_time

    def Prepareresult(self):
        # Store the execution time in a variable.
        self.makedir("result/")
        self.makedir("result/" + self.execute_time)
        self.makedir("result/" + self.execute_time + "/model")
        # Copy the YAML file that contains the execution environment.
        shutil.copy("config.yaml", "result/" + self.execute_time)

    def PreservationResult(self, model, modelname):
        # save weights and model.
        model.save("./result/" + self.execute_time + "/model/" + modelname + "_end_epoch.h5")
        model.save_weights("./result/" + self.execute_time + "/model/" + modelname + "end_model_weight.h5")

        # write network architecture.
        json_model = model.to_json()
        open("./result/" + self.execute_time + "/model/model_json.json", 'w').write(json_model)
        yaml_model = model.to_yaml()
        open("./result/" + self.execute_time + "/model/model_yaml.yaml", 'w').write(yaml_model)

    def write_graph(self, history, write_enable, validation = False):
        # Draw and save the accuracy graph.
        plt.figure(figsize=(6,4))
        plt.plot(history.history['accuracy'])
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if validation:
            plt.plot(history.history['val_accuracy'])
            plt.legend(['traindata', 'validata'], loc='upper left')
        else:
            plt.legend(['traindata'], loc='upper left')
        if write_enable:
            plt.show()
        plt.savefig("./result/" + self.execute_time +"/" + "accuracy.png")

        # Draw and save the loss graph.
        plt.figure(figsize=(6,4))
        plt.plot(history.history['loss'])
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        if validation:
            plt.plot(history.history['val_loss'])
            plt.legend(['traindata', 'valdata'], loc='upper left')
        else:
            plt.legend(['traindata'], loc='upper left')
        if write_enable:
            plt.show()
        plt.savefig("./result/" + self.execute_time +"/loss.png")