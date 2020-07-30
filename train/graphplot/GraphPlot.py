import matplotlib.pyplot as plt
import numpy as np

class GraphPlot:
    def __init__():
        pass

    def write_graph(history, write_enable, datetime, validation = False):
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
        plt.savefig("./result/" + datetime +"/" + "accuracy.png")

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
        plt.savefig("./result/" + datetime +"/loss.png")