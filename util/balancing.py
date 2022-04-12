import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from moving_targets.callbacks import DataLogger


class ClassesHistogram(DataLogger):
    def __init__(self, num_columns=4, **plt_kwargs):
        super().__init__()
        self.num_columns = num_columns
        self.plt_kwargs = plt_kwargs

    def on_training_end(self, macs, x, y, p, val_data):
        # store class targets instead of class probabilities
        super(ClassesHistogram, self).on_training_end(macs, x, y, p.argmax(axis=1), val_data)

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        # for each iteration, we we plot the classes counts for both the predictions and the adjusted targets
        for it in self.iterations:
            ax = plt.subplot(num_rows, self.num_columns, it + 1, sharex=ax, sharey=ax)
            # this check is necessary to handle the pretraining step, where no adjusted target is present
            column, name = ('y', 'targets') if it == 0 else (f'z{it}', 'adjusted')
            data = np.concatenate((self.data[column].values, self.data[f'p{it}'].values))
            hue = np.concatenate((len(self.data) * [name], len(self.data) * ['predictions']))
            sns.countplot(x=data, hue=hue, ax=ax)
            ax.set(xlabel='class', ylabel='count')
            ax.set_title(f'iteration: {it}')
        plt.show()
