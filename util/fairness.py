import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from moving_targets.callbacks import DataLogger
from moving_targets.util import probabilities


class RegressionCallback(DataLogger):
    def __init__(self, protected, num_columns=3, **plt_kwargs):
        # protected   : the name of the protected feature
        # num_columns : the number of columns to display the subplots
        # plt_kwargs  : custom arguments to be passed to the 'matplotlib.pyplot.plot' function

        super().__init__()
        self.protected = protected
        self.num_columns = num_columns
        self.plt_kwargs = plt_kwargs

    def on_process_start(self, macs, x, y, val_data):
        # retrieve the subset of input features regarding the protected groups (in case of multiple groups, the index
        # must be obtained via argmax) and store them in the inner 'data' variable
        super(RegressionCallback, self).on_process_start(macs, x, y, val_data)
        group = x[[c for c in x.columns if c.startswith(self.protected)]].values.squeeze().astype(int)
        self.data['group'] = group.argmax(axis=1) if group.ndim == 2 else group

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        for it in self.iterations:
            ax = plt.subplot(num_rows, self.num_columns, it + 1, sharex=ax, sharey=ax)
            # this check is necessary to handle the pretraining step, where no adjusted target is present
            column, name = ('y', 'targets') if it == 0 else (f'z{it}', 'adjusted')
            data = pd.DataFrame.from_dict({
                'group': np.concatenate((self.data['group'].values, self.data['group'].values)),
                'targets': np.concatenate((self.data[column].values, self.data[f'p{it}'].values)),
                'hue': np.concatenate((len(self.data) * [name], len(self.data) * ['predictions']))
            })
            sns.boxplot(data=data, x='group', y='targets', hue='hue', ax=ax)
            ax.set_title(f'iteration: {it}')
        plt.show()


class ClassificationCallback(DataLogger):
    def __init__(self, protected, num_columns=3, **plt_kwargs):
        # protected   : the name of the protected feature
        # num_columns : the number of columns to display the subplots
        # plt_kwargs  : custom arguments to be passed to the 'matplotlib.pyplot.plot' function

        super().__init__()
        self.protected = protected
        self.num_columns = num_columns
        self.plt_kwargs = plt_kwargs

    def on_process_start(self, macs, x, y, val_data):
        # retrieve the subset of input features regarding the protected groups and store them in the inner 'data'
        # variable by replacing the input data which will not be useful during the plotting
        group = x[[c for c in x.columns if c.startswith(self.protected)]].values.squeeze().astype(int)
        self.data['group'] = group.argmax(axis=1) if group.ndim == 2 else group

    def on_training_end(self, macs, x, y, p, val_data):
        # store class targets instead of class probabilities
        super(ClassificationCallback, self).on_training_end(macs, x, y, probabilities.get_classes(p), val_data)

    def on_process_end(self, macs, x, y, val_data):
        plt.figure(**self.plt_kwargs)
        num_rows = int(np.ceil(len(self.iterations) / self.num_columns))
        ax = None
        for it in self.iterations:
            ax = plt.subplot(num_rows, self.num_columns, it + 1, sharex=ax, sharey=ax)
            # we do not use count plot since we would like to show the percentage of predicted classes per group in
            # order to better see how moving targets affect the class balancing, thus we use instead a standard bar
            # plot to plot the bars with normalized value counts
            groups = self.data.rename(columns={f'p{it}': 'prediction'}).groupby('group')
            groups['prediction'].value_counts(normalize=True).mul(100).unstack().plot(kind='bar', stacked=True, ax=ax)
            ax.set(xlabel='group', ylabel='%')
            ax.set_title(f'iteration: {it}')
        plt.show()
