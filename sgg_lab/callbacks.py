
"""Callbacks."""

import numpy as np

from keras.callbacks import Callback


def avg_logs(logs, filter_cond):
    """Average accuracy from logs."""
    return np.average([v for k, v in logs.items() if filter_cond(k)])


class ModelSaveBestAvgAcc(Callback):

    def __init__(self, filepath, verbose=False):
        """Init."""
        self._acc = 0
        self._filepath = filepath
        self._verbose = verbose
        self._cond = lambda l: l.startswith('val_') and l.endswith('acc')

    def on_epoch_end(self, epoch, logs=None):
        """Save best model checking every epoch."""
        acc = avg_logs(logs, self._cond) if logs else 0
        if acc > self._acc:
            filepath = self._filepath.format(epoch=epoch + 1, acc=acc, **logs)
            if self._verbose:
                print('\nEpoch %05d: saving model to %s' % (
                    epoch + 1, filepath))
            self._acc = acc
            self.model.save(filepath, overwrite=True)
