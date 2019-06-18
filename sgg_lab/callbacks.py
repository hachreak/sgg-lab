
"""Callbacks."""

import numpy as np

from keras.callbacks import Callback


def avg_logs(logs, filter_cond):
    """Average accuracy from logs."""
    return np.average([v for k, v in logs.items() if filter_cond(k)])


def filter_val_accs(l):
    """Filter condition: only validation accuracy."""
    return l.startswith('val_') and l.endswith('acc')


def filter_val_f1score(l):
    """Filter condition: only validation f1score."""
    return l.startswith('val_') and l.endswith('f1_score')


class ModelSaveBestAvgAcc(Callback):

    def __init__(self, filepath, verbose=False, cond=None):
        """Init."""
        self._acc = 0
        self._filepath = filepath
        self._verbose = verbose
        self._cond = cond or filter_val_accs

    def on_epoch_end(self, epoch, logs=None):
        """Save best model checking every epoch."""
        acc = avg_logs(logs, self._cond) if logs else 0
        if acc > self._acc:
            filepath = self._filepath.format(epoch=epoch+1, avgacc=acc, **logs)
            if self._verbose:
                print('\nEpoch %05d: saving model to %s' % (
                    epoch + 1, filepath))
            self._acc = acc
            self.model.save(filepath, overwrite=True)
