import pytorch_lightning as pl

class MetricHistoryCallback(pl.Callback):
    def __init__(self, metrics):
        self.metrics = metrics
        self.metric_history = {metric: [] for metric in self.metrics}

    def on_epoch_end(self, trainer):
        # Access and append the values to the history
        for metric_name in self.metrics:
            metric_value = trainer.callback_metrics.get(metric_name)  # Get the metric value
            if metric_value is not None:
                self.metric_history[metric_name].append(metric_value)

