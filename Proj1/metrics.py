import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """
    Computes the classification accuracy of a model.
    Args:
        model (nn.Module): model of interest with output (prediction, auxiliary)
        loader (DataLoader): data loader with sample structure
            that follows unpacking (input, target, classes)
    Returns:
        float: classification accuracy
    """
    
    accuracy = 0.
    counter = 0
    
    model.eval()
    
    with torch.no_grad():
        for (input, target, _) in loader:
            output, _ = model(input)
            
            accuracy += (output >= 0.5) == target
            counter += target.size(0)
                
    return (accuracy.sum() / counter).float().item()


class TrainingMetrics:
    """
    Custom class for tracking and plotting training metrics.
    
    Attributes:
        metrics (dict): dictionary that stores metrics on epoch-key
        current (int): current epoch number stored for printing
        run (int): current run when performing trials
        
    Usage:
        1. Define `metrics = TrainingMetrics()` before getting into epoch-loop
        2. Update metrics with `metrics.add_entry(e, l, a, r)`
        3. print(metrics) to display current metrics
    """
    
    def __init__(self) -> None:
        """Initialize with empty attributes"""
        self.metrics = {}
        self.current = None
        self.run = None
    
    def __repr__(self) -> str:
        """
        Representation method

        Returns:
            str: representation
        """
        return (f"TrainingMetrics instance of size {len(self.metrics)}")
    
    def __str__(self) -> str:
        """
        Print method

        Returns:
            str: to print when `print(self)` is called
        """
        metric = self.metrics[f"R{self.run}E{self.current}"]
        return (f"Epoch {metric['epoch']:02} \t"
                f"Loss {metric['loss']:07.3f} \t"
                f"Accuracy {metric['accuracy'] * 100:06.3f}")
        
    def add_entry(self, epoch: int, loss: float, accuracy: float, run: int = 1) -> None:
        """
        Add entry to metrics

        Args:
            epoch (int): current epoch
            loss (float): loss
            accuracy (float): accuracy
            run (int): the current run number (when doing trials)
        """
        self.run = run
        self.current = epoch
        self.metrics[f"R{run}E{epoch}"] = \
            dict(epoch=epoch, loss=loss, accuracy=accuracy, run=run)
            
    def save(self, filename: str = "training_metrics.csv"):
        with open(filename, 'w+') as outfile:
            outfile.write("epoch,loss,accuracy\n")
        
        with open(filename, 'a') as outfile:
            for i, metric in self.metrics.items():
                outfile.write(f"{metric['epoch']},"
                              f"{metric['loss']},"
                              f"{metric['accuracy']}\n")


class TestMetric():
    """[summary]
    
    Attributes:
        model (torch.nn.Module): model to evaluate
        loader (torch.utils.data.DataLoader): data loader
        confusion (dict): confusion matrix
        accuracy (float): testing accuracy
        precision (float): precision
        recall (float): recall
        f1_score (float): F1 score
    """
    
    def __init__(self, model, data_loader):
        """
        Initiate TestinMetrics instance.

        Args:
            model (nn.Module): model of interest with output (prediction, auxiliary)
            data_loader (DataLoader): data loader with sample structure
                that follows unpacking (input, target, classes)
        """
        self.model = model
        self.loader = data_loader
        self.confusion = dict(true_positive=0., false_negative=0.,
                              false_positive=0., true_negative=0.)
        self.accuracy = 0.
        self.precision = 0.
        self.recall = 0.
        self.f1_score = 0.
        
        self.compute()
               
    def compute(self) -> None:
        """Compute different metrics by evaluating the model"""
        
        self.model.eval()
        
        with torch.no_grad():
            for (input, target, _) in self.loader:

                output, _ = self.model(input)
                
                output = (output >= 0.5)
                
                for out, tar in zip(output, target):
                
                    tar = bool(tar)
                    
                    if out and tar:
                        self.confusion['true_positive'] += 1
                    elif not out and not tar:
                        self.confusion['true_negative'] += 1
                    elif out and not tar:
                        self.confusion['false_positive'] += 1
                    elif not out and tar:
                        self.confusion['false_negative'] += 1
        
        self.accuracy = (self.confusion['true_positive'] + self.confusion['true_negative']) \
            / sum(list(self.confusion.values()))
        
        if (self.confusion['true_positive'] + self.confusion['false_positive']) == 0.:
            self.precision = 0.
        else:
            self.precision = self.confusion['true_positive'] \
                / (self.confusion['true_positive'] + self.confusion['false_positive'])
        
        if (self.confusion['true_positive'] + self.confusion['false_negative']) == 0.:
            self.recall = 0.
        else:
            self.recall = self.confusion['true_positive'] \
                / (self.confusion['true_positive'] + self.confusion['false_negative'])
        
        if (self.precision + self.recall) == 0.:
            self.f1_score = 0.
        else:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        
    def __str__(self) -> str:
        """
        Print method

        Returns:
            str: to print when `print(self)` is called
        """
        return f"Acc. {self.accuracy * 100:06.3f} | Prec. {self.precision * 100:06.3f} | " \
            f"Rec. {self.accuracy * 100:06.3f} | F1 {self.f1_score * 100:06.3f}"
            
    def serialize(self) -> dict:
        """Serialize instance into dictionary

        Returns:
            dict: serialized object
        """
        return {
            'true_positive': self.confusion['true_positive'],
            'false_negative': self.confusion['false_negative'],
            'false_positive': self.confusion['false_positive'],
            'true_negative': self.confusion['true_negative'],
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }


class TestingMetrics():
    
    def __init__(self) -> None:
        """Constructor"""
        self.metrics = []
        self.materialized = None
        
    def add_entry(self, model: nn.Module, loader: DataLoader, time_per_epoch: float,
                  verbose: int) -> None:
        
        test_metric = TestMetric(model, loader)
        self.metrics.append(test_metric)
        
        if (verbose > 0):
            print(f"{test_metric} [{time_per_epoch:0.4f} sec. / epoch]")
            
    def materialize(self):
        self.materialized = list(map(lambda m: m.serialize(), self.metrics))
        
    def save(self, filename: str = "testing_metrics.csv"):
        if self.materialized is None:
            self.materialize()
            
        with open(filename, 'w+') as outfile:
            outfile.write("true_positive,false_negative,false_positive,"
                          "true_negative,accuracy,precision,recall,f1_score\n")
        
        with open(filename, 'a') as outfile:
            for t in range(len(self.materialized)):
                outfile.write(f"{self.materialized[t]['true_positive']},"
                              f"{self.materialized[t]['false_negative']},"
                              f"{self.materialized[t]['false_positive']},"
                              f"{self.materialized[t]['true_negative']},"
                              f"{self.materialized[t]['accuracy']},"
                              f"{self.materialized[t]['precision']},"
                              f"{self.materialized[t]['recall']},"
                              f"{self.materialized[t]['f1_score']}\n")
