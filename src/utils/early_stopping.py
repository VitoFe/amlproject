from typing import Optional


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Metric improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"EarlyStopping: Stopping! Best was {self.best_score:.4f} at epoch {self.best_epoch}"
                    )
                return True

        return False

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:  # mode == 'min'
            return score < self.best_score - self.min_delta

    def reset(self):
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
