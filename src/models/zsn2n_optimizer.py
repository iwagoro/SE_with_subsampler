import torch


class ZSN2NOptimizer:
    def __init__(self, config):
        self.config = config
        self.optimizer_type = config.training.optimizer.type
        self.optimizer_params = config.training.optimizer.params
        self.scheduler_type = None
        self.scheduler_params = None
        if config.training.get("scheduler") is not None:
            self.scheduler_type = config.training.scheduler.type
            self.scheduler_params = config.training.scheduler.params

    def configure_optimizers(self):
        # Optimizerの設定
        optimizer_cls = getattr(torch.optim, self.optimizer_type, None)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        optimizer = optimizer_cls(self.parameters(), **self.optimizer_params)

        # scheduler_typeが指定されていなければoptimizerのみ返す
        if not self.scheduler_type:
            return optimizer

        # Schedulerの設定
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.scheduler_type, None)
        if scheduler_cls is None:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
        scheduler = scheduler_cls(optimizer, **self.scheduler_params)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
