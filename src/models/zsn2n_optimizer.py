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
        # オプティマイザの選択
        if self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # スケジューラの選択
        if self.scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        elif self.scheduler_type == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **self.scheduler_params)
        elif self.scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        elif self.scheduler_type == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **self.scheduler_params)
        elif self.scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.scheduler_params)
        elif self.scheduler_type is None:
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
