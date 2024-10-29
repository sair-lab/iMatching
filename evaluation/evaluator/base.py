from abc import ABC
from typing import Any, Dict, Generic, List, Optional, TypeVar
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl


METRIC_TY = TypeVar('METRIC_TY')


class EvaluatorBase(nn.Module, Generic[METRIC_TY]):
    def __init__(self, module: pl.LightningModule, monitor_name: Optional[str] = None, compute_on_cpu=False) -> None:
        super().__init__()
        self._module = [module]
        self.dataloader_idx_name_mapping: Dict[int, str] = {}
        self._metrics: List[METRIC_TY] = nn.ModuleList() if not compute_on_cpu else list()
        self.cur_dataloader_idx: METRIC_TY = None
        self.monitor_name = monitor_name

    @property
    def module(self) -> pl.LightningModule:
        return self._module[0]

    @property
    def cur_metric(self) -> METRIC_TY:
        return self._metrics[self.cur_dataloader_idx]

    @property
    def metrics(self) -> List[METRIC_TY]:
        return self._metrics[:len(self.dataloaders)]

    @property
    def dataloaders(self) -> List[DataLoader]:
        if self.module.trainer.state.stage == "test":
            return self.module.trainer.test_dataloaders
        elif self.module.trainer.state.stage in ("validate", "sanity_check"):
            return self.module.trainer.val_dataloaders

    @property
    def cur_dataloader_name(self) -> str:
        return self.dataloader_idx_name_mapping[self.cur_dataloader_idx]

    @property
    def logger(self):
        return self.module.logger

    @property
    def stage_name(self):
        return self.module.trainer.state.stage.value

    def create_matric(self) -> METRIC_TY:
        raise NotImplementedError()

    def get_dataloader_name(self, batch: Any) -> str:
        _, (series_key, _) = batch

        if isinstance(series_key[0][0], str):
            return "+".join(part[0] for part in series_key)
        elif isinstance(series_key[0][0], tuple):
            return "_".join(part[0][0] for part in series_key)
        else:
            return str(series_key)

    def on_start(self):
        # add extra metric evaluators if necessary
        for _ in range(len(self.dataloaders) - len(self._metrics)):
            self._metrics.append(self.create_matric())
        self.to(self.module.device)

    def step_impl(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        pass

    def step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # record dataloader name
        self.dataloader_idx_name_mapping[dataloader_idx] = self.get_dataloader_name(batch)
        self.cur_dataloader_idx = dataloader_idx
        return self.step_impl(batch, batch_idx, dataloader_idx)

    def epoch_end(self, outputs):
        pass

    def log_monitor(self, value):
        if self.monitor_name is not None and self.stage_name == "validate":
            self.module.log(f"monitor/{self.monitor_name}", value, sync_dist=True)

    def format_stage_metric_name(self, name: str):
        return f"{self.stage_name}/{name}"

    def log_stage_metric(self, name: str, value):
        if self.stage_name == "test":
            # pl doesn't log steps at test time
            self.logger.log_metrics({self.format_stage_metric_name(name): value}, self.module.global_step)
        else:
            if isinstance(value, dict):
                self.module.log_dict({self.format_stage_metric_name(k): v for k, v in value.items()}, sync_dist=True)
            else:
                self.module.log(self.format_stage_metric_name(name), value, sync_dist=True)

    def reset(self):
        for metric in self._metrics:
            if isinstance(metric, torchmetrics.Metric):
                metric.reset()

    def metric_weighted_mean(self):
        if all(m._update_count == 0 and len(l) > 0 for m, l in zip(self._metrics, self.dataloaders)):
            return None

        metric_cum, metric_num = 0, 0
        for i, (metric, loader) in enumerate(zip(self._metrics, self.dataloaders)):
            n_data = len(loader.dataset)
            err = metric.compute()
            metric_cum += err * n_data
            metric_num += n_data
        return metric_cum / metric_num
