from dataclasses import dataclass
from typing import Any, Optional

import cv2
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers

from dataset import DataModule
from models.caps import CAPS
from models.patch2pix_wrapper import Patch2PixWrapper
from models.correspondence import CorrespondenceModel, ModelConfig
from models.r2d2 import R2D2, R2D2Config
from models.orb import ORBMatcher
from models.superpoint import SuperPoint, SuperPointConfig
from models.nn_matcher import NNMatcher
from models.spsg import SuperPointSuperGlue, SuperPointSuperGlueConfig
from utils.config import hydra_config
from vo.config import VOConfig


@hydra_config(name="main", group="schema")
@dataclass
class RunConfig:
    task: str = MISSING
    task_cat: str = MISSING
    trainer: Any = MISSING
    logger: Optional[Any] = None
    model: Any = MISSING
    datamodule: Any = MISSING
    # vo: VOConfig = MISSING
    seed: int = 42
    data_split_seed: int = 41
    log_dir: Optional[str] = MISSING
    data_root: str = MISSING
    ckpt_path: Optional[str] = MISSING
    patience: int = 5
    


@hydra.main(config_path="config", config_name="train-tartanair", version_base="1.1")
def run(cfg: RunConfig):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)
    cv2.setRNGSeed(cfg.seed)

    # tb_logger = None
    # tb = program.TensorBoard()
    # tb.configure(argv=[None, '--logdir', tb_logger.root_dir, '--bind_all', '--samples_per_plugin=images=50'])
    # print(('TensorBoard at %s \n' % tb.launch()))

    model: CorrespondenceModel = hydra.utils.instantiate(cfg.model)
    dataset: DataModule = hydra.utils.instantiate(cfg.datamodule, )

    if cfg.logger is not None and cfg.log_dir is not None:
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        logger = None

    if cfg.task_cat == "train":
        monitor_name = model.evaluator.monitor_name
        save_model_cb = pl_callbacks.ModelCheckpoint(
            monitor=f"monitor/{monitor_name}", mode="max", filename=f"epoch={{epoch}}-{monitor_name}={{monitor/{monitor_name}:.3f}}",
            save_top_k=2, save_last=True, every_n_epochs=1, auto_insert_metric_name=False)
        early_stopping_cb = pl_callbacks.EarlyStopping(
            monitor=f"monitor/{monitor_name}", min_delta=0.001, mode="max", patience=cfg.patience, verbose=True, strict=False)

        config = hydra.utils.instantiate(cfg.trainer)
        trainer = pl.Trainer(**config, logger=logger, 
                             callbacks=[save_model_cb, early_stopping_cb])
        trainer.test(model, dataset)
        # trainer.validate(model, dataset)
        trainer.fit(model, dataset, ckpt_path=cfg.ckpt_path)
        trainer.test(model, dataset, ckpt_path="best")
    elif cfg.task_cat == "test":
        config = hydra.utils.instantiate(cfg.trainer)
        trainer = pl.Trainer(**config, logger=logger)
        trainer.test(model, dataset, cfg.ckpt_path)
    elif cfg.task_cat == "debug":
        config = hydra.utils.instantiate(cfg.trainer)
        trainer = pl.Trainer(**config, logger=logger,
                             callbacks=[])
        trainer.test(model, dataset)
        # trainer.validate(model, dataset)
        trainer.fit(model, dataset, ckpt_path=cfg.ckpt_path)
        trainer.test(model, dataset, ckpt_path="best")
    elif cfg.task_cat == "export":
        d = {'state_dict': model.matching_model.matcher.state_dict()}
        torch.save(d, 'out.ckpt', )
        pass
        
if __name__ == '__main__':
    run()
