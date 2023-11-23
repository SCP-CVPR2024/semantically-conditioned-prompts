import os
import copy
from torchmetrics.functional import f1_score
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS, ViLTransformerSS_CLSKMeans, ViLTransformerSS_embedding, ViLTransformerSS_topkKV, ViLTransformerSS_PFC, ViLTransformerSS_collectCLS, ViLTransformerSS_QKV_ps, ViLTransformerSS_embedding_agnostic
from vilt.datamodules.multitask_datamodule import MTDataModule

import wandb
from pytorch_lightning.loggers import WandbLogger
import torch

# torch.autograd.set_detect_anomaly(True)
models = {
    "ViLTransformerSS":ViLTransformerSS,
    "ViLTransformerSS_CLSKMeans":ViLTransformerSS_CLSKMeans,
    "ViLTransformerSS_embedding":ViLTransformerSS_embedding,
    "ViLTransformerSS_embedding_agnostic":ViLTransformerSS_embedding_agnostic,
    "ViLTransformerSS_topkKV":ViLTransformerSS_topkKV,
    "ViLTransformerSS_PFC":ViLTransformerSS_PFC,
    "ViLTransformerSS_collectCLS":ViLTransformerSS_collectCLS,
    "ViLTransformerSS_QKV_ps":ViLTransformerSS_QKV_ps
}

@ex.automain
def main(_config):
    print("HOSTNAME: ", os.environ.get("HOSTNAME"))
    
    
    print("Current working directory: {0}".format(os.getcwd()))
    print("Lesgo lesgo\n")
    _config = copy.deepcopy(_config)
    print(_config)
    for k, v in _config.items():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(k, f" :{pad}", v)
    pl.seed_everything(_config["seed"])
    wandb.init(
        project="map",
        name=f"{_config['exp_name']}_{_config['model_name']}",
        mode="disabled"
    )

    dm = MTDataModule(_config, dist=True)

    print(_config["model_name"])
    model = models[_config["model_name"]](_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_{_config["model_name"]}',
    )
    # logger = WandbLogger(
    #     name=f'{exp_name}',
    #     log_model=True,
    #     dir=_config["log_dir"],
    # )    

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    # from pytorch_lightning.profiler import SimpleProfiler
    # profiler = SimpleProfiler()
    
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(_config["batch_size"], _config["per_gpu_batchsize"], num_gpus, _config["num_nodes"])
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        # auto_select_gpus=True, 
        # log_gpu_memory="min_max", 
        # profiler=profiler,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        print("\n\nEND TRAINING\n\nSTART TEST\n\n")
        best_checkpoint_path = checkpoint_callback.best_model_path
        print("\nbest_checkpoint_path: ", best_checkpoint_path)
        ckpt = torch.load(best_checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        trainer.test(model, datamodule=dm)
    else:
        ckpt = torch.load(_config["load_path"])
        state_dict = ckpt["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        trainer.test(model, datamodule=dm)
    wandb.finish()