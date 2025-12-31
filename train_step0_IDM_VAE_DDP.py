import yaml
from pytorch_lightning import Trainer
from model.IDM.utils.util import instantiate_from_config
import os
import os
import logging

# Tắt log của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Tắt các cảnh báo từ hệ thống
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")



class AttributeDict(dict):
    def __init__(self, data=None):
        if data is None:
            data = {}
        super().__init__(data)
        
        # Đệ quy: Chuyển các dict con thành AttributeDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttributeDict(value)
            elif isinstance(value, list):
                self[key] = [AttributeDict(i) if isinstance(i, dict) else i for i in value]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"AttributeDict has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"AttributeDict has no attribute '{key}'")
    def to_dict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, AttributeDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [i.to_dict() if isinstance(i, AttributeDict) else i for i in value]
            else:
                result[key] = value
        return result


def train():
    with open("./train/config/step0_train_IDM_VAE.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    config = AttributeDict(yaml_config)
    config.model.params.ckpt_path = None
    config.data.params.train.params.FudanVI_lmdb_folder = "/kaggle/input/fudan-lmdb/content/benchmark_dataset/scene/scene_val"
    config.data.params.validation.params.FudanVI_lmdb_folder = "/kaggle/input/fudan-lmdb/content/benchmark_dataset/scene/scene_val"
    config.data.params.batch_size = 8
    batch_size, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    model = instantiate_from_config(config.model)
    model.automatic_optimization = False
    model.learning_rate = float(base_lr)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_loader = data.train_dataloader()
    trainer = Trainer(max_epochs=3, accelerator="auto", devices=2, strategy="ddp_find_unused_parameters_true")
    trainer.fit(model, train_loader)
    ckpt_path = "/kaggle/working/step0_train_IDM_VAE.ckpt"
    trainer.save_checkpoint(ckpt_path)

if __name__ == "__main__":
    train()