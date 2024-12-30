from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, SingleDeviceStrategy
from dataclasses import dataclass, field
from data.dataloader import ChunkedDataLoader, ChunkedDataset
from models.modeling_qwen import Qwen2ForCausalLM, Qwen2DecoderLayer, QwenCrossEntropyLoss
from typing import Optional, Literal
import math
from utils.utils import generate_att_mask_pos_ids


@dataclass
class QwenTrainConfig:
    num_devices: int = 1
    precision: Optional[str] = None # "bf16-mixed"
    lr_max: float = 3e-4
    lr_min: float = 0.0
    weight_decay: float = 0.05
    beta_1: float = 0.9
    beta_2: float = 0.95
    grad_clip: float = 1.0
    iter_start: int = 0
    iter_max: int = (430000*2+220000)*8
    iter_warmup: int = 1000*8
    iter_lr_decay: int = (430000*2+220000)*8
    checkpoint_path: str = ""
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    batch_size_par_device: int = field(init=False) 
    global_batch_size: int = field(init=False) 

    def __post_init__(self):
        self.batch_size_par_device = self.gradient_accumulation_steps*self.micro_batch_size
        self.global_batch_size = self.batch_size_par_device*self.num_devices

# @dataclass
# class QwenTrainState:
#     model: Qwen2ForCausalLM 
#     optim: torch.optim.Optimizer 
#     num_iter: int = 0
        
#     def step(self):
#         self.num_iter += 1

#     def save(self, path: str):
#         pass

#     def load(self, path: str):
#         pass

class QwenTrainer:
    def __init__(self, conf: QwenTrainConfig, model: Qwen2ForCausalLM, dataset: ChunkedDataset, optim: Optional[torch.optim.Optimizer] = None):

        # is_resume is encoded in model
        self.num_devices = conf.num_devices
        self.precision = conf.precision
        self.lr_max = conf.lr_max
        self.lr_min = conf.lr_min
        self.weight_decay = conf.weight_decay
        self.beta_1 = conf.beta_1
        self.beta_2 = conf.beta_2
        self.grad_clip = conf.grad_clip
        self.iter_start = conf.iter_start
        self.iter_max = conf.iter_max
        self.iter_warmup = conf.iter_warmup
        self.iter_lr_decay = conf.iter_lr_decay
        self.checkpoint_path = conf.checkpoint_path
        self.micro_batch_size = conf.micro_batch_size
        self.gradient_accumulation_steps = conf.gradient_accumulation_steps

        self.iter_current = self.iter_start

        self.model = model

        if optim == None:
            self.optim = torch.optim.AdamW(
                model.parameters(), lr=self.lr_max, weight_decay=self.weight_decay, betas=(self.beta_1, self.beta_2), foreach=False
            ) 
        else:
            self.optim = optim

        if self.num_devices > 1:
            self.strategy = FSDPStrategy(#sharding_strategy='SHARD_GRAD_OP',
                                    auto_wrap_policy={Qwen2DecoderLayer},
                                    state_dict_type="full"
                                    )
        else:
            self.strategy = "auto" # SingleDeviceStrategy(device='cpu', precision=precision)

        self.fabric = L.Fabric(devices=self.num_devices, 
                          precision=self.precision,
                          strategy=self.strategy)

        # wandb and logger

        self.dataset = dataset

        # save path


    def train(self):
        fabric = self.fabric
        fabric.launch()

        # Check save path, if not exists, create one
        
        self.dataset.setup_distributed(fabric.global_rank, self.num_devices)

        fabric.seed_everything(3407)

        model: Qwen2ForCausalLM = fabric.setup_module(self.model)
    
        optim: torch.optim.Optimizer = fabric.setup_optimizers(self.optim)

        dataloader = ChunkedDataLoader(self.dataset, self.micro_batch_size, batch_start=self.iter_start)
        dataloader: ChunkedDataLoader = fabric.setup_dataloaders(dataloader)

        # state = {"model": model,"optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
        
        for data in dataloader:
            data: torch.Tensor
            if self.iter_current > self.iter_max:
                break
            lr = self.get_lr()

            for param_groups in optim.param_groups:
                param_groups["lr"] = lr
            
            input_ids = data[:, 0:data.shape[1] - 1]
            target_ids = data[:, 1:data.shape[1]]

            attention_mask, position_ids = generate_att_mask_pos_ids(input_ids, self.dataset.tokenizer.pad_token_id)

            is_accumulating = (self.iter_current + 1)%self.gradient_accumulation_steps != 0

            cross_entropy = QwenCrossEntropyLoss(ignore_index=self.dataset.tokenizer.pad_token_id)

            # todo: autocast
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                output_logits = model.forward(input_ids, attention_mask, position_ids)
                with fabric.autocast():
                    loss = cross_entropy(output_logits, target_ids)
                print(f"batch: {self.iter_current}, loss: {loss}")
                fabric.backward(loss/self.gradient_accumulation_steps)

            if not is_accumulating:
                fabric.clip_gradients(model, optim, self.grad_clip)
                optim.step()
                optim.zero_grad()
                # state step count

            self.iter_current += 1

        # checkpointing

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self):
        # 1) learning rate for warmup stage, learning rate linearly grow to max learning rate
        if self.iter_current < self.iter_warmup:
            return self.lr_max * (self.iter_current/self.iter_warmup)
        # 2) learning rate for final stage
        if self.iter_current > self.iter_lr_decay:
            return self.lr_min
        # 3) learning rate for cosine decay stage, learning rate gradually decay to min learning rate
        decay_ratio = (self.iter_current - self.iter_warmup)/(self.iter_lr_decay - self.iter_warmup)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.lr_min + coeff*(self.lr_max - self.lr_min)

def _test_trainer():
    from data.dataloader import ChunkedDatasetConfig
    import os

    # Prepare dataloader
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_downloaded/_test/step2_processed/sky")
    conf_dataset = ChunkedDatasetConfig(data_dir=data_dir,
                                        is_shuffle=False,
                                        l_ctx=2048,
                                        tokenizer=tokenizer,
                                        num_worker=2,
                                        buffer_size=100)
    dataset = ChunkedDataset(conf=conf_dataset)

    # Prepare model
    from transformers import AutoModelForCausalLM
    from transformers import Qwen2ForCausalLM as Qwen2ForCausalLMOld
    from models.modeling_qwen import Qwen2Config

    model_old: Qwen2ForCausalLMOld = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", attn_implementation = "sdpa")
    conf = Qwen2Config(hidden_size=1536,
                       pad_token_id=None,
                       vocab_size=151936,
                       num_hidden_layers=28,
                       rms_norm_eps=1e-06,
                       intermediate_size=8960,
                       hidden_act="silu",
                       attn_implementation='sdpa',
                       num_attention_heads=12,
                       num_key_value_heads=2,
                       max_position_embeddings=131072,
                       rope_theta=1000000.0,
                       attention_dropout=0.0
                       )
    model = Qwen2ForCausalLM(conf)
    model.sync_from_pretrained_model(model_old)

    # Prepare config
    conf = QwenTrainConfig()

    # Prepare trainer
    trainer = QwenTrainer(conf, model, dataset)

    # train
    trainer.train()


if __name__ == '__main__':
    from transformers import AutoTokenizer
    # Prepare tokenizer
    tokenizer_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, pad_token='<|endoftext|>')
    _test_trainer()