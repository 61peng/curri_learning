import glob
import json
import os
import warnings
import logging

import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Tokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer, set_seed

from src.metrics import eval_datasets
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader

logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, cfg, accelerator) -> None:
        self.task_name = cfg.dataset_reader.task_name
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.output_file = cfg.output_file
        self.accelerator = accelerator
        self.model_name = cfg.model_name

        set_seed(cfg.rand_seed)
        if 'Llama-2' in cfg.model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            # self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model, self.dataloader = self.init_model_dataloader(cfg)
        # self.model.resize_token_embeddings(len(self.tokenizer.vocab))

    def init_model_dataloader(self, cfg):
        self.dataset_reader.shard(self.accelerator)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size)
        if 'Llama-2' in cfg.model_name:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = LlamaForCausalLM.from_pretrained(
                cfg.model_name, 
                torch_dtype=torch.float16,
                max_memory={0: "80GiB"}, 
                quantization_config=bnb_config,
                device_map='auto'
                ).eval()

        elif 'Mixtral' in cfg.model_name:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, 
                torch_dtype=torch.float16, 
                max_memory={0: "80GiB", 1: "80GiB"}, 
                device_map='auto'
                ).eval()

        else:
            model = hu.instantiate(cfg.model).eval()
        
        model = self.accelerator.prepare(model)
        if hasattr(model, "module"):
            model = model.module
        
        return model, dataloader

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        avg_ice_num = 0
        total_num = 0
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:
            for i, entry in enumerate(dataloader):
                metadata = entry.pop("metadata")
                with torch.no_grad():
                    res = self.model.generate(input_ids=entry.input_ids,
                                              attention_mask=entry.attention_mask,
                                              eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                              pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                              max_new_tokens=100,
                                              do_sample=False)
                    a = int(entry.attention_mask.shape[1])  # maxlength???
                    for mdata, res_el in zip(metadata, res.tolist()):
                        mdata['generated'] = self.dataset_reader.tokenizer.decode(res_el[a:],
                                                                                  skip_special_tokens=True)
                        buffer.write(mdata)
                        avg_ice_num += len(mdata['prompt_list'])
                        total_num += 1

        logging.info(f"Average number of in-context examples after truncating is {avg_ice_num / total_num}")

    def write_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                data.extend(f.read())
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)

        with open(self.output_file, "w") as f:
            json.dump(data, f)

        data, metric = eval_datasets.app[self.task_name](self.output_file)
        logger.info(f"metric: {str(metric)}")
        with open(self.output_file + '_metric', "w") as f:
            logger.info(f'{self.output_file}:{metric}')
            json.dump({'metric': metric}, f)
        with open(self.output_file, "w") as f:
            json.dump(data, f)

        return data


@hydra.main(config_path="configs", config_name="inferencer")
def main(cfg):
    logger.info(cfg)
    accelerator = Accelerator()
    inferencer = Inferencer(cfg, accelerator)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferencer.forward()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_results()


if __name__ == "__main__":
    main()
