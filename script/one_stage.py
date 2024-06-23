import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import glob
import random
import logging
from accelerate import Accelerator
import torch
import json
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, set_seed
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.metrics import eval_datasets
import hydra
import hydra.utils as hu
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from src.utils.prompt import get_templet, add_demonstrations
from util import get_sequence
logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, cfg, accelerator) -> None:
        self.task_name = cfg.dataset_reader.task_name
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)  # 实例化
        self.output_file = cfg.output_file
        self.accelerator = accelerator
        self.model_name = cfg.model_name
        self.rerank_method = cfg.rerank_method
        self.sample_num = cfg.sample_num

        # 设置随机种子
        # torch.cuda.manual_seed(cfg.rand_seed)
        # torch.manual_seed(cfg.rand_seed)
        set_seed(cfg.rand_seed)
        if 'Llama' in cfg.model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            # self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model, self.dataloader = self.init_model_dataloader(cfg)

    def init_model_dataloader(self, cfg):
        self.dataset_reader.shard(self.accelerator)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size)
        if 'Llama-2' in cfg.model_name:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = LlamaForCausalLM.from_pretrained(
                cfg.model_name, 
                torch_dtype=torch.float16,
                max_memory={0: "80GiB", 1: "80GiB", 2: "80GiB", 3: "80GiB"}, 
                quantization_config=bnb_config,
                device_map='auto'
                ).eval()

        elif 'Mixtral' in cfg.model_name or "Qwen" in cfg.model_name:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, 
                torch_dtype=torch.float16, 
                max_memory={0: "80GiB", 1: "80GiB", 2: "80GiB", 3: "80GiB"}, 
                device_map='auto'
                ).eval()

        else:
            model = hu.instantiate(cfg.model).eval()
        
        model = self.accelerator.prepare(model)
        if hasattr(model, "module"):
            model = model.module
        
        return model, dataloader
    
    def mk_prompt(self, metadata):

        prompts = []
        for i, sentence in enumerate(metadata['ALL']):
            # 样例排序
            # samples_list = [(sample['X'][i], sample['Y_TEXT'][i]) for sample in metadata['examples']][:self.sample_num]
            samples = json.load(open(f'data/sample/{self.task_name}.json', 'r'))[self.rerank_method]
            if self.task_name == 'scinli':
                samples = [(id, f"Sentence1: {sample[0]}\tSentence2: {sample[1]}", sample[2]) for id, sample in enumerate(samples)]
            elif self.task_name == 'scierc':
                sample = []
                for id, example in enumerate(samples):
                    ner = []
                    for entity in example[1]:
                        ner.append([get_sequence(example[0][entity[0]: entity[1]+1]), entity[2]])
                    sample.append([id, get_sequence(example[0]), ner])
                samples = sample
            # 构造Prompt
            prompt_templet = get_templet(self.task_name, self.model_name)
            prompt = add_demonstrations(self.model_name, prompt_templet, samples, sentence)

            prompts.append(prompt)

        return prompts
    
    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:
            solved_X = buffer.read_buffer()
            for i, entry in enumerate(dataloader):
                metadata = entry.pop("metadata")
                if metadata['X'][0] in solved_X:
                    continue
                prompt = self.mk_prompt(metadata)
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    res = self.model.generate(**inputs,
                                              pad_token_id=self.tokenizer.pad_token_id,
                                              use_cache=True,
                                              do_sample=True,
                                              max_new_tokens=256)
                    a = int(inputs.data['attention_mask'].shape[1])  # maxlength???
                    for i, res_el in enumerate(res.tolist()):
                        completion = self.tokenizer.decode(res_el[a:], skip_special_tokens=True).strip()
                        mdata = {'X': metadata['X'][i],  'Y_TEXT': metadata['Y_TEXT'][i], 'prompt': prompt[i], 'generated': completion}
                        buffer.write(mdata)

    def write_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with BufferedJsonReader(path) as f:
                data.extend(f.read())
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)

        with open(self.output_file, "w") as f:
            json.dump(data, f)

        # data, metric = eval_datasets.app[self.task_name](self.output_file)
        # logger.info(f"metric: {str(metric)}")
        # with open(self.output_file + '_metric', "w") as f:
        #     logger.info(f'{self.output_file}:{metric}')
        #     json.dump({'metric': metric}, f)
        # with open(self.output_file, "w") as f:
        #     json.dump(data, f)

        return data

@hydra.main(config_path="configs", config_name="one_stage")
def main(cfg):
    logger.info(cfg)
    accelerator = Accelerator()
    inferencer = Inferencer(cfg, accelerator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inferencer.forward()
        inferencer.write_results()


if __name__ == "__main__":
    main()
