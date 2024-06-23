import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import glob
import random
import logging
import numpy as np
from accelerate import Accelerator
import torch
import json
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.metrics import eval_datasets
from transformers import BitsAndBytesConfig, set_seed
import hydra
import hydra.utils as hu
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from src.utils.prompt import get_templet, select_coverage, calculate_label_ppl, calculate_all_ppl, add_demonstrations, select_samples_multilabel
from src.datasets.labels import get_mapping_token
logger = logging.getLogger(__name__)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

mark_map = {
    "scierc": "sentence",
    "scicite": "text",
    "scinli": "id"
}

class Inferencer:
    def __init__(self, cfg, accelerator) -> None:
        self.task_name = cfg.dataset_reader.task_name
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)  # 实例化: 获取训练集和测试集的text、label、ctxs、ALL等信息
        self.output_file = cfg.output_file
        self.accelerator = accelerator
        self.model_name = cfg.model_name
        self.coverage = cfg.coverage
        self.rerank_method = cfg.rerank_method
        self.sample_num = cfg.sample_num
        self.max_new_token = cfg.max_new_token

        # 设置随机种子
        print(f"rand_seed: {cfg.rand_seed}")
        # torch.cuda.manual_seed(cfg.rand_seed)
        # torch.manual_seed(cfg.rand_seed)
        # seed_everything(cfg.rand_seed)
        set_seed(cfg.rand_seed)
        if 'Llama-2' in cfg.model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, padding_side="left")
            # self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            
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

        elif 'Mixtral' in cfg.model_name or 'Qwen' in cfg.model_name:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, 
                torch_dtype=torch.float16, 
                max_memory={0: "80GiB", 1: "80GiB", 2: "80GiB", 3: "80GiB"}, 
                # max_memory={0: "80GiB", 1: "80GiB"}, 
                device_map='auto'
                ).eval()
    
        else:
            model = hu.instantiate(cfg.model).eval()
        
        model = self.accelerator.prepare(model)
        if hasattr(model, "module"):
            model = model.module
        
        return model, dataloader
    
    def mk_prompt(self, metadata, cache_dict):

        def rerank(sample_list, cache_dict):
            # 首先基于预排序方法返回sample_num个样例
            if self.coverage:
                label_set = get_mapping_token(self.task_name)
                label_list = list(label_set.values())
                if self.task_name == 'scierc':
                    sample_list = select_samples_multilabel(sample_list, label_list, self.sample_num)
                else:
                    sample_list = select_coverage(sample_list, label_list, self.sample_num)
            else:
                sample_list = sample_list[:self.sample_num]

            # 其次基于rerank_method对样例进行排序
            if self.rerank_method == "random":
                random.shuffle(sample_list)
            elif self.rerank_method == 'similarity':
                sample_list.reverse()
            elif self.rerank_method == 'length':
                sample_list.sort(key=lambda x: len(x[1]))
            elif 'ppl' in self.rerank_method:
                samples_with_ppl = []
                model_name = self.model_name.split('/')[-1]
                ppl_key = f'{model_name}_{self.rerank_method}'
                for sample in sample_list:
                    # 缓存中不存在该样例的ppl 或 缓存中存在该样例ppl但不存在该模型的ppl
                    if str(sample[0]) not in cache_dict or ppl_key not in cache_dict[str(sample[0])]:
                        # 计算该模型的ppl
                        # print("calculating ppl...")
                        prompt_templet = get_templet(self.task_name, model_name)
                        prompt = prompt_templet.format(sentence=sample[1])
                        encodings = self.tokenizer(prompt+sample[2], return_tensors="pt").to("cuda")
                        prompt_len = self.tokenizer(prompt, return_tensors="pt").to("cuda").input_ids.size(1)
                        if self.rerank_method == 'ppl_label':
                            ppl = calculate_label_ppl(encodings, prompt_len, self.model).item()
                        elif self.rerank_method == 'ppl_all':
                            ppl = calculate_all_ppl(encodings, self.model).item()
                        # 保存ppl
                        if str(sample[0]) not in cache_dict:
                            cache_dict[str(sample[0])] = {'sentence': sample[1], 'label': sample[2]}
                        cache_dict[str(sample[0])][ppl_key] = ppl
                    # 缓存里有该模型的ppl
                    else:
                        ppl = cache_dict[str(sample[0])][ppl_key]
                    samples_with_ppl.append((sample, ppl))
                sorted_samples_with_ppl = sorted(samples_with_ppl, key=lambda x: x[1])
                sample_list = [sample for sample, ppl in sorted_samples_with_ppl]

            return sample_list, cache_dict

        prompts = []
        for i, sentence in enumerate(metadata['ALL']):
            # 样例排序
            samlpes_list = [(sample['id'][i].item(), sample['ALL'][i], sample['Y_TEXT'][i]) for sample in metadata['examples']]
            samples, cache_dict = rerank(samlpes_list, cache_dict)
            # 构造Prompt
            prompt_templet = get_templet(self.task_name, self.model_name)
            prompt = add_demonstrations(self.model_name, prompt_templet, samples, sentence)

            prompts.append(prompt)

        return prompts, cache_dict
    
    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        with BufferedJsonWriter(f"{self.output_file}tmp_{self.accelerator.device}.bin") as buffer:
            solved_X = buffer.read_buffer()
            cache_dict = json.load(open(f'data/cache_dict/{self.task_name}.json', 'r'))
            for i, entry in enumerate(dataloader):  # 通过get_item融合训练集与测试集
                metadata = entry.pop("metadata")
                if metadata['X'][0] in solved_X:
                    continue
                # 构造prompt
                prompt, cache_dict = self.mk_prompt(metadata, cache_dict)
                # print(prompt[0])
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    res = self.model.generate(**inputs,
                                              pad_token_id=self.tokenizer.pad_token_id,
                                              use_cache=True,
                                              do_sample=True,
                                              max_new_tokens=self.max_new_token)
                    a = int(inputs.data['attention_mask'].shape[1])
                    for i, res_el in enumerate(res.tolist()):
                        # print(self.tokenizer.decode(res_el))
                        completion = self.tokenizer.decode(res_el[a:], skip_special_tokens=True).strip()
                        # print(f"pred: {completion}\n gt: {metadata['Y_TEXT'][i]}")
                        mdata = {'X': metadata['X'][i], 'Y_TEXT': metadata['Y_TEXT'][i], 'prompt': prompt[i], 'generated': completion}
                        buffer.write(mdata)
            json.dump(cache_dict, open(f'data/cache_dict/{self.task_name}.json', 'w'))

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

@hydra.main(config_path="configs", config_name="two_stage")
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
