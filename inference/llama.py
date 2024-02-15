# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import fire
import torch
import json
from tqdm import tqdm
import time

from transformers import LlamaTokenizer
from util import load_model, read_dataset, get_sequence

PATH_DICT = {
    "llama2_13b": "/data_share/model_hub/llama/Llama-2-13b-hf",
    "llama2_70b": "/data_share/model_hub/Llama-2-70b-hf"
}

def mk_prompt(data_name, samples, doc):


    if data_name == "scierc":
        sample = []
        for example in samples:
            ner = []
            for entity in example[1]:
                ner.append([get_sequence(example[0][entity[0]: entity[1]+1]), entity[2]])
            sample.append([get_sequence(example[0]), ner])
        prompt = f"Extract scientific entities from sentences. The scientific entity category includes ['Method', 'Task', 'Metric', 'Material', 'Generic', 'OtherScientificTerm', 'Generic']. \nSentence: {sample[0][0]} [/INST]{sample[0][1]}</s><s>[INST] Sentence: {sample[1][0]} [/INST] {sample[1][1]}<s>[INST] Sentence: {sample[2][0]} [/INST] {sample[2][1]}</s><s>[INST] Sentence: {sample[3][0]} [/INST] {sample[3][1]}</s><s>[INST] Sentence: {sample[4][0]} [/INST] {sample[4][1]}</s><s>[INST] Sentence: {doc['sentence']} [/INST]"

    elif data_name == "scicite":
        prompt = f"Identify the intent of a citation in scientific papers. Choose the citation intention of the following sentence from ['method', 'background', 'result']. \nSentence: {samples[0][0]}\n{samples[0][1]}\n\nSentence: {samples[1][0]}\n{samples[1][1]}\n\nSentence: {samples[2][0]}\n{samples[2][1]}\n\nSentence: {samples[3][0]}\n{samples[3][1]}\n\nSentence: {samples[4][0]}\n{samples[4][1]}\n\nSentence: {doc['text']}\n"

    elif data_name == "scinli":
        prompt = f"Identify the semantic relationship between the following pair of sentences. The semantic relationship includes ['reasoning', 'entailment', 'contrasting', 'neutral']. Here are some examples: \n\nSentence1: {samples[0][0]} \nSentence2:{samples[0][1]}\nRelation:{samples[0][2]}\nSentence1: {samples[1][0]} \nSentence2: {samples[1][1]}\nRelation:{samples[1][2]}\nSentence1: {samples[2][0]} \nSentence2: {samples[2][1]}\nRelation:{samples[2][2]}\nSentence1: {samples[3][0]} \nSentence2: {samples[3][1]}\nRelation:{samples[3][2]}\n\nSentence1: {doc['sentence1']} \nSentence2: {doc['sentence2']}\nRelation:"

    return prompt

def main(
    model_name: str="llama2_70b",
    curriculum: str="human",
    data_name: str="scinli",
    quantization: bool=True,
    max_new_tokens =2, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=False, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=False,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    # temperature: float=0.5, # [optional] The value used to modulate the next token probabilities.
    # top_k: int=20, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):
    print(f"model_name: {model_name}, curriculum: {curriculum}, data_name: {data_name})")
    data_path = f"data/{data_name}/test.json"
    sample_path = f"data/sample/{data_name}.json"
    output_path = f"output/{data_name}/{model_name}_{curriculum}.json"

    # 读取数据集
    dataset = read_dataset(data_path)
    # 读取样例数据
    sample = json.load(open(sample_path, "r", encoding="utf-8"))

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model_path = PATH_DICT[model_name]
    model = load_model(model_path, quantization)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )
    model.eval()
    
    # 打开输出文件
    f_output = open(output_path, "w", encoding="utf-8")
    for doc in tqdm(dataset):
        # 生成prompt并tokenize
        prompt = mk_prompt(data_name, sample[curriculum], doc)
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        # start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                # temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                # top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )
        # e2e_inference_time = (time.perf_counter()-start)*1000
        # print(f"the inference time is {e2e_inference_time} ms")
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = completion.split('\nRelation:')[-1]
        doc['pred'] = answer.strip()
        # del doc["instruction"]
        f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f_output.flush()  # flush the buffer to disk
    # 关闭输出文件
    f_output.close()
    
 
if __name__ == "__main__":
    fire.Fire(main)
