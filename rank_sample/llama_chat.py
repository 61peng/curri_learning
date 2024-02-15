# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import fire
import torch
import json
from tqdm import tqdm
import sys
sys.path.append('/home/llmtrainer/LLM/lyp/curri_learn')
from transformers import LlamaTokenizer
from util import load_model, get_sequence

PATH_DICT = {
    "llama2_7b": "/data_share/model_hub/llama/Llama-2-7b-chat-hf",
    "llama2_13b": "/data_share/model_hub/llama/Llama-2-13b-chat-hf",
    "llama2_70b": "/data_share/model_hub/llama/Llama-2-70b-chat-hf"
}


def mk_prompt(data_name, samples):
    
    sys_message = "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"


    if data_name == "scierc":
        sample = []
        for example in samples:
            ner = []
            for entity in example[1]:
                ner.append([get_sequence(example[0][entity[0]: entity[1]+1]), entity[2]])
            sample.append([get_sequence(example[0]), ner])
        prompt = f"<s>[INST] {sys_message}\n\nHere are 5 demonstrations used to prompt the model to solve NER tasks:  \nSentence: {sample[0][0]}\tOutput: {sample[0][1]}\nSentence: {sample[1][0]}\tOutput: {sample[1][1]}\nSentence: {sample[2][0]}\tOutput: {sample[2][1]}\nSentence: {sample[3][0]}\tOutput: {sample[3][1]}\nSentence: {sample[4][0]}\tOutput: {sample[4][1]}\n\nPlease sort these demonstrations in order from simple to difficult.[/INST]"

    elif data_name == "scicite":
        prompt = f"<s>[INST] {sys_message}\n\nIdentify the intent of a citation in scientific papers. Choose the citation intention of the following sentence from ['method', 'background', 'result']. Sentence: {samples[0][0]} [/INST]{samples[0][1]}</s><s>[INST]Sentence: {samples[1][0]} [/INST]{samples[1][1]}</s><s>[INST]Sentence: {samples[2][0]} [/INST]{samples[2][1]}</s><s>[INST]Sentence: {samples[3][0]} [/INST]{samples[3][1]}</s><s>[INST]Sentence: {samples[4][0]} [/INST]{samples[4][1]}</s><s>[/INST]"

    elif data_name == "scinli":
        prompt = f"<s>[INST] {sys_message}\n\nHere are 4 demonstrations used to prompt the model to solve scientific NLI tasks: \nSentence1: {samples[0][0]} \tSentence2: {samples[0][1]}\tOutput: {samples[0][2]}\nSentence1: {samples[1][0]} \tSentence2: {samples[1][1]}\tOutput: {samples[1][2]}\nSentence1: {samples[2][0]} \tSentence2: {samples[2][1]}\tOutput: {samples[2][2]}\nSentence1: {samples[3][0]} \tSentence2: {samples[3][1]}\tOutput: {samples[3][2]}\n\nPlease sort these demonstrations in order from simple to difficult based on your understanding.[/INST]"

    return prompt

def main(
    model_name: str="llama2_70b",
    curriculum: str="random",
    data_name: str="scinli",
    quantization: bool=True,
    max_new_tokens =1024, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.95, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.6, # [optional] The value used to modulate the next token probabilities.
    top_k: int=20, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs
):
    print(f"model_name: {model_name}, curriculum: {curriculum}, data_name: {data_name})")
    sample_path = f"data/sample/{data_name}.json"

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
    

    prompt = mk_prompt(data_name, sample[curriculum])
    batch = tokenizer(prompt, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    # start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                # top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
        )

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = completion.split('[/INST]')[-1]
    print(answer)


if __name__ == "__main__":
    fire.Fire(main)
