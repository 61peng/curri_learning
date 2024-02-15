import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import sys
sys.path.append('/home/llmtrainer/LLM/lyp/curri_learn')
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from util import read_dataset, get_sequence

mark_map = {
    "scierc": "sentence",
    "scicite": "text",
    "scinli": "id"
}

def mk_prompt(samples):

    if data_name == "scierc":
        sample = []
        for example in samples:
            ner = []
            for entity in example[1]:
                ner.append([get_sequence(example[0][entity[0]: entity[1]+1]), entity[2]])
            sample.append([get_sequence(example[0]), ner])

        prompt = f"<s> [INST]Here are 5 demonstrations used to prompt the model to solve NER tasks: \nSentence: {sample[0][0]}\tOutput: {sample[0][1]}\nSentence: {sample[1][0]}\tOutput: {sample[1][1]}\nSentence: {sample[2][0]}\tOutput: {sample[2][1]}\nSentence: {sample[3][0]}\tOutput: {sample[3][1]}\nSentence: {sample[4][0]}\tOutput: {sample[4][1]}\n\nPlease sort these demonstrations in order from simple to difficult.[/INST]"

    elif data_name == "scicite":
        prompt = f"<s> [INST]Here are 5 demonstrations used to prompt the model to solve text classification tasks: \nSentence: {samples[0][0]} [/INST]{samples[0][1]}</s>[INST] Sentence: {samples[1][0]} [/INST] {samples[1][1]}</s>[INST] Sentence: {samples[2][0]} [/INST] {samples[2][1]}</s>[INST] Sentence: {samples[3][0]} [/INST] {samples[3][1]}</s>[INST] Sentence: {samples[4][0]} [/INST] {samples[4][1]}[/INST]"
    
    elif data_name == "scinli":
        prompt = f"<s> [INST]Here are 4 demonstrations used to prompt the model to solve scientific natural language inference tasks: \nSentence1: {samples[0][0]} \tSentence2: {samples[0][1]}\tOutput: {samples[0][2]}\nSentence1: {samples[1][0]} \tSentence2: {samples[1][1]}\tOutput: {samples[1][2]}\nSentence1: {samples[2][0]} \tSentence2: {samples[2][1]}\tOutput: {samples[2][2]}\nSentence1: {samples[3][0]} \tSentence2: {samples[3][1]}\tOutput: {samples[3][2]}\n\nPlease sort these demonstrations in order from simple to difficult.[/INST]"


    return prompt



if __name__ == "__main__":
    seed = 42
    curriculum = "random"
    data_name = "scinli"
    model_path = "/data_share/model_hub/Mixtral/Mixtral-8x7B-Instruct-v0.1"
    sample_path = f"data/sample/{data_name}.json"

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # 读取样例数据
    samples = json.load(open(sample_path, "r", encoding="utf-8"))

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto').eval()
    
    # 生成prompt并tokenize
    prompt = mk_prompt(samples[curriculum])
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # 生成答案
    with torch.no_grad():
        outputs = model.generate(**inputs, 
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=1024
                                )
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = completion.split('[/INST]')[-1].strip()
    print(answer)
