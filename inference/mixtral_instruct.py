import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

def mk_prompt(doc, samples):

    if data_name == "scierc":
        sample = []
        for example in samples:
            ner = []
            for entity in example[1]:
                ner.append([get_sequence(example[0][entity[0]: entity[1]+1]), entity[2]])
            sample.append([get_sequence(example[0]), ner])

        prompt = f"<s> [INST]You are a scientific literature analyst. Extract scientific entities from sentences. The scientific entity category includes ['Method', 'Task', 'Metric', 'Material', 'Generic', 'OtherScientificTerm', 'Generic']. \nSentence: {sample[0][0]} [/INST]{sample[0][1]}</s>[INST] Sentence: {sample[1][0]} [/INST] {sample[1][1]}</s>[INST] Sentence: {sample[2][0]} [/INST] {sample[2][1]}</s>[INST] Sentence: {sample[3][0]} [/INST] {sample[3][1]}</s>[INST] Sentence: {sample[4][0]} [/INST] {sample[4][1]}</s>[INST] Sentence: {doc['sentence']} [/INST]"

    elif data_name == "scicite":
        prompt = f"<s> [INST]You are a scientific literature analyst. Identify the intent of a citation in scientific papers. The citation intention includes ['method', 'background', 'result']. \nSentence: {samples[0][0]} [/INST]{samples[0][1]}</s>[INST] Sentence: {samples[1][0]} [/INST] {samples[1][1]}</s>[INST] Sentence: {samples[2][0]} [/INST] {samples[2][1]}</s>[INST] Sentence: {sample[3][0]} [/INST] {samples[3][1]}</s>[INST] Sentence: {samples[4][0]} [/INST] {samples[4][1]}</s>[INST] Sentence: {doc['text']} [/INST]"
    
    elif data_name == "scinli":
        prompt = f"<s> [INST]You are a scientific literature analyst. Identify the semantic relationship between the following pair of sentences. The semantic relationship includes ['reasoning', 'entailment', 'contrasting', 'neutral']. \nSentence1: {samples[0][0]} Sentence2: {samples[0][1]}[/INST]{samples[0][2]}</s>[INST] Sentence1: {samples[1][0]} Sentence2: {samples[1][1]}[/INST] {samples[1][2]}</s>[INST] Sentence1: {samples[2][0]} Sentence2: {samples[2][1]}[/INST] {samples[2][2]}</s>[INST] Sentence1: {samples[3][0]} Sentence2: {samples[3][1]}[/INST] {samples[3][2]}</s>[INST] Sentence1: {doc['sentence1']} Sentence2: {doc['sentence2']}[/INST]"


    return prompt



if __name__ == "__main__":

    curriculum = "mixtral"
    data_name = "scinli"
    model_path = "/data_share/model_hub/Mixtral/Mixtral-8x7B-Instruct-v0.1"
    data_path = f"data/{data_name}/test.json"
    sample_path = f"data/sample/{data_name}.json"
    output_path = f"output/{data_name}/mixtral_instruct_{curriculum}.json"
    
    # 读取数据集
    dataset = read_dataset(data_path)
    # 读取样例数据
    samples = json.load(open(sample_path, "r", encoding="utf-8"))
    # 读取已处理数据
    solved_sentence = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            solved_sentence = []
            for line in f:
                doc = json.loads(line)
                solved_sentence.append(doc[mark_map[data_name]])

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto').eval()
    
    # 打开输出文件
    f_output = open(output_path, "a+", encoding="utf-8")
    for doc in tqdm(dataset):
        if doc[mark_map[data_name]] in solved_sentence:
            continue
        # 生成prompt并tokenize
        prompt = mk_prompt(doc, samples[curriculum])
        doc['context'] = prompt
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(**inputs, 
                                    pad_token_id=tokenizer.eos_token_id,
                                    max_new_tokens=256
                                    )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = completion.split('[/INST]')[-1].strip()
        doc['pred'] = answer
        # del doc["instruction"]
        f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f_output.flush()  # flush the buffer to disk
    # 关闭输出文件
    f_output.close()
