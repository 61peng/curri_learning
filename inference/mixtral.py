import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from util import read_dataset


def mk_prompt(doc, sample):
    if data_name == "scicite":
        prompt = f"You are a scientific literature analyst. Identifying the intent of a citation in scientific papers. The citation intention includes ['method', 'background', 'result'].\nSentence: {sample[0][0]}\tOutput: {sample[0][1]}\nSentence: {sample[1][0]}\tOutput: {sample[1][1]}\nSentence: {sample[2][0]}\tOutput: {sample[2][1]}\nSentence: {sample[3][0]}\tOutput: {sample[3][1]}\nSentence: {sample[4][0]}\tOutput: {sample[4][1]}\nSentence: {doc['text']}\tOutput: "
    elif data_name == "scinli":
        prompt = f"Identify the semantic relationship between the following pair of sentences. The semantic relationship includes ['reasoning', 'entailment', 'contrasting', 'neutral']. Here are some examples: \n\nSentence1: {sample[0][0]} \nSentence2:{sample[0][1]}\nRelation:{sample[0][2]}\nSentence1: {sample[1][0]} \nSentence2: {sample[1][1]}\nRelation:{sample[1][2]}\nSentence1: {sample[2][0]} \nSentence2: {sample[2][1]}\nRelation:{sample[2][2]}\nSentence1: {sample[3][0]} \nSentence2: {sample[3][1]}\nRelation:{sample[3][2]}\n\nSentence1: {doc['sentence1']} \nSentence2: {doc['sentence2']}\nRelation:"
    return prompt


if __name__ == "__main__":

    curriculum = "random"
    data_name = "scicite"
    model_path = "/data_share/model_hub/Mixtral/Mixtral-8x7B-v0.1"
    data_path = f"data/{data_name}/test.json"
    sample_path = f"data/sample/{data_name}.json"
    output_path = f"output/{data_name}/mixtral_{curriculum}_v2.json"
    
    # 读取数据集
    dataset = read_dataset(data_path)
    # 读取样例数据
    sample = json.load(open(sample_path, "r", encoding="utf-8"))

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
    
    # 打开输出文件
    f_output = open(output_path, "w", encoding="utf-8")
    for doc in tqdm(dataset):
        # 生成prompt并tokenize
        prompt = mk_prompt(doc, sample[curriculum])
        doc['context'] = prompt
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # 生成答案
        outputs = model.generate(**inputs, 
                                pad_token_id=tokenizer.eos_token_id,
                                max_new_tokens=1
                                )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = completion.split('\tOutput:')[-1]
        doc['pred'] = answer.strip()
        # del doc["instruction"]
        f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f_output.flush()  # flush the buffer to disk
    # 关闭输出文件
    f_output.close()
