import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import re
import tqdm
from util import read_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig





def generate_sample(model, tokenizer, text, sample):
    
    prompt = f"You are a scientific literature analyst. Identifying the intent of a citation in scientific papers. The citation intention includes ['method', 'background', 'result'].\nSentence: {sample[0][0]}\tOutput: {sample[0][1]}\nSentence: {sample[1][0]}\tOutput: {sample[1][1]}\nSentence: {sample[2][0]}\tOutput: {sample[2][1]}\nSentence: {sample[3][0]}\tOutput: {sample[3][1]}\nSentence: {sample[4][0]}\tOutput: {sample[4][1]}\nSentence: {text}\tOutput:"
    
    
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    
    answer = output.split('\tOutput:')[-1].strip()

    # print(answer)
    return answer

if __name__ == "__main__":
    
    curriculum = "human"
    data_name = "scicite"
    checkpoint_path = '/data_share/model_hub/qwen/Qwen-14B'

    sample_file = f"data/sample/{data_name}.json"
    input_file = f"data/{data_name}/test.json"
    output_file = f"output/{data_name}/qwen_14b_{curriculum}.json"

    dataset = read_dataset(input_file)

    with open(sample_file, "r", encoding="utf-8") as f:
        sample = json.load(f)

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True, bf16=True, use_flash_attn=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, device_map="auto", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    model.generation_config.max_new_tokens = 1

    f_output = open(output_file, "w", encoding="utf-8")

    acc_res = []
    for doc in tqdm.tqdm(dataset):
        completion = generate_sample(model, tokenizer, doc['text'], sample[curriculum])
        doc["pred"] = completion
        f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f_output.flush()  # flush the buffer to disk

    f_output.close()
