import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
import re
import tqdm
from util import read_dataset, get_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

mark_map = {
    "scierc": "sentence",
    "scicite": "text",
    "scinli": "id"
}

def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, doc, samples):
    if data_name == "scierc":
        sample = []
        for example in samples:
            ner = []
            for entity in example[1]:
                ner.append([get_sequence(example[0][entity[0]: entity[1]+1]), entity[2]])
            sample.append([get_sequence(example[0]), ner])
        history = [
            [f"You are a scientific literature analyst. Extract scientific entities from sentences. The scientific entity category includes ['Method', 'Task', 'Metric', 'Material', 'Generic', 'OtherScientificTerm', 'Generic']. Sentence: {sample[0][0]}", str(sample[0][1])],
            [f"Sentence: {sample[1][0]}", str(sample[1][1])],
            [f"Sentence: {sample[2][0]}", str(sample[2][1])],
            [f"Sentence: {sample[3][0]}", str(sample[3][1])],
            [f"Sentence: {sample[4][0]}", str(sample[4][1])]
        ]
        question = f"Sentence: {doc['sentence']}"

    if data_name == "scicite":
        history = [
            [f"You are a scientific literature analyst. Identifying the intent of a citation in scientific papers. The citation intention includes ['method', 'background', 'result']. Sentence: {sample[0][0]}", sample[0][1]],
            [f"Sentence: {samples[1][0]}", samples[1][1]],
            [f"Sentence: {samples[2][0]}", samples[2][1]],
            [f"Sentence: {samples[3][0]}", samples[3][1]],
            [f"Sentence: {samples[4][0]}", samples[4][1]]
        ]
        question = f"Sentence: {doc['text']}"

    elif data_name == "scinli":
        history = [
            [f"You are a scientific literature analyst. Identify the semantic relationship between the following pair of sentences. The semantic relationship includes ['reasoning', 'entailment', 'contrasting', 'neutral']. Sentence1: {samples[0][0]} Sentence2:{samples[0][1]}", samples[0][2]],
            [f"Sentence1: {samples[1][0]} Sentence2:{samples[1][1]}", samples[1][2]],
            [f"Sentence1: {samples[2][0]} Sentence2:{samples[2][1]}", samples[2][2]],
            [f"Sentence1: {samples[3][0]} Sentence2:{samples[3][1]}", samples[3][2]],
        ]
        question = f"Sentence1: {doc['sentence1']} Sentence2:{doc['sentence2']}"

    response, _ = model.chat(
        tokenizer,
        question,
        history=history,
    )

    return response

if __name__ == "__main__":
    
    curriculum = "random"
    data_name = "scierc"
    checkpoint_path = '/data_share/model_hub/qwen/Qwen-72B-Chat'

    sample_file = f"data/sample/{data_name}.json"
    input_file = f"data/{data_name}/test.json"
    output_file = f"output/{data_name}/qwen_72b_chat_{curriculum}.json"

    dataset = read_dataset(input_file)
    # 读取已处理数据
    solved_sentence = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            solved_sentence = []
            for line in f:
                solved_doc = json.loads(line)
                solved_sentence.append(solved_doc[mark_map[data_name]])

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

    f_output = open(output_file, "a+", encoding="utf-8")

    acc_res = []
    for doc in tqdm.tqdm(dataset):
        if doc[mark_map[data_name]] in solved_sentence:
            continue
        completion = generate_sample(model, tokenizer, doc, sample[curriculum])
        doc["pred"] = completion
        f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
        f_output.flush()  # flush the buffer to disk

    f_output.close()
