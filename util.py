import json
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model

def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs

# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def get_sequence(word_list):
    sequence = ''.join([word if word.startswith((',', '.', ':', ';', '!', '?', '%')) else ' ' + word for word in word_list]).strip()
    return sequence

def read_dataset(path):
    if "scierc" in path:
        dataset = []
        with open (path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                data = {}
                data['sentence'] = get_sequence(line['sentence'])
                entity = []
                for ner in line['ner']:
                    entity.append([get_sequence(line['sentence'][ner[0]:ner[1]+1]), ner[2]])
                data['entity'] = entity
                relation = []
                for rel in line['relation']:
                    relation.append([get_sequence(line['sentence'][rel[0]:rel[1]+1]), get_sequence(line['sentence'][rel[2]:rel[3]+1]), rel[4]])
                data['relation'] = relation
                dataset.append(data)
    elif "scicite" or "scinli" in path:
        dataset = []
        with open (path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                dataset.append(line)
    
    return dataset