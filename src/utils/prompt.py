import torch

TASK_DESCRIPTIONS = {
    
    "scicite": "Identify the intent of a citation in scientific papers. Choose the citation intention of the following sentence from ['method', 'background', 'result']." ,
    "scinli": "Identify the semantic relationship between the following pair of sentences. The semantic relationship includes ['reasoning', 'entailment', 'contrasting', 'neutral'].",
    "scierc": "Extract scientific entities from sentences. The scientific entity category includes ['Method', 'Task', 'Metric', 'Material', 'Generic', 'OtherScientificTerm', 'Generic']."
}

def get_templet(task, model):
    
    if "Mixtral" in model:

        templet = "[INST]You are a scientific literature analyst." + TASK_DESCRIPTIONS[task] + "\n{sentence}[/INST]"

        
    elif "Llama-2" in model:
        
        sys_message = "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"

        templet = "<s>[INST] "+sys_message+"\n\n"+TASK_DESCRIPTIONS[task]+"{sentence} [/INST]"

    elif "Qwen" in model:

        templet = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + TASK_DESCRIPTIONS[task] + "{sentence}<|im_end|>\n"

    return templet

def add_demonstrations(model, templet, samples, input):
    
    if "Mixtral" in model:

        for i in range(len(samples)):
            if i == 0:
                prompt = templet.format(sentence=samples[i][1])+f"{samples[i][2]}</s>"
            else:
                prompt += f"[INST]{samples[i][1]}[/INST]{samples[i][2]}</s>"
            
        prompt += f"[INST]{input}[/INST]"
        
    elif "Llama-2" in model:

        for i in range(len(samples)):
            if i == 0:
                prompt = templet.format(sentence=samples[i][1])+f"{samples[i][2]}</s>"
            else:
                prompt += f"<s>[INST] {samples[i][1]} [/INST] {samples[i][2]} </s>"
        
        prompt += f"<s>[INST] {input} [/INST]"

    elif "Qwen" in model:
        for i in range(len(samples)):
            if i == 0:
                prompt = templet.format(sentence=samples[i][1])+f"<|im_start|>assistant\n{samples[i][2]}<|im_end|>\n"
            else:
                prompt += f"<|im_start|>user\n{samples[i][1]}<|im_end|>\n<|im_start|>assistant\n{samples[i][2]}<|im_end|>\n"
        
        prompt += f"<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"


    return prompt

def select_coverage(sentences_with_labels, required_labels, sample_num):
    selected_samples = []
    label_coverage = {label: False for label in required_labels}
    
    # First pass: select samples to cover all required labels
    for id, sentence, label in sentences_with_labels:
        if label in label_coverage and not label_coverage[label]:
            selected_samples.append((id, sentence, label))
            label_coverage[label] = True
        if len(selected_samples) == sample_num:
            break
    
    # Second pass: fill in remaining slots if any
    if len(selected_samples) < sample_num:
        for id, sentence, label in sentences_with_labels:
            if (id, sentence, label) not in selected_samples:
                selected_samples.append((id, sentence, label))
            if len(selected_samples) == sample_num:
                break
    
    return selected_samples

def select_samples_multilabel(sentences_with_labels, required_labels, sample_num):
    selected_samples = []
    label_coverage = {label: False for label in required_labels}
    label_count = {label: 0 for label in required_labels}
    
    # First pass: select samples to cover all required labels
    for id, sentence, str_labels_list in sentences_with_labels:
        labels_list = eval(str_labels_list)
        labels = [l[1] for l in labels_list]
        if any(label in label_coverage and not label_coverage[label] for label in labels):
            selected_samples.append((id, sentence, str_labels_list))
            for label in labels:
                if label in label_coverage:
                    label_coverage[label] = True
                if label in label_count:
                    label_count[label] += 1
        if len(selected_samples) == sample_num:
            break
    
    # Second pass: fill in remaining slots to maximize label counts
    if len(selected_samples) < sample_num:
        for id, sentence, str_labels_list in sentences_with_labels:
            labels_list = eval(str_labels_list)
            labels = [l[1] for l in labels_list]
            if (id, sentence, str_labels_list) not in selected_samples:
                selected_samples.append((id, sentence, str_labels_list))
                for label in labels:
                    if label in label_count:
                        label_count[label] += 1
                if len(selected_samples) == sample_num:
                    break
    
    # Sort selected samples to prioritize those covering more labels
    selected_samples.sort(key=lambda x: len(set(x[1]) & set(required_labels)), reverse=True)
    
    # Reduce to the first 5 samples if more than 5
    selected_samples = selected_samples[:5]
    
    return selected_samples

def calculate_label_ppl(encodings, prompt_len, model):
    # 只算label的困惑度

    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    target_ids[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    ppl = torch.exp(outputs.loss)

    return ppl

def calculate_all_ppl(encodings, model):
    max_length = model.config.max_position_embeddings
    # max_length = 10
    stride = 512  # 步长
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):  # begin_loc = 0, 5, 20, 30, ...
        end_loc = min(begin_loc + max_length, seq_len)  # 输入序列的长度或者最大长度
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl