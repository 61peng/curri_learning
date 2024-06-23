import json
import re
import csv
def evaluate_ner_list(json_file):
    gt_labels = []
    pred_labels = []
    # with open(json_file, 'r') as f:
    #     lines = f.readlines()
        # for i, line in enumerate(lines):
        #     doc = json.loads(line)
        #     try:
        #         pred_label = eval(doc["pred"])
        #     except:
        #         pred_label = []
            
        #     pred_labels.append(pred_label)
        #     gt_labels.append(doc["entity"])
    lines = json.load(open(json_file, 'r'))
    for line in lines:
        gt_label = eval(line['Y_TEXT'])
        try:
            pred_label = eval(line['generated'])
        except:
            pred_label = []
        pred_labels.append(pred_label)
        gt_labels.append(gt_label)
    scores = count_sample(gt_labels, pred_labels)
    return scores


def evaluate_ner_dict(json_file):
    # 读取jsonl文件
    gt_labels = []
    pred_labels = []
    with open(json_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # if i < 9000 or i > 10000:
            #     continue
            doc = json.loads(line)
            if True:
                try:
                    pattern = r'\{[^{}]*\}'
                    matches = re.findall(pattern, doc["pred"])
                    raw_dict = eval(matches[0].replace("\'\'", ""))
                    if type(raw_dict) == list:
                        raw_dict = raw_dict[0]
                    inference_dict = {}
                    for k, v in raw_dict.items():
                        if v == None or v == []:
                            continue
                        inference_dict[k] = v
                    # 将infernece_dict转化为列表
                    for k, v in inference_dict.items():
                        for e in v:
                            pred_labels.append((k, str(e).lower()))
                    # pred_labels.append(inference_dict)
                    
                except:
                    # gt_label = json.loads(doc["output"].replace("'", "\""))

                    # pred_label = json.loads(doc["inference"].replace("'s ", "\'s "))
                    print(f"模型预测结果为{doc['pred']}")
                    pred_labels.append([])
                
                true_entities = []
                
                for k, v in doc["output"].items():
                    for e in v:
                        true_entities.append((k, str(e).lower()))
                gt_labels.append(doc["output"])
        
        scores = count_sample(gt_labels, pred_labels)
        return scores

def count_sample(gt_labels, pred_labels):
    # 计算precision, recall, f1
    metrics_list = {}
    scores = {}
    entity_types = ['Method', 'Task', 'Metric', 'Material', 'OtherScientificTerm', 'Generic']
    sample_dict = {'tp': 0, 'fp': 0, 'fn': 0}
    metrics_list['all'] = sample_dict
    all_tp = 0
    all_fp = 0
    all_fn = 0
    for et in entity_types:
        tp = 0
        fp = 0
        fn = 0
        for gt_label, pred_label in zip(gt_labels, pred_labels):
            
            for p in pred_label:
                if type(p) == list and len(p) == 2:
                    if p[1] == et and p in gt_label:
                        tp += 1
                        all_tp += 1
                    elif p[1] == et and p not in gt_label:
                        fp += 1
                        all_fp += 1
            for t in gt_label:
                if t[1] == et and t not in pred_label:
                    fn += 1
                    all_fn += 1

        scores[et] = calculate_matric(tp, fp, fn)

    scores['all'] = calculate_matric(all_tp, all_fp, all_fn)

    return scores

def calculate_matric(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {'Precision': round(precision, 4), 'Recall': round(recall, 4), 'F1 Score': round(f1, 4)}

if __name__ == '__main__':
    input_path = 'output/pred_data/scinli/llama2-70b/manual+random_seed1_bs1.json'
    score = evaluate_ner_list(input_path)
    print(json.dumps(score, indent=4))