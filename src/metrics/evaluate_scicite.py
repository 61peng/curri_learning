import json
import re
class_list = ["method", "background", "result"]
# class_list =["Motivation", "Extends", "CompareOrContrast", "Background", "Future", "Uses"]
# class_list = ["Introduction", "Related-Work", "Method", "Experiment", "Conclusion"]
# curriculum = "human"
# model_name = "llama2_7b_chat"
# file = open(f"output/mistral_instruct_votek+curriculum.json", "r")
# data = [json.loads(line.strip()) for line in file.readlines()]
file_path = "iccl/output_data/pred_data/manual_human_seed3407_bs1.json"
try:
    data = json.load(open(file_path, "r"))
    jline = False
except:
    data = [json.loads(line.strip()) for line in open(file_path, "r").readlines()]
    jline = True

pre_dict = {c:0 for c in class_list+["others"]}
# pre_dict = {c:0 for c in class_list}
gold_dict = {c:0 for c in class_list}
acc_dict = {c:0 for c in class_list}

for line in data:
    if jline:
        pre = line['pred']
        gold = line['label'].lower()
    else:
        pre = line['generated']
        gold = line['Y_TEXT'].lower()
    # pre = line['pred']
    # gold = line['label'].lower()
    # gold = gold.replace("related work", "related-work")
    pre = pre.lower()
    # pre = pre.replace("related work", "related-work")
    pre_label = ""
    gold_dict[gold] += 1
    # 预测结果与为类型一致
    if pre in class_list:
        pre_label = pre
    else:
        match = re.search(r'\b(method|background|result)\b', pre, re.IGNORECASE)
        if match:
            pre_label = match.group(0).lower()
        else:
            pre_label = "others"
            # pre_label = "background"
    pre_dict[pre_label] += 1
    if pre_label == gold:
        acc_dict[pre_label] += 1

f1_dict = {}
for c in class_list:
    pre = pre_dict[c]
    gold = gold_dict[c]
    acc = acc_dict[c]
    f1 = 2*acc/(pre+gold)
    f1_dict[c] = f1

# print("curriculum: ", curriculum)
print("pre_dict: ", pre_dict)
print("gold_dict: ", gold_dict)
print("acc_dict: ", acc_dict)

weighted_precison = 0
for c in class_list:
    if pre_dict[c] == 0:
        continue
    weighted_precison += acc_dict[c]*gold_dict[c] / pre_dict[c]
weighted_precison = weighted_precison/sum(gold_dict.values())


weighted_recall = 0
for c in class_list:
    weighted_recall += acc_dict[c]*gold_dict[c] / gold_dict[c]
weighted_recall = weighted_recall/sum(gold_dict.values())


weighted_f1 = 0
for c in class_list:
    weighted_f1 += f1_dict[c]*gold_dict[c]
weighted_f1 = weighted_f1/sum(gold_dict.values())
# weighted_f1 = 2*weighted_precison*weighted_recall/(weighted_precison+weighted_recall)


print("weighted_precison: ", weighted_precison)
print("weighted_recall: ", weighted_recall)
print("weighted_f1: ", weighted_f1)




