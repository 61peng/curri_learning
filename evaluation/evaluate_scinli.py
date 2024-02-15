import json
import re
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, f1_score

def evaluate_nli(json_file):
    labels = []
    preds = []
    porter_stemmer = PorterStemmer()
    try:
        docs = json.load(open(json_file, 'r'))
    except:
        docs = [json.loads(line.strip()) for line in open(json_file, 'r')]
    total = len(docs)
    for doc in docs:
        # 使用正则表达式匹配So the answer is后面的单词
        # match_few_shot = re.search(r"So the answer is (\w+)", doc["pred"], re.IGNORECASE)
        # if match_few_shot:
        #     preds.append(porter_stemmer.stem(match_few_shot.group(1)))
        # else:
        # 预测结果不为空
        if doc["pred"]:
            # doc["pred"] = doc["pred"].replace("contrast", "contrasting").replace("entail", "entailment")
            match = re.search(r'\b(reasoning|contrasting|entailment|neutral)\b', doc["pred"], re.IGNORECASE)
            if match:
                preds.append(porter_stemmer.stem(match.group(0)))
            else:
                preds.append("none")
        else:
            preds.append("none")
        # 添加真值的词根
        labels.append(porter_stemmer.stem(doc["label"]))
    
    # 确认pred_labels和gt_labels的长度一致
    assert len(preds) == len(labels)

    # correct = 0
    # for pred, label in zip(preds, labels):
    #     if pred == label:
    #         correct += 1
    # acc = correct / total

    acc = accuracy_score(labels, preds)

    marco_f1 = f1_score(labels, preds, average='macro')

    return acc, marco_f1

if __name__ == '__main__':

    acc, marco_f1 = evaluate_nli('output/scinli/llama2_70b_chat_llama.json')
    print("Accuracy: ", acc)
    # 保留小数点后两位
    print("macro-F1: ", round(marco_f1, 4))