import json
import re
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
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
        if doc["generated"]:
            # doc["pred"] = doc["pred"].replace("contrast", "contrasting").replace("entail", "entailment")
            match = re.search(r'\b(reasoning|contrasting|entailment|neutral)\b', doc["generated"], re.IGNORECASE)
            if match:
                preds.append(porter_stemmer.stem(match.group(0)))
            else:
                preds.append("contrast")
        else:
            preds.append("contrast")
        # 添加真值的词根
        labels.append(porter_stemmer.stem(doc["Y_TEXT"]))
    
    # 确认pred_labels和gt_labels的长度一致
    assert len(preds) == len(labels)

    # correct = 0
    # for pred, label in zip(preds, labels):
    #     if pred == label:
    #         correct += 1
    # acc = correct / total
    # 输出混淆矩阵
    print(confusion_matrix(labels, preds, labels=['reason', 'contrast', 'entail', 'neutral']))
    acc = accuracy_score(labels, preds)

    marco_f1 = f1_score(labels, preds, average='macro')

    return acc, marco_f1

if __name__ == '__main__':

    acc, marco_f1 = evaluate_nli('output/pred_data/scinli/llama2-70b/manual+random_seed3407_bs1.json')
    # 保留小数点后两位
    print("macro-F1: ", round(marco_f1, 4))
    print("Accuracy: ", acc)
