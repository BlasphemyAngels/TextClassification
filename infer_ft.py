import jieba
from utils import load_stop_list

stops = load_stop_list()

with open("./badcase", "r") as f:
    lines = f.readlines()
    #  data = list(map(lambda line: line.rstrip("\n").split("\t")[1], lines))
    data = list(map(lambda line: line.rstrip("\n"), lines))

cut_texts = []
for text in data:
    cut_text = [word for word in jieba.cut(text) if False or (word not in stops)]
    cut_texts.append(" ".join(cut_text))

lines = "\n".join(cut_texts)

with open("badcase", "w") as f:
    f.write(lines)

