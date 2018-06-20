import re
import fasttext

classifier = fasttext.load_model("./fasttext_model.bin")

with open("./cut_random100", "r") as f:
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip("\n"), lines))

labels = classifier.predict_proba(lines)
labels = list(map(lambda x: re.sub("__label__", "", x[0][0]) + " " + str(x[0][1]), labels))

res = list(zip(lines, labels))
res = list(map(lambda x: " ".join(x), res))

res = "\n".join(res)

with open("res", "w") as f:
    f.write(res)

