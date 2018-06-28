import re
from fastText import FastText

classifier = FastText.load_model("./fastText")

with open("./tmp_test_content", "r") as f:
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip("\n"), lines))

labels, probs = classifier.predict(lines)

labels = list(map(lambda x: re.sub("__label__", "", x[0]), labels))
print(labels)


probs = list(map(lambda x: str(x[0]), list(probs)))

res = list(zip(labels, probs))

res = list(map(lambda x: "\t".join(x), res))

res = list(zip(lines, res))
res = list(map(lambda x: "\t ".join(x), res))

res = "\n".join(res)

with open("res3", "w") as f:
    f.write(res)
