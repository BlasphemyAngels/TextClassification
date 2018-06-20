from websearch import Segmenter

classifier = fasttext.load_model("fasttext_model.bin")

with open("./random100", "r") as f:
    jines = f.readlines()
    data = list(map(lambda line: line.rstrip("\n").split("\t")[1], lines))

seg = Segmenter()
cut_texts = [[token.word() for token in list(seg.cut_token(text)) if not token.is_stopword()] for text in data]
cut_texts = list(map(lambda x: " ".join(x), cut_texts))

print(cut_texts)
