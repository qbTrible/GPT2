import os

books_utf_8 = r"../data/books_utf_8"
books_tokenized = r"../data/books_tokenized"
vocab_path = "../data/vocab.txt"

if not os.path.exists(books_tokenized):
    os.makedirs(books_tokenized)

with open(vocab_path, "r+", encoding="utf-8") as f:
    tokens = f.read().split()

for i, filename in enumerate(os.listdir(books_utf_8)):

    if i < len(os.listdir(books_utf_8)):
        print("正在处理第", i + 1, "个文件{0}".format(filename))
        with open(os.path.join(books_utf_8, filename), "r+", encoding="utf-8") as f:
            dst = ["0"]
            w = f.read(1)
            while w:
                print(w)
                if w == '\n' or w == '\r' or w == '\t' or ord(w) == 307:
                    dst.append("1")
                    pass
                elif w == ' ':
                    dst.append("3")
                    pass
                else:
                    try:
                        dst.append(str(tokens.index(w)))
                    except Exception:
                        # exit()
                        dst.append("2")

                w = f.read(1)
        with open(os.path.join(books_tokenized, "{}".format(filename)), "w+", encoding="utf-8") as df:
            df.write(" ".join(dst))
print("完成所有文件的编码!")