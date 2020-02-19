import codecs
import os

books_original_path = r"../data/books_original"
books_utf_8_path = r"../data/books_utf_8"


# noinspection PyArgumentEqualDefault
def ReadFile(filePath, encoding=""):
    with codecs.open(filePath, 'r', encoding) as f:
        return f.read()


def WriteFile(filePath, u, encoding=""):
    with codecs.open(filePath, "w", encoding) as f:
        # f.write(u.encode(encoding,errors="ignore"))
        f.write(u)


def UTF8_2_GBK(src, dst):
    content = ReadFile(src, encoding="utf-8")
    WriteFile(dst, content, encoding="gb18030")


def GBK_2_UTF8(src, dst):
    content = ReadFile(src, encoding="gb18030")
    WriteFile(dst, content, encoding="utf-8")


def main():
    print("请选择转换模式, 需要转换的文件格式是G/U")
    a = input()
    if a == "U" or "G":
        for i, filename in enumerate(os.listdir(books_original_path)):
            original_path = os.path.join(books_original_path, filename)
            utf_8_path = os.path.join(books_utf_8_path, filename)
            if a == 'U':
                """++++++++UTF8-2-gbk+++++++++"""
                UTF8_2_GBK(original_path, utf_8_path)
            elif a == "G":
                """++++++++GBK-2-utf8+++++++++"""
                GBK_2_UTF8(original_path, utf_8_path)
    else:
        print("您的输入是{0}, 输入不正确，请重新输入".format(a))
    print("转换完成")


if __name__ == '__main__':
    main()
