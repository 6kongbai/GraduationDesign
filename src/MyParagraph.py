import re

from src.CodeType import CodeType


class MyParagraph:
    """
    文档段落
    """

    def __init__(self, text: str, code_type: CodeType):
        """
        :param text:段落内容
        :param code_type:代码类型
        """
        # 段落文本
        self.text: str = text
        # 代码类型
        self.code_type: CodeType = code_type
        # 句子列表
        self.sentences: list = cut_sent(text)

    def check(self) -> CodeType:
        """
        判断是否为代码段落
        :return:
        """
        if len(self.text) < 1:
            return CodeType.NULL
        elif self.code_check():
            return self.code_type
        elif self.exegesis_check():
            return CodeType.EXEGESIS
        elif self.check_chinese():
            return CodeType.CHINESE
        else:
            return CodeType.SUPPORTED

    def code_check(self) -> bool:
        """
        判断是否为代码
        :return:
        """
        code_regex = ''
        if self.code_type == CodeType.JAVA:
            code_regex = r'''(?x)  # 开启 VERBOSE 模式
                (?:package|import)\s+[\w.]+\s*;?  # 匹配导包或 import 语句
                |  # 或
                \b(?:public|private|protected|static|final|\s)+[\w<>]+\s+\w+\s*\([^()]*\)\s*  # 匹配方法定义
                |  # 或
                \b(?:public|private|protected|static|final|\s)+(?:class|interface|enum|@\w+)\s+\w+\s*(?:<[\w, ]+>)?\s*(?:implements\s+\w+(?:,\s*\w+)*\s*)?(?:extends\s+\w+)?\s*{.*?}  # 匹配类、接口、枚举、注解定义及其类体部分
                |  # 或
                \b(?:public|private|protected|static|final|\s)+\w+\s+\w+\s*(?:=\s*[^;]+)?\s*;  # 匹配变量定义
            '''
        if self.code_type == CodeType.PYTHON:
            code_regex = r'^\s*(import\s+[\w.]+\s*;|def\s+\w+\s*\(.*\)\s*:|class\s+\w+\s*:)'
        if self.code_type == CodeType.C:
            code_regex = r'^\s*(#include\s+[\w.]+\s*;|int\s+\w+\s*\(.*\)\s*:|class\s+\w+\s*:)'
        if self.code_type == CodeType.CPLUSPLUS:
            code_regex = r'^\s*(#include\s+[\w.]+\s*;|int\s+\w+\s*\(.*\)\s*:|class\s+\w+\s*:)'
        return re.search(code_regex, self.text) is not None

    def check_chinese(self) -> bool:
        """
        判断是否为中文
        :return:
        """
        chinese_regex = r'(?:[\u4e00-\u9fa5]|[0-9]).*[\u4e00-\u9fa5]*(?:[a-zA-Z0-9]*[,.?!;:\'"(){}\[\]<>]*)*'
        return re.match(chinese_regex, self.text) is not None

    def exegesis_check(self) -> bool:
        """
        判断是否为注释
        :return:
        """
        exegesis_regex = r"(?m)//.*?$|/\*.*?\*/"
        return re.match(exegesis_regex, self.text) is not None


def cut_sent(para: str) -> list:
    # 将单字符的断句符替换为换行符
    para = re.sub(r'([。！？\?])([^”’])', r"\1\n\2", para)
    # 将英文省略号替换为换行符
    para = re.sub(r'(\.{6})([^”’])', r"\1\n\2", para)
    # 将中文省略号替换为换行符
    para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，将分句符\n放到双引号后
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 去除段尾多余的换行符
    para = para.rstrip()
    # 分割成句子列表
    sentences = para.split("\n")
    return sentences
