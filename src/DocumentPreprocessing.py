import re
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import docx
from docx.document import Document
from docx.text.paragraph import Paragraph
from docx.parts.image import ImagePart
from docx.table import _Cell, Table
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
import os

device = torch.device("cuda")
# 加载预训练模型并将其移动到 GPU
model = SentenceTransformer('DMetaSoul/Dmeta-embedding').to(device)


class MyParagraph:
    """
    文档段落
    """

    def __init__(self, index, text):
        """
        :param index:
        :param text:
        """
        self.text: str = text
        self.index: int = index
        self.sentences: list = cut_sent(text)

    def check(self):
        """
        判断是否为空或者是否小于16个字符
        :return:
        """
        return len(self.text) < 16

    def check_java(self):
        """
        判断是否为java代码
        :return:
        """
        return re.match(r'^\s*(package\s+[\w.]+\s*;|public|private|protected|class|interface)\s+\w+', self.text)

    def check_python(self):
        """
        判断是否为python代码
        :return:
        """
        return re.match(r'^\s*(import\s+[\w.]+\s*;|def\s+\w+\s*\(.*\)\s*:|class\s+\w+\s*:)', self.text)

    def check_c(self):
        """
        判断是否为c代码
        :return:
        """
        return re.match(r'^\s*(#include\s+[\w.]+\s*;|int\s+\w+\s*\(.*\)\s*:|class\s+\w+\s*:)', self.text)

    def check_cplusplus(self):
        """
        判断是否为c++代码
        :return:
        """
        return re.match(r'^\s*(#include\s+[\w.]+\s*;|int\s+\w+\s*\(.*\)\s*:|class\s+\w+\s*:)', self.text)

    def check_chinese(self):
        """
        判断是否为中文
        :return:
        """
        return re.match(r'^[\u4e00-\u9fa5_a-zA-Z0-9]+$', self.text)


class MyDocument:
    """
    文档
    """

    def __init__(self, path: str, index: int):
        """
        初始化文档
        :param path:
        :param index:
        """
        # 文档路径
        self.path: str = path
        # 文档索引
        self.doc_index: int = index
        # 文档对象
        self.doc: Document = docx.Document(path)
        # 文档段落
        self.paragraphs: pd.DataFrame = pd.DataFrame()
        # 文档句子嵌入向量
        self.embeddings: torch.Tensor = torch.Tensor()
        # 文档标题
        self.title: str = os.path.basename(path)
        # 解析文档
        self._parse_word()

    def paser_part(self, block):
        """
        处理文档部分
        :param block:
        :return:
        """
        if isinstance(block, CT_P):
            para = Paragraph(block, self.doc)
            if is_image(para, self.doc):
                return get_ImagePart(para, self.doc)
            return Paragraph(block, self.doc)
        elif isinstance(block, CT_Tbl):
            return Table(block, self.doc)

    def block_process(self, block, block_index, rows_to_append):
        """
        处理文档部分
        :param block:
        :param block_index:
        :param rows_to_append:
        :return:
        """
        if isinstance(block, Paragraph):
            # 判断该子对象是否是正文
            if block.style.name == 'Normal':
                paragraph = MyParagraph(block_index, block.text)
                # 判断该段落是否为空或者是否小于16个字符
                if paragraph.check():
                    return
                rows_to_append.extend([
                    {'document_index': self.doc_index, 'paragraph_index': block_index, 'sent_index': sent_index,
                     'sent': sent}
                    for sent_index, sent in enumerate(paragraph.sentences)
                ])
            # 判断是否为标题1。如果是Heading 2则判断是否为标2，以此类推。
            elif block.style.name == 'Heading 1':
                pass
        elif isinstance(block, ImagePart):
            # 判断该子对象是否是图片
            pass
        elif isinstance(block, Table):
            # 判断该子对象是否是表格
            pass

    def _parse_word(self):
        """
        解析word文档
        :return:
        """
        rows_to_append = []
        it = iter(self.doc.element.body)  # 创建一个迭代器
        block_index = 0
        while True:
            try:
                part = next(it)  # 获取下一个部分
                block = self.paser_part(part)  # 处理当前部分
                self.block_process(block, block_index, rows_to_append)
                block_index += 1
            except StopIteration:
                break  # 迭代结束时退出循环
        self.paragraphs = pd.DataFrame(rows_to_append,
                                       columns=['document_index', 'paragraph_index', 'sent_index', 'sent'])
        # 计算句子嵌入向量
        self.calculate_embeddings()

    def calculate_embeddings(self):
        # 提取句子文本，保留段落号和句子号信息
        sentences = self.paragraphs['sent'].values
        # 使用model.encode计算句子嵌入矩阵
        self.embeddings = model.encode(sentences, convert_to_tensor=True).to(device)

    def get_sentence_info(self, index):
        return self.paragraphs.iloc[index, :]


def cut_sent(para):
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


# 该行只能有一个图片
def is_image(graph: Paragraph, doc: Document):
    images = graph._element.xpath('.//pic:pic')  # 获取所有图片
    for image in images:
        for img_id in image.xpath('.//a:blip/@r:embed'):  # 获取图片id
            part = doc.part.related_parts[img_id]  # 根据图片id获取对应的图片
            if isinstance(part, ImagePart):
                return True
    return False


# 获取图片（该行只能有一个图片）
def get_ImagePart(graph: Paragraph, doc: Document):
    images = graph._element.xpath('.//pic:pic')  # 获取所有图片
    for image in images:
        for img_id in image.xpath('.//a:blip/@r:embed'):  # 获取图片id
            part = doc.part.related_parts[img_id]  # 根据图片id获取对应的图片
            if isinstance(part, ImagePart):
                return part
    return None


def calculate(embeddings1, embeddings2):
    # 计算两两相似度
    return embeddings1 @ embeddings2.T


if __name__ == '__main__':
    name = 'B20200103213-唐睿智-21软件01-软件工程基础实训II(2023)-校内集中实训课程实训报告.docx'
    ROOT_DIR_P = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # 项目根目录
    word_path = os.path.join(ROOT_DIR_P,
                             f"files/{name}")  # pdf文件路径及文件名
    d1 = MyDocument(word_path, 0)
