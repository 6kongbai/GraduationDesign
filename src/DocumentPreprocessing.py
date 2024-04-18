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


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


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


def iter_block_items(parent):
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, parent)
            if is_image(paragraph, parent):
                yield get_ImagePart(paragraph, parent)
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


class MyParagraph:
    def __init__(self, index, text):
        self.text = text
        self.index = index
        self.sentences = cut_sent(text)

    def check(self):
        return len(self.text) < 16

    def check_java(self):
        return re.match(r'^\s*(package\s+[\w.]+\s*;|public|private|protected|class|interface)\s+\w+', self.text)

    def __str__(self):
        str = f"段落：{self.index}\n"
        for index, sentence in enumerate(self.sentences):
            str += f"第{index}句:{sentence} \n"
        return str


class MyDocument:
    def __init__(self, path: str, index: int):
        self.path = path
        self.doc_index = index
        self.paragraphs = None
        self.embeddings = None
        self.title = os.path.basename(path)
        self._process_document()

    def _process_document(self):
        doc = docx.Document(self.path)
        rows_to_append = []
        for part_index, part in enumerate(iter_block_items(doc)):
            if isinstance(part, Paragraph):
                # 判断该子对象是否是正文
                if part.style.name == 'Normal':
                    paragraph = MyParagraph(part_index, part.text)
                    # 判断该段落是否为空或者是否小于16个字符
                    if paragraph.check():
                        continue

                    rows_to_append.extend([
                        {'document_index': self.doc_index, 'paragraph_index': part_index, 'sent_index': sent_index,
                         'sent': sent}
                        for sent_index, sent in enumerate(paragraph.sentences)
                    ])
                # 判断是否为标题1。如果是Heading 2则判断是否为标2，以此类推。
                elif part.style.name == 'Heading 1':
                    pass
            elif isinstance(part, ImagePart):
                # 判断该子对象是否是图片
                pass
            elif isinstance(part, Table):
                # 判断该子对象是否是表格
                pass
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


def calculate(embeddings1, embeddings2):
    # 计算两两相似度
    return embeddings1 @ embeddings2.T


if __name__ == '__main__':
    name = 'B20200103213-唐睿智-21软件01-软件工程基础实训II(2023)-校内集中实训课程实训报告.docx'
    ROOT_DIR_P = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # 项目根目录
    word_path = os.path.join(ROOT_DIR_P,
                             f"files/{name}")  # pdf文件路径及文件名
    d1 = MyDocument(word_path, 0)
