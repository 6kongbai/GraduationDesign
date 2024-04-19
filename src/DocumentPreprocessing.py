from types import NoneType
from sentence_transformers import SentenceTransformer
from docx.document import Document
from docx.text.paragraph import Paragraph
from docx.parts.image import ImagePart
from docx.table import _Cell, Table
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from src.CodeType import CodeType
from src.MyParagraph import MyParagraph
import torch
import pandas as pd
import docx
import os

device = torch.device("cuda")
# 加载预训练模型并将其移动到 GPU
model = SentenceTransformer('DMetaSoul/Dmeta-embedding').to(device)


class MyDocument:
    """
    文档
    """

    def __init__(self, path: str, index: int, code_type: str):
        """
        初始化文档
        :param path:文档路径
        :param index:文档序号
        :param code_type:代码类型 0:java 1:python 2:c 3:c++
        """
        # 文档路径
        self.path: str = path
        # 文档索引
        self.doc_index: int = index
        # 文档对象
        self.doc: Document = docx.Document(path)
        # 代码类型
        self.code_type: CodeType = CodeType.get_code_type(code_type)
        # 文档段落
        self.paragraphs: pd.DataFrame = pd.DataFrame()
        # 文档句子嵌入向量
        self.embeddings: torch.Tensor = torch.Tensor()
        # 文档标题
        self.title: str = os.path.basename(path)
        # 文档中的源代码
        self.codes: list = []
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

    def block_process(self, block, block_index, rows_to_append) -> CodeType:
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
                paragraph = MyParagraph(block.text, self.code_type)
                flag = paragraph.check()
                if flag != CodeType.NULL and flag != self.code_type and len(paragraph.text) > 16:
                    rows_to_append.extend([
                        {
                            'document_index': self.doc_index,
                            'paragraph_index': block_index,
                            'sent_index': sent_index,
                            'sent': sent
                        }
                        for sent_index, sent in enumerate(paragraph.sentences)
                    ])
                return flag

            # 判断是否为标题1。如果是Heading 2则判断是否为标2，以此类推。
            elif block.style.name == 'Heading 1':
                pass
        elif isinstance(block, ImagePart):
            # 判断该子对象是否是图片
            pass
        elif isinstance(block, Table):
            # 判断该子对象是否是表格
            pass

    def code_process(self, block) -> CodeType:
        paragraph = MyParagraph(block.text, self.code_type)
        flag = paragraph.check()
        if flag != CodeType.CHINESE:
            self.codes[-1] += paragraph.text + '\n'
        return flag

    def _parse_word(self):
        """
        解析word文档
        :return:
        """
        rows_to_append = []
        it = iter(self.doc.element.body)  # 创建一个迭代器
        block_index = 0
        flag = True  # 当为True时，表示当前部分为非代码部分，否则为代码部分
        while True:
            try:
                part = next(it)  # 获取下一个部分
                block = self.paser_part(part)  # 获取当前部分
                if isinstance(block, NoneType):
                    continue
                if flag:
                    part_type = self.block_process(block, block_index, rows_to_append)  # 处理当前部分
                    if part_type == self.code_type:
                        self.codes.append("""""")
                        self.code_process(block)
                        flag = False
                else:
                    part_type = self.code_process(block)
                    if part_type == CodeType.CHINESE:
                        flag = True
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

    def calculate_similarity(self, other_doc: 'MyDocument'):
        similarity_matrix = calculate(self.embeddings, other_doc.embeddings)
        max_values, indices = torch.max(similarity_matrix, dim=1)
        for i, j in enumerate(indices):
            if max_values[i] > 0.85:
                print('原句子:', self.get_sentence_info(i)['sent'], '\n相似句子:',
                      other_doc.get_sentence_info(j.item())['sent'],
                      f'\n相似分数:{max_values[i]:.2f}\n')

    def get_sentence_info(self, index):
        return self.paragraphs.iloc[index, :]


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
    word_path = os.path.join(ROOT_DIR_P, f"files/{name}")  # pdf文件路径及文件名
    d1 = MyDocument(word_path, 0, 'java')
    print(d1.paragraphs)
