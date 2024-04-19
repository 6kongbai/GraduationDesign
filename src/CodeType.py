from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class CodeType(AutoName):
    # java
    JAVA = auto()
    # python
    PYTHON = auto()
    # c
    C = auto()
    # c++
    CPLUSPLUS = auto()
    # 中文
    CHINESE = auto()
    # 空
    NULL = auto()
    # 不合格
    UNSUPPORTED = auto()
    # 合格
    SUPPORTED = auto()
    # 注释
    EXEGESIS = auto()

    @classmethod
    def get_code_type(cls: 'CodeType', type_name: str) -> 'CodeType':
        if type_name == 'java':
            return cls.JAVA
        elif type_name == 'python':
            return cls.PYTHON
        elif type_name == 'c':
            return cls.C
        elif type_name == 'c++':
            return cls.CPLUSPLUS
