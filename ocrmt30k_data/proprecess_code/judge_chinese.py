import re
def judge_chinese(query):
    pattern="[\u4e00-\u9fa5]+"#中文正则表达式
    reg = re.compile(pattern) #生成正则对象
    results=reg.findall(query)
    if results:
        return True
    else:
        return False