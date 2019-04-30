import jieba.posseg as pseg
import paras

def get_pseg(text):
    words = pseg.cut(text)
    res = []
    for w in words:
        if w.flag == 'x':
            pos = 'w'
        else:
            pos = w.flag
        res.append({'word': w.word, 'pos': pos})
    return res

def merge():
    with open(paras.TRAIN_DATA, 'r', encoding='utf-8') as f:
        tmp = f.read()
    with open(paras.DEV_DATA, 'r', encoding='utf-8') as f:
        tmp += f.read()
    with open(paras.TRAIN_DATA_MERGE, 'w', encoding='utf-8') as f:
        f.write(tmp)