import jieba.posseg as pseg

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