import jieba
import json

trainFile = r'D:\learn\TextCNN\data\my_train.json'
stopWordFile = 'stopwords.txt'
word2idFile = 'word2id.txt'
lengthFile = 'sen_length.txt'

def read_stopword(file):
    data = open(file, 'r', encoding='utf-8').read().split('\n')
    print(data[0:5])
    return data

def main():
    worddict = {}
    stoplist = read_stopword(stopWordFile)
    datas = open(trainFile, 'r', encoding='utf-8').read().split('\n')
    datas = list(filter(None, datas))
    data_num = len(datas)
    len_dic = {}
    for line in datas:
        line = json.loads(line)
        title = line['title']
        title_seg = jieba.cut(title,cut_all=False)
        length = 0
        for w in title_seg:
            if w in stoplist:
                continue
            length+=1
            if w in worddict:
                worddict[w]+=1
            else:
                worddict[w]=1
    if length in len_dic:
        len_dic[length]+=1
    else:
        len_dic[length]=1

    worldlist = sorted(worddict.items(),key=lambda item:item[1],reverse=True)
    f = open(word2idFile, 'w', encoding='utf-8')
    ind = 0
    for w in worldlist:
        line = w[0] + ' ' + str(ind) + ' ' + str(w[1]) + '\n'
        ind +=1
        f.write(line)

    for k,v in len_dic.items():
        len_dic[k] = round(v * 1.0 / data_num, 3)
    len_list = sorted(len_dic.items(),key=lambda item:item[0],reverse=True)
    f = open(lengthFile, 'w', encoding='utf-8')
    for t in len_list:
        d = str(t[0]) + ' ' + str(t[1]) + '\n'
        f.write(d)

if __name__ == '__main__':
    main()