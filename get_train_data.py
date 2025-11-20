import json
TrainJsonFile = r'D:\learn\TextCNN\data\baike_qa_train.json'
myTrainJsonFile = r'D:\learn\TextCNN\data\my_train.json'
ValJsonFile = r'D:\learn\TextCNN\data\baike_qa_valid.json'
myValJsonFile = r'D:\learn\TextCNN\data\my_valid.json'

wantedClass = {'教育':0,'健康':0,'生活':0,'娱乐':0,'游戏':0}
wantedNum = 5000
numWantedAll = wantedNum * 5

def main(inFile,MyFile):
    datas = open(inFile,'r',encoding='utf-8').readlines()
    f = open(MyFile,'w',encoding='utf-8')
    num = 0
    for line in datas:
        data = json.loads(line)
        cla = data['category'][0:2]
        if cla in wantedClass and wantedClass[cla] < wantedNum:
            json_data = json.dumps(data,ensure_ascii=False)
            f.write(json_data)
            f.write('\n')
            wantedClass[cla] += 1
            num += 1
            if num // 500:
                print("processed %s row" % num)
            if num >= numWantedAll:
                print("over")
                break

if __name__ == '__main__':
    main(TrainJsonFile,myTrainJsonFile)