# TextCNN-just-click
简单到批爆的textcnn，只要点击即可使用，可用来学习textcnn

你需要自己创建一个data文件夹，然后去  https://github.com/brightmart/nlp_chinese_corpus  里下载第三个百度百科数据集

之后依次点击 get_train_data.py ， get_wordlist.py ， sen2id.py ，train.py 即可得到自己的model啦。

不过在test的过程中如果觉得准确率过于低下，可以尝试更改get_train_data.py中的wantedNum中的值默认5000，不过我推荐你使用AI帮助你修改代码
或者不改
