from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np
import os

trainDataFile = 'traindata_vec.txt'
valDataFile = 'valdata_vec.txt'

# 添加全局配置
MIN_SEQ_LEN = 5  # 最小序列长度，等于最大卷积核大小
VOCAB_SIZE = 40848  # 根据您的词表大小设置


def get_valdata(file=valDataFile):
    """获取验证数据"""
    if not os.path.exists(file):
        print(f"警告: 验证数据文件 {file} 不存在")
        return []

    valData = open(file, 'r', encoding='utf-8').read().split('\n')
    valData = list(filter(None, valData))
    random.shuffle(valData)
    return valData


class textCNN_data(Dataset):
    def __init__(self, data_file=trainDataFile, min_seq_len=MIN_SEQ_LEN, vocab_size=VOCAB_SIZE):
        # 检查文件是否存在
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件 {data_file} 不存在")

        self.min_seq_len = min_seq_len
        self.vocab_size = vocab_size

        # 读取数据
        trainData = open(data_file, 'r', encoding='utf-8').read().split('\n')
        trainData = list(filter(None, trainData))

        # 数据验证和清洗
        self.valid_data = []
        self.invalid_data = []

        for line in trainData:
            try:
                data = list(filter(None, line.split(',')))
                data = [int(x) for x in data]

                # 检查数据格式
                if len(data) < 2:  # 至少包含标签和一个词
                    self.invalid_data.append(line)
                    continue

                cla = data[0]
                sentence = data[1:]

                # 检查序列长度
                if len(sentence) < self.min_seq_len:
                    # 填充序列到最小长度
                    sentence = sentence + [0] * (self.min_seq_len - len(sentence))
                elif len(sentence) > 100:  # 限制最大长度
                    sentence = sentence[:100]

                # 检查词索引是否在有效范围内
                valid_sentence = []
                for word_idx in sentence:
                    if 0 <= word_idx < self.vocab_size:
                        valid_sentence.append(word_idx)
                    else:
                        # 将无效索引替换为0（填充符号）
                        valid_sentence.append(0)

                self.valid_data.append((cla, valid_sentence))

            except Exception as e:
                self.invalid_data.append(line)
                print(f"数据解析错误: {e}, 行内容: {line[:100]}...")

        # 统计信息
        print(f"有效数据: {len(self.valid_data)} 条")
        print(f"无效数据: {len(self.invalid_data)} 条")

        if len(self.valid_data) == 0:
            raise ValueError("没有有效数据可供训练")

        # 打乱数据
        random.shuffle(self.valid_data)

        # 打印样本信息
        if len(self.valid_data) > 0:
            sample_cla, sample_sentence = self.valid_data[0]
            print(f"样本序列长度: {len(sample_sentence)}")
            print(f"样本标签: {sample_cla}")
            print(f"样本词索引范围: {min(sample_sentence)} - {max(sample_sentence)}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        cla, sentence = self.valid_data[idx]

        # 将标签从1-based转换为0-based（1-5变为0-4）
        # 确保标签在有效范围内
        if cla < 1 or cla > 5:
            print(f"警告: 无效标签 {cla}，将其设置为1")
            cla = 1

        # 标签从1-5转换为0-4
        cla = cla - 1

        sentence_array = np.array(sentence, dtype=np.int64)

        return cla, sentence_array

    def get_data_stats(self):
        """获取数据统计信息"""
        if len(self.valid_data) == 0:
            return {}

        # 统计序列长度
        lengths = [len(sentence) for _, sentence in self.valid_data]

        # 统计词索引范围
        all_words = []
        for _, sentence in self.valid_data:
            all_words.extend(sentence)

        stats = {
            'total_samples': len(self.valid_data),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'min_word_idx': min(all_words) if all_words else 0,
            'max_word_idx': max(all_words) if all_words else 0,
            'invalid_data_count': len(self.invalid_data)
        }

        return stats


def textCNN_dataLoader(param):
    """创建数据加载器"""
    try:
        dataset = textCNN_data()

        # 打印数据统计信息
        stats = dataset.get_data_stats()
        print("数据统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        batch_size = param['batch_size']
        shuffle = param['shuffle']

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        # 返回一个空的DataLoader
        return DataLoader([])


if __name__ == "__main__":
    # 测试数据加载器
    try:
        dataset = textCNN_data()
        cla, sen = dataset.__getitem__(0)

        print(f"标签: {cla}")
        print(f"句子形状: {sen.shape}")
        print(f"句子内容: {sen}")

        # 测试数据加载器
        param = {'batch_size': 2, 'shuffle': True}
        loader = textCNN_dataLoader(param)

        print("数据加载器测试成功!")

    except Exception as e:
        print(f"测试失败: {e}")