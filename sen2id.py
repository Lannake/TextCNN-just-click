# -*- coding: utf-8 -*-

"""
将 title 的文本内容转换为向量id的形式
修改版：不依赖停用词文件和标签文件
"""

import json
import jieba
import random
import os

# 文件路径配置
trainFile = 'D:\learn\TextCNN\data\my_train.json'
valFile = r'D:\learn\TextCNN\data\baike_qa_valid.json'
word2idFile = 'word2id.txt'
trainDataVecFile = 'traindata_vec.txt'
valDataVecFile = 'valdata_vec.txt'
maxLen = 20


def load_word_dict(word2id_file):
    """加载词表文件"""
    word2id = {}
    id2word = {}

    if not os.path.exists(word2id_file):
        print(f"错误：词表文件 {word2id_file} 不存在")
        return word2id, id2word

    try:
        with open(word2id_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    print(f"警告：第 {line_num} 行格式不正确: {line}")
                    continue

                word = parts[0]
                try:
                    word_id = int(parts[1])
                    word2id[word] = word_id
                    id2word[word_id] = word
                except ValueError:
                    print(f"警告：第 {line_num} 行包含无效的ID: {line}")
                    continue

        print(f"成功加载词表，共 {len(word2id)} 个词语")
        return word2id, id2word

    except Exception as e:
        print(f"读取词表文件时出错: {e}")
        return word2id, id2word


def process_data(in_file, out_file, word2id):
    """处理数据文件，将文本转换为向量"""
    if not os.path.exists(in_file):
        print(f"错误：输入文件 {in_file} 不存在")
        return

    # 从数据中提取所有类别并创建映射
    category_to_id = {}
    next_id = 1  # 从1开始分配ID

    # 首先扫描所有数据以收集类别
    try:
        with open(in_file, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            lines = list(filter(None, lines))  # 移除空行

            for line in lines:
                try:
                    data = json.loads(line)
                    category = data.get('category', '')[0:2]  # 取前两位
                    if category and category not in category_to_id:
                        category_to_id[category] = next_id
                        next_id += 1
                except json.JSONDecodeError:
                    continue

        print(f"发现 {len(category_to_id)} 个类别: {category_to_id}")

        # 处理数据并生成向量
        with open(in_file, 'r', encoding='utf-8') as f_in, \
                open(out_file, 'w', encoding='utf-8') as f_out:

            lines = f_in.read().split('\n')
            lines = list(filter(None, lines))
            random.shuffle(lines)

            processed_count = 0
            for line in lines:
                try:
                    data = json.loads(line)
                    title = data.get('title', '')
                    category = data.get('category', '')[0:2]

                    if not title or not category:
                        continue

                    # 获取类别ID
                    if category not in category_to_id:
                        # 如果遇到新类别（理论上不会发生），动态添加
                        category_to_id[category] = next_id
                        next_id += 1

                    category_id = category_to_id[category]

                    # 分词并转换为向量
                    words = jieba.cut(title, cut_all=False)
                    vector = [category_id]

                    for word in words:
                        if word in word2id:
                            vector.append(word2id[word])

                    # 调整向量长度
                    if len(vector) > maxLen + 1:
                        vector = vector[:maxLen + 1]
                    else:
                        vector.extend([0] * (maxLen + 1 - len(vector)))

                    # 写入输出文件
                    f_out.write(','.join(map(str, vector)) + '\n')
                    processed_count += 1

                    if processed_count % 1000 == 0:
                        print(f"已处理 {processed_count} 条数据")

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"处理数据时出错: {e}")
                    continue

            print(f"处理完成！共处理 {processed_count} 条数据，输出到 {out_file}")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")


def main():
    # 加载词表
    word2id, id2word = load_word_dict(word2idFile)

    if not word2id:
        print("无法继续，词表加载失败")
        return

    # 处理验证数据
    print("开始处理验证数据...")
    process_data(valFile, valDataVecFile, word2id)

    # 如果需要处理训练数据，取消下面的注释
    print("开始处理训练数据...")
    process_data(trainFile, trainDataVecFile, word2id)


if __name__ == '__main__':
    main()