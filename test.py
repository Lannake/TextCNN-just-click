import torch
import torch.nn as nn
import numpy as np
import json
import jieba
import os
import sys
from model import textCNN
from textCNN_data import textCNN_data, get_valdata

# # 强制使用CPU以避免GPU兼容性问题
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 参数定义（与训练时保持一致）
textCNN_param = {
    'vocab_size': 40848,
    'embed_dim': 60,
    'class_num': 5,
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}

# 类别映射（根据您的数据调整）
class_mapping = {
    0: '教育',
    1: '健康',
    2: '生活',
    3: '娱乐',
    4: '游戏'
}


def load_model(model_path='D:\learn\TextCNN\model\epoch_10.pkl'):
    """加载训练好的模型"""
    device = torch.device("cpu")

    # 初始化模型
    model = textCNN(textCNN_param)
    model.to(device)

    # 加载权重
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"成功加载模型权重: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None, device
    else:
        print(f"模型文件不存在: {model_path}")
        # 尝试加载其他可能的模型文件
        possible_files = ['weight.pkl', 'model/epoch_10.pkl', 'model/epoch_5.pkl']
        for file in possible_files:
            if os.path.exists(file):
                try:
                    model.load_state_dict(torch.load(file, map_location=device))
                    print(f"成功加载模型权重: {file}")
                    break
                except:
                    continue
        else:
            print("未找到可用的模型文件")
            return None, device

    model.eval()  # 设置为评估模式
    return model, device


def load_word_dict(word2id_file='word2id.txt'):
    """加载词表"""
    word2id = {}
    if not os.path.exists(word2id_file):
        print(f"词表文件不存在: {word2id_file}")
        return word2id

    try:
        with open(word2id_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    word_id = int(parts[1])
                    word2id[word] = word_id
        print(f"成功加载词表，共 {len(word2id)} 个词语")
    except Exception as e:
        print(f"加载词表失败: {e}")

    return word2id


def preprocess_text(text, word2id, max_len=20):
    """预处理单条文本"""
    # 分词
    words = jieba.cut(text, cut_all=False)

    # 转换为ID序列
    word_ids = []
    for word in words:
        if word in word2id:
            word_ids.append(word2id[word])

    # 调整长度
    if len(word_ids) > max_len:
        word_ids = word_ids[:max_len]
    else:
        word_ids.extend([0] * (max_len - len(word_ids)))

    return torch.tensor(word_ids, dtype=torch.long).unsqueeze(0)  # 添加batch维度


def test_single_text(model, device, text, word2id):
    """测试单条文本"""
    # 预处理文本
    input_tensor = preprocess_text(text, word2id)
    input_tensor = input_tensor.to(device)

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence, probabilities.squeeze().numpy()


def evaluate_model(model, device, test_loader):
    """在整个测试集上评估模型性能"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # 从batch中提取数据
            clas, sentences = batch

            # 将数据转换为张量并移动到设备
            sentences = torch.stack(sentences).to(device)
            clas = torch.tensor(clas, dtype=torch.long).to(device)

            outputs = model(sentences)
            _, predicted = torch.max(outputs.data, 1)

            # 处理每个样本
            for i in range(clas.size(0)):
                label = clas[i].item()

                # 检查标签是否在有效范围内
                if label < 0 or label >= 5:
                    continue

                total += 1
                class_total[label] += 1

                if predicted[i].item() == label:
                    correct += 1
                    class_correct[label] += 1

                # 保存预测结果
                all_predictions.append({
                    'true_label': label,
                    'predicted_label': predicted[i].item(),
                    'confidence': torch.softmax(outputs[i], dim=0)[predicted[i]].item()
                })

    # 计算总体准确率
    overall_accuracy = 100 * correct / total if total > 0 else 0

    # 计算每个类别的准确率
    class_accuracy = {}
    for i in range(5):
        if class_total[i] > 0:
            class_accuracy[class_mapping[i]] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracy[class_mapping[i]] = 0

    return overall_accuracy, class_accuracy, all_predictions, total


def interactive_test(model, device, word2id):
    """交互式测试"""
    print("\n=== 交互式测试模式 ===")
    print("输入文本进行分类（输入'quit'退出）")

    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() == 'quit':
            break

        if not text:
            continue

        predicted_class, confidence, probabilities = test_single_text(model, device, text, word2id)

        print(f"\n预测结果:")
        print(f"类别: {class_mapping[predicted_class]}")
        print(f"置信度: {confidence:.4f}")
        print(f"各类别概率:")
        for i, prob in enumerate(probabilities):
            print(f"  {class_mapping[i]}: {prob:.4f}")


def analyze_valdata(val_data):
    """分析验证数据的格式和标签分布"""
    print("\n=== 验证数据分析 ===")

    label_counts = {}
    line_lengths = []

    for i, line in enumerate(val_data[:10]):  # 只分析前10行
        try:
            items = line.strip().split(',')
            line_lengths.append(len(items))

            if len(items) > 0:
                label = int(items[0])
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

            if i < 3:  # 打印前3行的详细信息
                print(f"第{i + 1}行: 长度={len(items)}, 标签={items[0] if items else '无'}")
        except Exception as e:
            print(f"解析第{i + 1}行时出错: {e}")

    print(f"\n标签分布: {label_counts}")
    print(
        f"行长度统计: 最小={min(line_lengths) if line_lengths else 0}, 最大={max(line_lengths) if line_lengths else 0}")

    return label_counts


def main():
    """主函数"""
    print("=== TextCNN 模型测试 ===")

    # 加载模型
    model, device = load_model()
    if model is None:
        return

    # 加载词表
    word2id = load_word_dict()
    if not word2id:
        print("无法继续测试，词表加载失败")
        return

    # 测试选项
    print("\n选择测试模式:")
    print("1. 使用验证集测试")
    print("2. 交互式测试（手动输入文本）")
    print("3. 测试单个文件")
    print("4. 分析验证数据格式")

    choice = input("请输入选择 (1/2/3/4): ").strip()

    if choice == '1' or choice == '4':
        # 使用验证集测试或分析数据
        print("\n=== 验证集处理 ===")

        # 创建测试数据集
        try:
            # 使用验证数据文件
            val_data = get_valdata()
            if not val_data:
                print("验证数据为空，无法测试")
                return

            # 分析数据格式
            if choice == '4':
                analyze_valdata(val_data)
                if choice != '1':
                    return

            # 创建自定义数据集类
            class CustomTestDataset(torch.utils.data.Dataset):
                def __init__(self, data):
                    self.data = []
                    self.invalid_count = 0

                    for i, line in enumerate(data):
                        try:
                            items = line.strip().split(',')

                            # 检查数据格式
                            if len(items) < 21:
                                self.invalid_count += 1
                                continue

                            # 尝试解析标签
                            cla = int(items[0])

                            # 尝试两种标签格式
                            if 1 <= cla <= 5:
                                # 格式1: 标签为1-5
                                cla = cla - 1  # 转换为0-based
                            elif 0 <= cla <= 4:
                                # 格式2: 标签已经是0-4
                                pass
                            else:
                                # 无效标签
                                self.invalid_count += 1
                                continue

                            # 解析句子数据
                            sentence = []
                            for item in items[1:21]:
                                try:
                                    sentence.append(int(item))
                                except:
                                    sentence.append(0)  # 无效词用0代替

                            # 转换为张量
                            sentence_tensor = torch.tensor(sentence, dtype=torch.long)
                            self.data.append((cla, sentence_tensor))

                        except Exception as e:
                            self.invalid_count += 1
                            if i < 5:  # 只打印前5个错误的详细信息
                                print(f"第{i + 1}行解析错误: {e}")
                            continue

                    print(f"有效数据: {len(self.data)} 条")
                    print(f"无效数据: {self.invalid_count} 条")

                    # 统计标签分布
                    label_dist = {}
                    for cla, _ in self.data:
                        if cla not in label_dist:
                            label_dist[cla] = 0
                        label_dist[cla] += 1
                    print(f"标签分布: {label_dist}")

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx]

            # 创建数据集实例
            custom_dataset = CustomTestDataset(val_data)

            if len(custom_dataset) == 0:
                print("没有有效数据可供测试")
                return

            test_loader = torch.utils.data.DataLoader(
                custom_dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=lambda batch: (
                    [item[0] for item in batch],  # 标签列表
                    [item[1] for item in batch]  # 句子张量列表
                )
            )

            if choice == '1':
                # 评估模型
                accuracy, class_accuracy, predictions, total_samples = evaluate_model(model, device, test_loader)

                print(f"\n测试结果:")
                print(f"测试样本数: {total_samples}")
                print(f"总体准确率: {accuracy:.2f}%")
                print(f"各类别准确率:")
                for class_name, acc in class_accuracy.items():
                    print(f"  {class_name}: {acc:.2f}%")

                # 保存详细结果
                with open('test_results.txt', 'w', encoding='utf-8') as f:
                    f.write(f"测试样本数: {total_samples}\n")
                    f.write(f"总体准确率: {accuracy:.2f}%\n")
                    f.write("各类别准确率:\n")
                    for class_name, acc in class_accuracy.items():
                        f.write(f"  {class_name}: {acc:.2f}%\n")

                    f.write("\n详细预测结果:\n")
                    for i, pred in enumerate(predictions[:100]):  # 只保存前100条
                        f.write(f"{i + 1}. 真实: {class_mapping[pred['true_label']]}, "
                                f"预测: {class_mapping[pred['predicted_label']]}, "
                                f"置信度: {pred['confidence']:.4f}\n")

                print("详细结果已保存到 test_results.txt")

        except Exception as e:
            print(f"验证集处理失败: {e}")
            import traceback
            traceback.print_exc()

    elif choice == '2':
        # 交互式测试
        interactive_test(model, device, word2id)

    elif choice == '3':
        # 测试单个文件
        file_path = input("请输入测试文件路径: ").strip()
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return

        print(f"\n正在测试文件: {file_path}")
        # 这里可以添加文件测试逻辑，类似于验证集测试


    else:
        print("无效选择")


if __name__ == "__main__":
    main()
