import torch
import os
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader

from model import textCNN
import textCNN_data

# 强制使用CPU以避免GPU兼容性问题
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 参数定义
textCNN_param = {
    'vocab_size': 40848,  # 使用实际词表大小
    'embed_dim': 60,
    'class_num': 5,
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}

dataLoader_param = {
    'batch_size': 64,  # 减小batch_size
    'shuffle': True
}


def validate_model(net, val_data, criterion, device):
    """验证模型性能"""
    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for clas, sentences in val_data:
            # 将数据移动到正确的设备
            sentences = sentences.type(torch.LongTensor).to(device)
            clas = clas.type(torch.LongTensor).to(device)

            outputs = net(sentences)
            loss = criterion(outputs, clas)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += clas.size(0)
            correct += (predicted == clas).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    net.train()
    return total_loss / len(val_data) if len(val_data) > 0 else 0, accuracy


def train():
    # 由于GPU兼容性问题，强制使用CPU
    device = torch.device("cpu")
    print(f"使用设备: {device}")

    print("初始化网络...")
    net = textCNN(textCNN_param)
    net.to(device)

    # 手动初始化权重
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    net.apply(init_weights)

    weightFile = 'weight.pkl'
    if os.path.exists(weightFile):
        print('加载预训练权重...')
        try:
            net.load_state_dict(torch.load(weightFile, map_location=device))
            print('权重加载成功')
        except Exception as e:
            print(f'权重加载失败: {e}')

    print(net)

    # 初始化数据集
    print('初始化数据集...')
    try:
        # 创建数据集实例
        dataset = textCNN_data.textCNN_data()
        print(f"数据集大小: {len(dataset)}")

        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=dataLoader_param['batch_size'],
                                  shuffle=dataLoader_param['shuffle'])
        val_loader = DataLoader(val_dataset, batch_size=dataLoader_param['batch_size'],
                                shuffle=False)

        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")

    except Exception as e:
        print(f"数据集初始化失败: {e}")
        return

    # 优化器和损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # 创建日志文件
    timestamp = time.strftime('%y%m%d%H%M')
    log = open(f'log_{timestamp}.txt', 'w', encoding='utf-8')
    log.write('epoch step loss\n')

    print("开始训练...")

    best_accuracy = 0
    for epoch in range(10):  # 先训练10个epoch进行测试
        net.train()
        epoch_loss = 0
        total_batches = len(train_loader)

        for i, (clas, sentences) in enumerate(train_loader):
            # 移动到设备
            sentences = sentences.type(torch.LongTensor).to(device)
            clas = clas.type(torch.LongTensor).to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = net(sentences)
            loss = criterion(outputs, clas)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 打印训练信息
            if (i + 1) % 10 == 0:
                print(f"Epoch: {epoch + 1}/10, Batch: {i + 1}/{total_batches}, Loss: {loss.item():.4f}")

                log_data = f"{epoch + 1} {i + 1} {loss.item():.4f}\n"
                log.write(log_data)
                log.flush()

        # 每个epoch结束后进行验证
        val_loss, accuracy = validate_model(net, val_loader, criterion, device)
        print(f"验证集 - Epoch: {epoch + 1}, Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(net.state_dict(), 'best_model.pkl')
            print(f"保存最佳模型，准确率: {accuracy:.2f}%")

        # 定期保存模型
        if (epoch + 1) % 5 == 0:
            os.makedirs('model', exist_ok=True)
            model_path = f"model/epoch_{epoch + 1}.pkl"
            torch.save(net.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")

    # 关闭日志文件
    log.close()
    print("训练完成！")


if __name__ == "__main__":
    train()