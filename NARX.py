import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 定义NARX模型
class NARX(nn.Module):
    def __init__(self):
        super(NARX, self).__init__()
        # TODO:
        self.input_dim = 96 + 10  # 96个特征 + 10个类别的反馈
        self.hidden_dims = [10, 20, 10]
        # TODO:
        self.output_dim = 10  # 10个类别
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.fc3 = nn.Linear(self.hidden_dims[1], self.hidden_dims[2])
        self.fc4 = nn.Linear(self.hidden_dims[2], self.output_dim)

    def forward(self, x, y_prev):
        combined_input = torch.cat((x, y_prev), dim=1)
        hidden1 = torch.tanh(self.fc1(combined_input))
        hidden2 = torch.tanh(self.fc2(hidden1))
        hidden3 = torch.tanh(self.fc3(hidden2))
        y = nn.functional.softmax(self.fc4(hidden3), dim=1)
        return y

# 生成模拟数据
# TODO:
x_data = torch.randn(1000, 96, 96)
y_data = torch.randint(0, 10, (1000,))

# 划分数据集
# TODO:
x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 初始化模型、损失函数和优化器
model = NARX()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 批量大小
batch_size = 32

# 训练模型
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        y_prev_batch = torch.zeros(len(x_batch), 10)

        batch_loss = 0
        for t in range(96):
            y_pred = model(x_batch[:, t, :], y_prev_batch)
            batch_loss += criterion(y_pred, y_batch)
            
            # 更新 y_prev
            y_prev_batch = y_pred.detach()

        epoch_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1}/100, Training Loss: {epoch_loss / (96 * (len(x_train) // batch_size))}")

    # 验证模型
    model.eval()
    val_loss = 0
    val_preds = []
    y_prev_val = torch.zeros(len(x_val), 10)
    with torch.no_grad():
        for t in range(96):
            y_pred_val = model(x_val[:, t, :], y_prev_val)
            val_loss += criterion(y_pred_val, y_val)
            y_prev_val = y_pred_val.detach()
            val_preds.append(torch.argmax(y_pred_val, dim=1).cpu().numpy())
    val_preds = torch.tensor(val_preds[-1])  # 取最后一个时间步的预测
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Loss: {val_loss.item() / 96}, Validation Accuracy: {val_acc}")

# 测试模型
model.eval()
test_loss = 0
test_preds = []
y_prev_test = torch.zeros(len(x_test), 10)
with torch.no_grad():
    for t in range(96):
        y_pred_test = model(x_test[:, t, :], y_prev_test)
        test_loss += criterion(y_pred_test, y_test)
        y_prev_test = y_pred_test.detach()
        test_preds.append(torch.argmax(y_pred_test, dim=1).cpu().numpy())
test_preds = torch.tensor(test_preds[-1])  # 取最后一个时间步的预测
test_acc = accuracy_score(y_test, test_preds)
print(f"Test Loss: {test_loss.item() / 96}, Test Accuracy: {test_acc}")

