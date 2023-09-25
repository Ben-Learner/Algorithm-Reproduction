import numpy as np
import skfuzzy as fuzz
from scipy.optimize import least_squares
from geneticalgorithm import geneticalgorithm as ga
from sklearn.preprocessing import OneHotEncoder

# 初始化参数
n_samples = 25
n_features = 96
n_time_steps = 96
n_rules = 3
n_classes = 10  # 10 分类
n_params_per_rule = n_features + 1  # a (n_features) + b (1)
epsilon = 1e-8 #增加数值稳定性

# 随机生成模拟数据
X_train = np.random.rand(n_samples, n_features, n_time_steps)
y_train = np.random.randint(0, n_classes, size=n_samples)

# One-hot 编码
encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

# 修改后的前向传播和误差计算函数，适用于T-S模型
def forward_and_error_TS(params, X, y_onehot):
    # 提取前件参数
    means = params[:n_rules * n_features].reshape(n_rules, n_features)
    stds = params[n_rules * n_features:2 * n_rules * n_features].reshape(n_rules, n_features)
    
    # 提取后件参数
    consequent_params = params[2 * n_rules * n_features:].reshape(n_rules, n_classes, n_params_per_rule) #3,10,97
    
    memberships = np.array([[fuzz.gaussmf(X[:, i, :], means[j, i], stds[j, i]) for i in range(n_features)] for j in range(n_rules)])
    rule_outputs = np.min(memberships, axis=1)
    
    # 归一化
    rule_outputs_normalized = rule_outputs / (np.sum(rule_outputs, axis=0) + epsilon)
    
    # 计算后件的输出
    # 这里，我们使用输入特征的加权和（线性函数）作为后件
    X_avg = np.mean(X, axis=2)  
    consequent_output = np.einsum('ijk,lk->ijl', consequent_params[:, :, :-1], X_avg) + consequent_params[:, :, -1][:, :, np.newaxis]
    
    # 加权后件的输出
    rule_outputs_normalized_reshaped = np.mean(rule_outputs_normalized, axis=-1)[:, np.newaxis, :] #3,1,1000 3,10,1000
    weighted_output = np.sum(rule_outputs_normalized_reshaped * consequent_output, axis=0).T
    
    # Softmax激活
    output_exp = np.exp(weighted_output)
    output_softmax = output_exp / (np.sum(output_exp, axis=1, keepdims=True) + epsilon)

    # 计算交叉熵损失
    error = -np.mean(np.sum(y_onehot * np.log(output_softmax + epsilon), axis=1))

    # output_softmax = output_exp / np.sum(output_exp, axis=1, keepdims=True)
    
    
    
    # error = -np.mean(np.sum(y_onehot * np.log(output_softmax), axis=1))
    print(error)
    return output_softmax, error

# 预测函数
def predict_TS(params, X, y_train_onehot):
    output_softmax, _ = forward_and_error_TS(params, X, y_train_onehot)
    return np.argmax(output_softmax, axis=1)

# 使用遗传算法优化参数的适应度函数
def fitness_function_TS(params):
    _, error = forward_and_error_TS(params, X_train, y_train_onehot)
    return error

# 遗传算法参数边界
varbound_TS = np.array([[0, 1]] * (2 * n_rules * n_features + n_rules * n_classes * n_params_per_rule))

# 遗传算法参数设置
algorithm_param = {
    'max_num_iteration': 1000,
    'population_size': 10,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

# 运行遗传算法
model_TS = ga(function=fitness_function_TS, dimension=2 * n_rules * n_features + n_rules * n_classes * n_params_per_rule, 
             variable_type='real', variable_boundaries=varbound_TS, algorithm_parameters=algorithm_param)
model_TS.run()

# 获取遗传算法优化后的参数
optimized_params_TS = model_TS.output_dict['variable']

# 使用最小二乘法优化后件参数
result_TS = least_squares(lambda params: forward_and_error_TS(params, X_train, y_train_onehot)[1], optimized_params_TS, max_nfev=10)
optimized_params_ls_TS = result_TS.x

# 输出优化后的参数
print("Optimized Parameters (T-S Model):", optimized_params_ls_TS)

# 计算训练集上的准确率(这里应该替换成测试集合)
y_train_pred_TS = predict_TS(optimized_params_ls_TS, X_train, y_train_onehot)
accuracy_train_TS = np.mean(y_train_pred_TS == y_train) * 100  # 将准确率转换为百分比
print(f"Training Accuracy (T-S Model): {accuracy_train_TS:.2f}%")
