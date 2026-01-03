from kan import *
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import numpy as np

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

time = 12.15

# create dataset
df  = np.loadtxt("example-3.txt", skiprows=1)
data_input = df[:, 0].tolist()  # 提取时间数据
data_label = df[:, 1].tolist()  # 提取位移结果数据

# 将数据转换为 PyTorch 张量
input_tensor = torch.tensor(data_input).view(-1, 1)  # 使得数据成为 n x 1 的形状
label_tensor = torch.tensor(data_label).view(-1, 1)

# 使用 train_test_split 将数据分割为训练集和测试集，test_size 控制测试集的比例
train_input, test_input, train_label, test_label = train_test_split(input_tensor, label_tensor, test_size=0.5, random_state=42)

# 将结果打包为字典
dataset = {
    'train_input': train_input,
    'test_input': test_input,
    'train_label': train_label,
    'test_label': test_label
}

model = KAN(width=[1, 40, 1], grid=int(5 * time), k=3, seed=0, noise_scale=0.001, grid_range=[0, time])

# plot KAN at initialization
model(dataset['train_input']);
model.plot()



mode = "manual"  # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0, 0, 0, 'sin', a_range=(17.1, 17.3));
    model.fix_symbolic(0, 0, 1, 'sin', a_range=(17.4, 17.6));
    model.fix_symbolic(0, 0, 2, 'sin', a_range=(17.7, 17.9));
    model.fix_symbolic(0, 0, 3, 'sin', a_range=(16.8, 17.0));
    model.fix_symbolic(0, 0, 4, 'sin', a_range=(15.9, 16.0));
    model.fix_symbolic(0, 0, 5, 'sin', a_range=(15.6, 15.7));
    model.fix_symbolic(0, 0, 6, 'sin', a_range=(15.2, 15.4));
    model.fix_symbolic(0, 0, 7, 'sin', a_range=(14.9, 15.1));
    model.fix_symbolic(0, 0, 8, 'sin', a_range=(15.6, 15.7));
    model.fix_symbolic(0, 0, 9, 'sin', a_range=(16.2, 16.3));
    model.fix_symbolic(0, 0, 10, 'sin', a_range=(16.5, 16.7));
    model.fix_symbolic(0, 0, 11, 'sin', a_range=(16.2, 16.3));
    model.fix_symbolic(0, 0, 12, 'sin', a_range=(14.6, 14.8));
    model.fix_symbolic(0, 0, 13, 'sin', a_range=(14.0, 14.2));
    model.fix_symbolic(0, 0, 14, 'sin', a_range=(8.7, 9.0));
    model.fix_symbolic(0, 0, 15, 'sin', a_range=(7.6, 7.8));
    model.fix_symbolic(0, 0, 16, 'sin', a_range=(6.1, 6.2));
    model.fix_symbolic(0, 0, 17, 'sin', a_range=(18.0, 18.2));
    model.fix_symbolic(0, 0, 18, 'sin', a_range=(18.3, 18.5));
    model.fix_symbolic(0, 0, 19, 'sin', a_range=(19.6, 19.7));
    model.fix_symbolic(0, 0, 20, 'cos', a_range=(17.1, 17.3));
    model.fix_symbolic(0, 0, 21, 'cos', a_range=(17.4, 17.6));
    model.fix_symbolic(0, 0, 22, 'cos', a_range=(17.7, 17.9));
    model.fix_symbolic(0, 0, 23, 'cos', a_range=(16.8, 17.0));
    model.fix_symbolic(0, 0, 24, 'cos', a_range=(15.9, 16.0));
    model.fix_symbolic(0, 0, 25, 'cos', a_range=(15.6, 15.7));
    model.fix_symbolic(0, 0, 26, 'cos', a_range=(15.2, 15.4));
    model.fix_symbolic(0, 0, 27, 'cos', a_range=(14.9, 15.1));
    model.fix_symbolic(0, 0, 28, 'cos', a_range=(15.6, 15.7));
    model.fix_symbolic(0, 0, 29, 'cos', a_range=(16.2, 16.3));
    model.fix_symbolic(0, 0, 30, 'cos', a_range=(16.5, 16.7));
    model.fix_symbolic(0, 0, 31, 'cos', a_range=(16.2, 16.3));
    model.fix_symbolic(0, 0, 32, 'cos', a_range=(14.6, 14.8));
    model.fix_symbolic(0, 0, 33, 'cos', a_range=(14.0, 14.2));
    model.fix_symbolic(0, 0, 34, 'cos', a_range=(8.7, 9.0));
    model.fix_symbolic(0, 0, 35, 'cos', a_range=(7.6, 7.8));
    model.fix_symbolic(0, 0, 36, 'cos', a_range=(6.1, 6.2));
    model.fix_symbolic(0, 0, 37, 'cos', a_range=(18.0, 18.2));
    model.fix_symbolic(0, 0, 38, 'cos', a_range=(18.3, 18.5));
    model.fix_symbolic(0, 0, 39, 'cos', a_range=(19.6, 19.7));


    model.fix_symbolic(1, 0, 0, 'x');
    model.fix_symbolic(1, 1, 0, 'x');
    model.fix_symbolic(1, 2, 0, 'x');
    model.fix_symbolic(1, 3, 0, 'x');
    model.fix_symbolic(1, 4, 0, 'x');
    model.fix_symbolic(1, 5, 0, 'x');
    model.fix_symbolic(1, 6, 0, 'x');
    model.fix_symbolic(1, 7, 0, 'x');
    model.fix_symbolic(1, 8, 0, 'x');
    model.fix_symbolic(1, 9, 0, 'x');
    model.fix_symbolic(1, 10, 0, 'x');
    model.fix_symbolic(1, 11, 0, 'x');
    model.fix_symbolic(1, 12, 0, 'x');
    model.fix_symbolic(1, 13, 0, 'x');
    model.fix_symbolic(1, 14, 0, 'x');
    model.fix_symbolic(1, 15, 0, 'x');
    model.fix_symbolic(1, 16, 0, 'x');
    model.fix_symbolic(1, 17, 0, 'x');
    model.fix_symbolic(1, 18, 0, 'x');
    model.fix_symbolic(1, 19, 0, 'x');
    model.fix_symbolic(1, 20, 0, 'x');
    model.fix_symbolic(1, 21, 0, 'x');
    model.fix_symbolic(1, 22, 0, 'x');
    model.fix_symbolic(1, 23, 0, 'x');
    model.fix_symbolic(1, 24, 0, 'x');
    model.fix_symbolic(1, 25, 0, 'x');
    model.fix_symbolic(1, 26, 0, 'x');
    model.fix_symbolic(1, 27, 0, 'x');
    model.fix_symbolic(1, 28, 0, 'x');
    model.fix_symbolic(1, 29, 0, 'x');
    model.fix_symbolic(1, 30, 0, 'x');
    model.fix_symbolic(1, 31, 0, 'x');
    model.fix_symbolic(1, 32, 0, 'x');
    model.fix_symbolic(1, 33, 0, 'x');
    model.fix_symbolic(1, 34, 0, 'x');
    model.fix_symbolic(1, 35, 0, 'x');
    model.fix_symbolic(1, 36, 0, 'x');
    model.fix_symbolic(1, 37, 0, 'x');
    model.fix_symbolic(1, 38, 0, 'x');
    model.fix_symbolic(1, 39, 0, 'x');


elif mode == "auto":
    #automatic mode
    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'abs']  # 定义了一个列表 lib，其中包含了一系列的数学符号和函数，包括 x、x^2、x^3、x^4、exp（指数函数）、log（对数函数）、sqrt（平方根）、tanh（双曲正切）、sin（正弦函数）和 abs（绝对值）
    model.auto_symbolic()

model.fit(dataset, opt="LBFGS", steps=300);
model.plot()

# 可视化结构位移
t_test = torch.linspace(0, time, 2000).unsqueeze(1)
with torch.no_grad():
    u_pred = model(t_test)

# 读取excel表格数据
data = pd.read_excel('Interpretability-3.xlsx',sheet_name='Sheet1',header=1)
x = data.iloc[:, 0].values
y = data.iloc[:,1].values


# 绘制图形
plt.figure()
plt.plot(t_test.numpy(), u_pred.numpy(), label='PINN', linestyle='-')
plt.plot( x,y, label='Truth', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement_Time')
plt.grid(True)
plt.legend()
plt.show()

latex_expression = latex(model.symbolic_formula()[0][0])
print(latex_expression)  # 输出 LaTeX 格式