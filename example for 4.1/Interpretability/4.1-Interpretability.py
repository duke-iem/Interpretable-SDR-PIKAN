from kan import *
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import numpy as np

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

time = 10

# create dataset
df  = np.loadtxt("example-1.txt", skiprows=1)
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

model = KAN(width=[1, 6, 1], grid=int(5 * time), k=3, seed=0, noise_scale=0.001, grid_range=[0, time])

# plot KAN at initialization
model(dataset['train_input']);
model.plot()



mode = "manual"  # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0, 0, 0, 'sin', a_range=(0.6, 1.0));
    model.fix_symbolic(0, 0, 1, 'sin', a_range=(1.4, 1.6));
    model.fix_symbolic(0, 0, 2, 'sin', a_range=(2.0, 2.5));
    model.fix_symbolic(0, 0, 3, 'cos', a_range=(0.6, 1.0));
    model.fix_symbolic(0, 0, 4, 'cos', a_range=(1.4, 1.6));
    model.fix_symbolic(0, 0, 5, 'cos', a_range=(2.0, 2.5));


    model.fix_symbolic(1, 0, 0, 'x');
    model.fix_symbolic(1, 1, 0, 'x');
    model.fix_symbolic(1, 2, 0, 'x');
    model.fix_symbolic(1, 3, 0, 'x');
    model.fix_symbolic(1, 4, 0, 'x');
    model.fix_symbolic(1, 5, 0, 'x');


elif mode == "auto":
    #automatic mode
    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'abs']  # 定义了一个列表 lib，其中包含了一系列的数学符号和函数，包括 x、x^2、x^3、x^4、exp（指数函数）、log（对数函数）、sqrt（平方根）、tanh（双曲正切）、sin（正弦函数）和 abs（绝对值）
    model.auto_symbolic()

model.fit(dataset, opt="LBFGS", steps=50);
model.plot()

# 可视化结构位移
t_test = torch.linspace(0, time, 2000).unsqueeze(1)
with torch.no_grad():
    u_pred = model(t_test)

# 读取excel表格数据
data = pd.read_excel('Interpretability-1.xlsx',sheet_name='Sheet1',header=1)
x = data.iloc[:, 0].values
y = data.iloc[:,1].values


# 绘制图形
plt.figure()
plt.plot(t_test.numpy(), u_pred.numpy(), label='PIKAN', linestyle='-')
plt.plot( x,y, label='Truth', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement_Time')
plt.grid(True)
plt.legend()
plt.show()

latex_expression = latex(model.symbolic_formula()[0][0])
print(latex_expression)  # 输出 LaTeX 格式