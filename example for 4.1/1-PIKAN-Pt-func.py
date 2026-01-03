"""
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt with initial condition: U(t=0)=0, and U_dot(t=0)=0
Pt is a continuous function
Author: 杜轲 duke@iem.ac.cn
Date: 2024/11/3
"""
###########################环境配置##################
# 导入所需的库
from AIStructDynSolve import *
from AIStructDynSolve.config import DEVICE
import numpy as np
import torch
################ 输入结构动力学参数 #################
# 定义质量矩阵
mass = np.array([40000,40000,40000,40000,40000,40000,40000,40000,40000,40000])  # 质量数组
M = np.diag(mass)  # 形成对角矩阵

# 定义刚度矩阵
K = np.array([
    [11040000,-5520000,0,0,0,0,0,0,0,0],
    [-5520000,11040000,-5520000,0,0,0,0,0,0,0],
    [0,-5520000,11040000,-5520000,0,0,0,0,0,0],
    [0,0,-5520000,11040000,-5520000,0,0,0,0,0],
    [0,0,0,-5520000,11040000,-5520000,0,0,0,0],
    [0,0,0,0,-5520000,11040000,-5520000,0,0,0],
    [0,0,0,0,0,-5520000,11040000,-5520000,0,0],
    [0,0,0,0,0,0,-5520000,11040000,-5520000,0],
    [0,0,0,0,0,0,0,-5520000,11040000,-5520000],
    [0,0,0,0,0,0,0,0,-5520000,5520000]
])  # 刚度矩阵

# 定义阻尼矩阵
C =0.15*M+0.00609*K
#C = np.array([
    #[1.884881, -0.93423, 0, 0, 0, 0, 0, 0, 0, 0],
    #[--0.93423, 1.884881, -0.93423, 0, 0, 0, 0, 0, 0, 0],
    #[0, -0.93423, 1.884881, -0.93423, 0, 0, 0, 0, 0, 0],
    #[0, 0, -0.93423, 1.884881, -0.93423, 0, 0, 0, 0, 0],
    #[0, 0, 0, -0.93423, 1.884881, -0.93423, 0, 0, 0, 0],
    #[0, 0, 0, 0, -0.93423, 1.884881, -0.93423, 0, 0, 0],
    #[0, 0, 0, 0, 0, -0.93423, 1.884881, -0.93423, 0, 0],
    #[0, 0, 0, 0, 0, 0, -0.93423, 1.884881, -0.93423, 0],
    #[0, 0, 0, 0, 0, 0, 0, -0.93423, 1.884881, -0.93423],
    #[0, 0, 0, 0, 0, 0, 0, 0, -0.93423, 0.950648]
#])  # 阻尼系数矩阵

DOF = M.shape[0]  # 计算自由度

# 模态分析
eigenvalues, eigenvectors = StructDynSystem.ModalAnalyzer(M, K)
# 计算圆频率
wn = np.sqrt(eigenvalues).tolist()  # 计算圆频率
print("natural circular frequency of structure ωn:", wn)  # 打印圆频率
wn0=wn[0] # 计算主频率
##########################################
# 设置时间范围
time = 10.0  # 总模拟时间（单位：秒）

# 定义荷载函数结构，其中 t 和Pt 都是张量
def Pt_func(t):
    Pt = [3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t), 3000*torch.sin(1.885*t)]
    #Pt =  [torch.sin(t)+torch.cos(t),torch.sin(t),...] #按照自由度顺序，列表里面给出每个自由度的外荷载Pt
    return Pt
################ 神经网络超参数设置 #################
# 定义超参数
Adamsteps = 10001  # Adam优化算法的训练步数
LBFGSsteps = 10001  # L-BFGS优化算法的训练步数
init_lr = 0.001  # 初始学习率
NumberODE = int(time * 90)  # 常微分方程（ODE）采样点数
NumberInitial = int(time * 10)  # 初始条件采样点数
#############################################

############################FKAN##############
layer_size = [1] + [60]  + [DOF]  # 输入层+隐藏层+输出层
net = NeuralNetwork.KAN(layer_size, grid_size=int(time)*5, grid_range=[0.0, time]).to(DEVICE)  # 采用KAN神经网络

# 结构动力系统参数
StructDynSystemParm = StructDynSystem.StructDynSystemParm(M, C, K)

# 结构动力系统中Pt荷载输入
inputPt = StructDynSystem.inputPtFunc(StructDynSystemParm, time, Pt_func)

# 结构动力系统的损失函数
loss_functions = LossFunctions.LossPtFunc(net, StructDynSystemParm, inputPt, NumberODE, NumberInitial)

# 创建训练器并训练
trainer = Trainer.TrainerForwardProblem(net, loss_functions, Adamsteps, LBFGSsteps)
trainer.train_with_adam(init_lr)  # 使用 Adam 训练
trainer.train_with_lbfgs()  # 使用 L-BFGS 训练

# 将模型训练结果保存到文件
torch.save(trainer.results, 'example1.pth')

# 可视化结果
visualizer = Postprocessor.Visualizer()
visualizer.plot_loss_curve(trainer.losses)
visualizer.plot_displacement(trainer.results)

