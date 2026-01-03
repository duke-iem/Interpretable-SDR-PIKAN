"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2024/10/26
"""
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from torch.optim import LBFGS  # L-BFGS优化算法
from ..config import DEVICE

class TrainerInverseProblem:
    def __init__(self, net, loss_functions, Adamsteps, LBFGSsteps, loss_weight=None, HardBC=False):
        if HardBC:
            if loss_weight is None:
                loss_weight = [1.0, 10000.0]
        else:
            if loss_weight is None:
                loss_weight = [1.0, 1.0, 1.0, 10000.0]  # 默认使用四个损失权重

        self.HardBC = HardBC
        self.lossweight = loss_weight
        self.net = net
        self.loss_functions = loss_functions
        self.Adamsteps = Adamsteps
        self.LBFGSsteps = LBFGSsteps
        self.duration_time = loss_functions.gettime()
        self.trainable_params = loss_functions.get_trainable_params()
        self.trainable_params_name = loss_functions.get_trainable_params_name()
        self.dimensionsless_unit = loss_functions.get_dimensionsless_unit()
        self.dimensionsless_Pt = loss_functions.get_dimensionsless_Pt()
        self.total_training_time = 0  # 记录总训练时间
        self.losses = []   # 定义损失记录列表
        self.train_params = [] # 定义训练参数记录列表
        self.results = {}

    def train_with_adam(self, init_lr=0.001, scheduler_type='StepLR', step_size=1000, gamma=1.0,T_max=200, factor=0.1, patience=10, show_epochs=100):
        opt = torch.optim.Adam(params=list(self.net.parameters()) + self.trainable_params, lr=init_lr)
        # 选择学习率调度器
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max)
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=factor, patience=patience)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        # 使用Adam训练
        print("Start training......")
        print(f"Start Adam training for {self.Adamsteps} steps")
        if self.HardBC:
            print("Step    ", "   Training loss(lossODE,lossObserve)       ", "Training total loss")
        else:
            print("Step    ", "   Training loss(lossODE,lossInitialU,lossInitialU_dot,lossObserve)       ", "Training total loss")

        # 开始计时
        start_time = time.time()
        for i in range(self.Adamsteps):
            opt.zero_grad()
            if self.HardBC:
                totalloss = self.lossweight[0] * self.loss_functions.lossODE() + \
                            self.lossweight[1] * self.loss_functions.lossObserve()
            else:
                totalloss = self.lossweight[0] * self.loss_functions.lossODE() + \
                            self.lossweight[1] * self.loss_functions.lossInitialU() + \
                            self.lossweight[2] * self.loss_functions.lossInitialU_dot() + \
                            self.lossweight[3] * self.loss_functions.lossObserve()
            totalloss.backward()
            opt.step()
            scheduler.step()
            self.losses.append(totalloss.item())
            for name,params in zip(self.trainable_params_name, self.trainable_params):
                params = params * self.dimensionsless_unit
                # 将转换后的结果添加到 train_params 列表中
                self.train_params.append({
                    name: params.cpu().detach().numpy()  # 将张量转移到 CPU，detach 并转换为 NumPy 数组
                })
            plt.ion()  # 开启交互模式
            if i % show_epochs == 0:
                if self.HardBC:
                    print(i, "   ", [self.loss_functions.lossODE().item(), self.loss_functions.lossObserve().item()], "   ",[totalloss.item()])
                else:
                    print(i, "   ", [self.loss_functions.lossODE().item(),
                                         self.loss_functions.lossInitialU().item(),
                                         self.loss_functions.lossInitialU_dot().item(),
                                         self.loss_functions.lossObserve().item()], "   ",
                          [totalloss.item()])
                for name, params in zip(self.trainable_params_name, self.trainable_params):
                    params = params * self.dimensionsless_unit
                    # 使用 f-string 进行格式化输出
                    print(f"Updated {name}: {params.cpu().detach().numpy()}")
                t_test = torch.linspace(0, self.duration_time, 2000, device=DEVICE).unsqueeze(1)  # 设置测试时间
                with torch.no_grad():
                    u_pred = self.net(t_test)* self.dimensionsless_Pt   # 计算预测位移
                    plt.cla()
                    for j in range(u_pred.shape[1]):  # 遍历每组数据
                        plt.plot(t_test.cpu().numpy(), u_pred.detach().cpu()[:, j], label=f'DOF {j + 1}', linewidth=3)
                    plt.legend()
                    plt.xlabel('t_test')
                    plt.ylabel('u_pred')
                    plt.title('Predicted Displacement for All DOFs')
                    plt.show()
                    plt.pause(0.1)
        end_time = time.time()
        adam_training_time = end_time - start_time
        self.total_training_time += adam_training_time  # 更新总训练时间
    def train_with_lbfgs(self, max_iter=1000, tolerance_grad=0, tolerance_change=0, line_search_fn='strong_wolfe',show_epochs=100):
        lbfgs_opt = LBFGS(params=list(self.net.parameters()) + self.trainable_params, max_iter=max_iter, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, line_search_fn=line_search_fn)
        lbfgsiteration = 0
        start_time = time.time()

        def closure():
            nonlocal lbfgsiteration
            lbfgs_opt.zero_grad()
            if self.HardBC:
                totalloss = self.lossweight[0] * self.loss_functions.lossODE() + \
                            self.lossweight[1] * self.loss_functions.lossObserve()
            else:
                totalloss = self.lossweight[0] * self.loss_functions.lossODE() + \
                            self.lossweight[1] * self.loss_functions.lossInitialU() + \
                            self.lossweight[2] * self.loss_functions.lossInitialU_dot() + \
                            self.lossweight[3] * self.loss_functions.lossObserve()
            totalloss.backward()
            self.losses.append(totalloss.item())
            for name,params in zip(self.trainable_params_name, self.trainable_params):
                params = params * self.dimensionsless_unit
                # 将转换后的结果添加到 train_params 列表中
                self.train_params.append({
                    name: params.cpu().detach().numpy()  # 将张量转移到 CPU，detach 并转换为 NumPy 数组
                })

            if lbfgsiteration % show_epochs == 0:
                if self.HardBC:
                    print(lbfgsiteration, "   ", [self.loss_functions.lossODE().item(), self.loss_functions.lossObserve().item()], "   ",[totalloss.item()])
                else:
                    print(lbfgsiteration, "   ", [self.loss_functions.lossODE().item(),
                                         self.loss_functions.lossInitialU().item(),
                                         self.loss_functions.lossInitialU_dot().item(),
                                         self.loss_functions.lossObserve().item()], "   ",
                          [totalloss.item()])
                for name, params_dimensionsless in zip(self.trainable_params_name, self.trainable_params):
                    # 使用 f-string 进行格式化输出
                    print(f"Updated {name}: {(params_dimensionsless * self.dimensionsless_unit).cpu().detach().numpy()}")
                t_test = torch.linspace(0, self.duration_time, 2000, device=DEVICE).unsqueeze(1)
                with torch.no_grad():
                    u_pred = self.net(t_test)* self.dimensionsless_Pt
                    plt.cla()
                    for i in range(u_pred.shape[1]):
                        plt.plot(t_test.cpu().numpy(), u_pred.cpu().numpy()[:, i], label=f'DOF {i + 1}', linewidth=3)
                    plt.legend()
                    plt.xlabel('t_test')
                    plt.ylabel('u_pred')
                    plt.title('Predicted Displacement for All DOFs')
                    plt.show()
                    plt.pause(0.1)

            lbfgsiteration += 1  # 更新迭代次数

            if lbfgsiteration >= self.LBFGSsteps:  # 检查是否达到最大迭代次数
                raise StopIteration  # 抛出异常以停止优化

            return totalloss

        print(f"Start L-BFGS training for {self.LBFGSsteps} steps")
        try:
            while True:
                lbfgs_opt.step(closure)
        except StopIteration:
            print("Training finished")
            # 结束计时
            end_time = time.time()
            lbfgs_training_time = end_time - start_time
            self.total_training_time += lbfgs_training_time  # 更新总训练时间
            print(f"Total training time: {self.total_training_time} seconds")
            # 保存训练结果，将模型和标量存储在一个字典中
            self.results = {
                'net': self.net,
                'dimensionsless_Pt': self.dimensionsless_Pt,
                'duration_time': self.duration_time,
            }

            # 保存更新参数值
            # 打开文件写入数据
            with open('train_params_matrix.txt', 'w') as f:
                # 写入文件的表头，根据 self.trainable_params_name 动态生成表头
                header = '\t'.join(self.trainable_params_name) + '\n'
                f.write(header)

                # 获取每组项的数量
                group_size = len(self.trainable_params_name)

                # 遍历参数列表，每 group_size 项为一组
                for i in range(0, len(self.train_params), group_size):
                    param_strings = []

                    # 确保每组内所有参数都有对应的数据
                    for j in range(group_size):
                        # 检查索引是否越界
                        if i + j < len(self.train_params):
                            # 获取第 i+j 项的所有参数
                            params = self.train_params[i + j]
                            for param_name in self.trainable_params_name:
                                matrix = params.get(param_name)
                                # 格式化矩阵为字符串，如果参数缺失则填充空字符串
                                matrix_str = np.array2string(matrix, separator=',', formatter={
                                    'all': lambda x: f'{x:.8f}'}) if matrix is not None else ''
                                param_strings.append(matrix_str)
                        else:
                            # 如果该项缺失，填充空字符串以保持列对齐
                            param_strings.extend([''] * len(self.trainable_params_name))

                    # 写入当前组的所有参数字符串
                    f.write('\t'.join(param_strings) + '\n')

            # 输出提示信息
            print("Data saved to train_params_matrix.txt")



