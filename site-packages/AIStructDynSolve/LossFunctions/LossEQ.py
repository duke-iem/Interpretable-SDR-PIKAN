"""
AIStructDynSolve
A framework focused on solving structural dynamics problems using artificial intelligence (AI) methods
solve the following ODE of MDOF
M*U_dotdot+C*U_dot+K*U=Pt
initial condition: U(t=0)=InitialU, and U_dot(t=0)=InitialU_dot
Author: 杜轲 duke@iem.ac.cn
Date: 2023/12/26
"""
import torch
import numpy as np
from .Gradients import Gradients
from skopt.sampler import Lhs, Halton, Hammersly, Sobol
from ..config import DEVICE, DEFAULT_DTYPE

class LossEQ:
    def __init__(
        self,
        net,
        struct_dyn_system,
        inputEQ,
        number_ode,
        number_initial,
        observe_data = None,  # Default is None
        train_params = None,  # Default is None
        loss_fn = None,  # Default is None, which will set to MSELoss if not provided
        sampling_method='Hammersley',  # Default to Hammersley, can be 'Lhs', 'Halton', 'Sobol','Uniform'
        InitialU = 0.0 ,  # Default is 0.0
        InitialU_dot = 0.0,  # Default is 0.0
    ):
        self.net = net
        # 从 StructDynSystem 类实例获取 M, C, K 矩阵
        mass, damping, stiffness = struct_dyn_system.get_matrices()
        # Check if 'stiffness' "mass" "damping" is in trainable_params and set as a torch parameter if true

        if train_params is not None and "mass" in {key.lower() for key in train_params}:
            self.mass = torch.nn.Parameter(mass, requires_grad=True)
        else:
            self.mass = mass

        if train_params is not None and "stiffness" in {key.lower() for key in train_params}:
            self.stiffness = torch.nn.Parameter(stiffness, requires_grad=True)
        else:
            self.stiffness = stiffness

        if train_params is not None and "damping" in {key.lower() for key in train_params}:
            self.damping = torch.nn.Parameter(damping, requires_grad=True)
        else:
            self.damping = damping

        self.dimensionsless_unit = struct_dyn_system.get_dimensionsless_unit()
        # 从 inputEQ 类实例获取 duration_time, eq_time, eq_acc参数
        self.duration_time, self.eq_time, self.eq_acc,self.dimensionsless_Pt = inputEQ.get_EQparm()

        self.DOF = self.mass.shape[0]  # 计算自由度
        self.number_ode = number_ode
        self.number_initial = number_initial
        self.observe_data = observe_data
        self.train_params = train_params
        self.sampling_method = sampling_method
        self.InitialU = InitialU
        self.InitialU_dot = InitialU_dot

        # Set the loss function, default is MSELoss
        # Example usage:
        # 选择不同的损失函数
        # loss_func = torch.nn.L1Loss(reduction='mean')  # 可以选择L1Loss
        # 或者可以选择其他损失函数，例如SmoothL1Loss, CrossEntropyLoss, 等
        if loss_fn is None:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss_fn = loss_fn


    def gettime(self):
        return self.duration_time

    def _get_sampler(self):
        """Returns the appropriate sampler based on the chosen sampling method."""
        samplers = {
            'Uniform': lambda: np.random.uniform(low=0.0, high=self.duration_time, size=(self.number_ode, 1)),
            'Lhs': Lhs,
            'Halton': Halton,
            'Sobol': Sobol,
            'Hammersley': Hammersly,
        }

        if self.sampling_method not in samplers:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}")

        sampler = samplers[self.sampling_method]
        if callable(sampler):
            return sampler()
        else:
            return sampler()

    def lossODE(self):
        # Get the appropriate sampler
        sampler = self._get_sampler()

        if self.sampling_method == 'Uniform':
            x_in = sampler  # For uniform, it's a direct numpy array
        else:
            # For other methods, generate samples and reshape them
            samples = sampler.generate(dimensions=[(0, self.duration_time)], n_samples=self.number_ode)
            x_in = np.array(samples).reshape(-1, 1)

        t_x_in = torch.from_numpy(x_in).to(DEFAULT_DTYPE).to(DEVICE).requires_grad_(True)
        u_collection = self.net(t_x_in)

        # Second and first derivatives for ODE calculation
        u_collection_dotdot = []
        u_collection_dot = []

        for i in range(self.DOF):
            u_collection_dotdot.append(Gradients.compute(u_collection[:, i], t_x_in, 2))
            u_collection_dot.append(Gradients.compute(u_collection[:, i], t_x_in, 1))

        u_collection_dotdot = torch.cat(u_collection_dotdot, dim=1).to(DEVICE)
        u_collection_dot = torch.cat(u_collection_dot, dim=1).to(DEVICE)

        # Interpolate external acceleration
        acc_interpolation = np.interp(x_in, self.eq_time, self.eq_acc/self.dimensionsless_Pt)
        acc = torch.from_numpy(acc_interpolation).repeat(1, self.DOF).to(DEFAULT_DTYPE).to(DEVICE)
        #Pt = -self.mass @ acc.T
        # 计算整体的残差
        #residual = self.mass @ u_collection_dotdot.T + self.damping @ u_collection_dot.T + self.stiffness @ u_collection.T - Pt
        residual = torch.matmul(self.mass, u_collection_dotdot.T) + torch.matmul(self.damping, u_collection_dot.T) + torch.matmul(self.stiffness, u_collection.T) - torch.matmul(-self.mass, acc.T)
        # 拆分为按行计算每个损失,使用torch.unbind()按行分解
        #residual_rows = torch.unbind(residual, dim=0)
        # 提取self.stiffness矩阵的对角元素
        #diagonal_values = torch.diag(self.stiffness)  # 获取对角线元素
        # 初始化总损失为0
        #total_loss = torch.tensor(0.0, device=DEVICE)
        # 对每一行计算损失并累加
        #for i, residual_row in enumerate(residual_rows):
        #    # 计算当前行的MSE损失
        #    row_loss = self.loss_fn(residual_row, torch.zeros_like(residual_row, device=DEVICE))
        #    # 将损失除以对应行的对角元素
        #    row_loss /= diagonal_values[i]
        #    # 累加到总损失
        #    total_loss += row_loss

        # 计算每行的MSE损失
        row_losses = torch.mean(residual ** 2, dim=1)
        # 提取self.stiffness矩阵的对角元素
        diagonal_values = torch.diag(self.stiffness)  # 获取对角线元素
        # 归一化行损失
        normalized_row_losses = row_losses / diagonal_values
        # 累加总损失
        total_loss = torch.sum(normalized_row_losses)

        # Calculate ODE loss
        #u_collectionT = u_collection.T
        #total_loss = self.loss_fn(
        #    self.mass @ u_collection_dotdot.T + self.damping @ u_collection_dot.T + self.stiffness @ u_collection.T - Pt,
        #    torch.zeros_like(u_collectionT, device=DEVICE)
        #)
        return total_loss

    ## 求边界条件的损失函数
    def lossInitialU(self):
        t_0 = torch.zeros(self.number_initial, 1, device=DEVICE).requires_grad_(True)  # 创建初始时间点
        u_0 = self.net(t_0)  # 神经网络输出
        loss_u = self.loss_fn(u_0.T, torch.zeros_like(u_0.T, device=DEVICE) + self.InitialU)
        return loss_u  # u(t=0)=0的损失

    def lossInitialU_dot(self):
        t_0 = torch.zeros(self.number_initial, 1, device=DEVICE).requires_grad_(True)  # 创建初始时间点
        u_0 = self.net(t_0)  # 神经网络输出

        # 计算u对x的一阶导数
        u_0_dot = [Gradients.compute(u_0[:, i], t_0, 1) for i in range(self.DOF)]
        u_0_dot = torch.cat(u_0_dot, dim=1).to(DEVICE)
        loss_u_dot = self.loss_fn(u_0_dot.T, torch.zeros_like(u_0.T, device=DEVICE) + self.InitialU_dot)

        return loss_u_dot  # u_dot(t=0)=0的损失

    ##求观测数据的损失
    def lossObserve(self):
        # 确保 observe_data 不为空
        if self.observe_data is not None:
            if not isinstance(self.observe_data[1], np.ndarray):
                # 如果self.observe_data[1]不是np.array矩阵
                observe_time = self.observe_data[1].to_numpy().reshape(-1, 1)  # 时间列
            else:
                observe_time = self.observe_data[1].reshape(-1, 1)  # 时间列
            observelist = self.observe_data[2]  # 位移或加速度观察值列表
            observelist_DOF = self.observe_data[3]  # 每列位移或加速度对应的自由度
            observe_t_tensor = torch.tensor(observe_time, dtype=DEFAULT_DTYPE, device=DEVICE)
            total_loss = 0  # 初始化总损失

            # 判断是位移（Disp）还是加速度（Acc）
            if self.observe_data[0] == "Disp":
                # 对于位移的计算
                for dof, disp_observe in zip(observelist_DOF, observelist):
                    # 获取神经网络输出对应自由度的位移
                    disp_tensor = self.net(observe_t_tensor)[:, dof - 1].unsqueeze(1)
                    if not isinstance(disp_observe, np.ndarray):
                        # 如果disp_observe不是np.array矩阵
                        disp_observe_tensor = torch.tensor(disp_observe.to_numpy().reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt
                    else:
                        disp_observe_tensor = torch.tensor(disp_observe.reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt

                    # 计算损失
                    loss_disp = self.loss_fn(disp_tensor, disp_observe_tensor)
                    # 累加损失
                    total_loss += loss_disp

            elif self.observe_data[0] == "Acc":
                # 对于加速度的计算
                for dof, acc_observe in zip(observelist_DOF, observelist):
                    # 计算加速度，这里假设 Gradients.compute 已经定义
                    acc_tensor = Gradients.compute(self.net(observe_t_tensor)[:, dof - 1], observe_t_tensor, 2)  # 二阶导数得到加速度
                    if not isinstance(acc_observe, np.ndarray):
                        # 如果disp_observe不是np.array矩阵
                        acc_observe_tensor = torch.tensor(acc_observe.to_numpy().reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt
                    else:
                        acc_observe_tensor = torch.tensor(acc_observe.reshape(-1, 1), dtype=DEFAULT_DTYPE, device=DEVICE)/self.dimensionsless_Pt

                    # 计算损失
                    loss_acc = self.loss_fn(acc_tensor, acc_observe_tensor)
                    # 累加损失
                    total_loss += loss_acc

            else:
                print("Observe data must be Disp or Acc")
                return None  # 如果观测数据既不是位移也不是加速度，则返回 None

        else:
            print("No observe_data provided.")
            return None  # 如果没有提供 observe_data，则返回 None

        # 返回总损失
        return total_loss


    def get_dimensionsless_unit(self):
        return self.dimensionsless_unit

    def get_dimensionsless_Pt(self):
        return self.dimensionsless_Pt
    def get_trainable_params_name(self):
        # 检查 self.train_params 是否为 None
        if self.train_params is None:
            raise ValueError("train_params is None. Please provide valid train_params.")
        return self.train_params

    def get_trainable_params(self):
        # 检查 self.train_params 是否为 None
        if self.train_params is None:
            raise ValueError("train_params is None. Please provide valid train_params.")
        # 初始化返回的参数列表
        params = []

        # 遍历 self.train_params，按照其顺序返回相应的属性
        for param in self.train_params:
            param_lower = param.lower()  # 将参数转换为小写，确保不区分大小写
            if param_lower == "mass":
                params.append(self.mass)
            elif param_lower == "stiffness":
                params.append(self.stiffness)
            elif param_lower == "damping":
                params.append(self.damping)

        return params
