"""
Quantum Recurrent Embedding Neural Networks (QRENN)
====================================================
A QuAIRKit implementation for the QRENN model as described in:
"Quantum Recurrent Embedding Neural Networks" - Jing et al.

This module provides a flexible and modular implementation of QRENN
for quantum supervised learning tasks.

Author: Mingrui
"""

import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

import torch

from quairkit import Circuit, State
from quairkit.database import *
from quairkit.qinfo import dagger


class QRENN(torch.nn.Module):
    """
    Quantum Recurrent Neural Network Circuit.

    该类作为 QRENN 架构的封装，内部维护一个 Circuit 实例。
    每次设置新数据时，会重新创建完整的电路以确保梯度正确传播。

    电路结构:

        可训练量子比特:  ─[U_train]─●─[U_train]─●─ ... ─[U_train]─
                                    │           │
        控制量子比特:    ─[H]──────[g(U)]─────[g(U†)]── ... ─────

    注意：
        - 数据生成器支持 batch 输入（3-tensor，第一维为 batch）
        - 可训练参数在多次 set_data 调用间保持

    Example:
        >>> qrenn = QRENNCircuit(
        ...     num_trainable=2,
        ...     num_control=2,
        ...     num_layers=3
        ... )
        >>> output = qrenn(state, data_generator=U)
        >>>
        >>> # 或分步操作
        >>> qrenn.set_data(U)
        >>> output = qrenn(state)
    """

    def __init__(
        self,
        num_trainable: int,
        num_control: int,
        num_layers: int,
        input_state: Optional[State] = None,
        use_conjugate: bool = False,
        system_dim: Union[int, List[int]] = 2,
    ):
        """
        初始化 QRENN 电路。

        Args:
            num_trainable: 可训练量子比特数量
            num_control: 控制量子比特数量
            num_layers: 电路层数
            use_conjugate: 是否在奇数层使用 U† 代替 U
            system_dim: 量子系统维度（默认为 qubit，即 2）

        Raises:
            ValueError: 如果参数值无效
        """
        # 验证输入参数
        if num_trainable <= 0:
            raise ValueError(f"num_trainable must be positive, got {num_trainable}")
        if num_control <= 0:
            raise ValueError(f"num_control must be positive, got {num_control}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")

        super().__init__()

        # 存储配置
        self._num_trainable = num_trainable
        self._num_control = num_control
        self._num_layers = num_layers
        self._input_state = input_state
        self._use_conjugate = use_conjugate
        self._system_dim = system_dim
        self._total_qubits = num_trainable + num_control

        # 预计算量子比特索引
        self._trainable_idx = list(range(num_trainable))
        self._control_idx = list(range(num_trainable, self._total_qubits))
        self._oracle_system_idx = [self._trainable_idx, self._control_idx]

        # 内部电路（初始时不含数据）
        self._circuit: Optional[Circuit] = None
        self._data_generator: Optional[torch.Tensor] = None

    def _build_circuit(self, data_generator: torch.Tensor) -> Circuit:
        """
        根据数据生成器构建完整电路。

        Args:
            data_generator: 数据编码的酉矩阵，支持 batch (3-tensor)

        Returns:
            构建好的 Circuit 实例

        Raises:
            ValueError: 如果 data_generator 形状不兼容
        """
        # 验证数据生成器的形状
        self._validate_data_generator(data_generator)

        circuit = Circuit(num_qubits=self._total_qubits, system_dim=self._system_dim)

        # 预计算共轭（如果需要）
        if self._use_conjugate:
            data_generator_dagger = dagger(data_generator)

        # 初始层：对控制量子比特应用 Hadamard
        circuit.universal_qudits(qubits_idx=self._control_idx)
        # circuit.x(qubits_idx=[3])

        # 处理层
        for layer_idx in range(self._num_layers):
            # 可训练酉算子
            circuit.universal_qudits(qubits_idx=self._trainable_idx)

            # 确定使用 U 还是 U†
            if self._use_conjugate and layer_idx % 2 == 1:
                oracle_matrix = data_generator_dagger
                gate_name = f'g(U^dag_{layer_idx})'
                latex_name = r'g(U^{\dagger})'
            else:
                oracle_matrix = data_generator
                gate_name = f'g(U_{layer_idx})'
                latex_name = r'g(U)'

            # 添加 controlled oracle
            circuit.oracle(
                oracle_matrix,
                qubits_idx=[self._trainable_idx] + self._control_idx,
                control_idx=2**self._num_trainable-1,
                gate_name=gate_name,
                latex_name=latex_name
            )

        # 最终可训练层
        circuit.universal_qudits(qubits_idx=self._trainable_idx)

        return circuit

    def _validate_data_generator(self, data_generator: torch.Tensor) -> None:
        """
        验证数据生成器的形状和类型。

        Args:
            data_generator: 数据编码的酉矩阵

        Raises:
            ValueError: 如果数据生成器形状或类型不兼容
        """
        if not isinstance(data_generator, torch.Tensor):
            raise TypeError(f"data_generator must be torch.Tensor, got {type(data_generator)}")

        # 检查形状：支持 2D (matrix) 或 3D (batch)
        if data_generator.dim() not in [2, 3]:
            raise ValueError(
                f"data_generator must be 2D (matrix) or 3D (batch), got {data_generator.dim()}D"
            )

        # 获取矩阵维度
        if data_generator.dim() == 2:
            matrix_shape = data_generator.shape
        else:  # 3D
            matrix_shape = data_generator.shape[1:]

        # 验证矩阵是方阵
        if matrix_shape[0] != matrix_shape[1]:
            raise ValueError(
                f"data_generator must be square matrices, got shape {matrix_shape}"
            )

        # 验证矩阵维度与控制量子比特数量匹配
        control_dim = 2 ** self._num_control if self._system_dim == 2 else self._system_dim
        if matrix_shape[0] != control_dim:
            warnings.warn(
                f"data_generator dimension {matrix_shape[0]} may not match "
                f"control qubit dimension {control_dim}. This may cause issues."
            )
    
    # =========================================================================
    # 属性访问器
    # =========================================================================
    
    @property
    def num_trainable(self) -> int:
        return self._num_trainable
    
    @property
    def num_control(self) -> int:
        return self._num_control
    
    @property
    def num_layers(self) -> int:
        return self._num_layers
    
    @property
    def num_qubits(self) -> int:
        return self._total_qubits
    
    @property
    def use_conjugate(self) -> bool:
        return self._use_conjugate
    
    @property
    def trainable_indices(self) -> List[int]:
        return self._trainable_idx.copy()
    
    @property
    def control_indices(self) -> List[int]:
        return self._control_idx.copy()
    
    @property
    def data_generator(self) -> Optional[torch.Tensor]:
        return self._data_generator
    
    @property
    def is_data_set(self) -> bool:
        return self._data_generator is not None
    
    @property
    def circuit(self) -> Optional[Circuit]:
        """获取当前内部电路（如果已构建）"""
        return self._circuit
    
    # =========================================================================
    # 数据设置与前向传播
    # =========================================================================
    
    def set_data(self, data_generator: torch.Tensor) -> 'QRENNCircuit':
        """
        设置数据生成器并构建电路。

        Args:
            data_generator: 数据编码的酉矩阵，支持 batch (3-tensor)

        Returns:
            self（支持链式调用）

        Raises:
            ValueError: 如果数据生成器无效
        """
        self._validate_data_generator(data_generator)
        self._data_generator = data_generator
        self._circuit = self._build_circuit(data_generator)
        return self
    
    def forward(
        self,
        state: Optional[State] = None,
        data_generator: Optional[torch.Tensor] = None,
    ) -> State:
        """
        前向传播：将输入态通过 QRENN 电路。

        Args:
            state: 输入量子态。如果为 None，则使用全零态。
            data_generator: 数据编码酉矩阵（可选）。
                           如果提供，会重建电路。

        Returns:
            输出量子态

        Raises:
            ValueError: 如果未设置数据生成器
        """
        if data_generator is not None:
            self.set_data(data_generator)

        if self._circuit is None:
            raise ValueError(
                "数据生成器未设置。请通过 set_data() 或 forward(data_generator=...) 设置。"
            )
        
        if state is not None:
            return self._circuit(state)

        return self._circuit(self._input_state)
    
    # =========================================================================
    # 工具方法
    # =========================================================================
    
    def plot(self, style: str = 'standard', 
             decimal: int = 2,
             dpi: int = 300, 
             print_code: bool = False, 
             show_plot: bool = True, 
             include_empty: bool = False,
             latex: bool = True, 
             **kwargs) -> None:
        """绘制当前电路（如果已构建）。"""
        if self._circuit is None:
            print("电路未构建。请先设置数据生成器。")
        else:
            self._circuit.plot(style=style, 
                               decimal=decimal,
                               dpi=dpi, 
                               print_code=print_code, 
                               show_plot=show_plot, 
                               include_empty=include_empty,
                               latex=latex, **kwargs)

    def get_info(self) -> Dict[str, Any]:
        """获取电路配置信息。"""
        return {
            'num_trainable': self._num_trainable,
            'num_control': self._num_control,
            'num_layers': self._num_layers,
            'use_conjugate': self._use_conjugate,
            'total_qubits': self._total_qubits,
            'is_data_set': self.is_data_set,
            'system_dim': self._system_dim,
        }

    def num_parameters(self, only_trainable: bool = True) -> int:
        """
        获取参数数量。

        Args:
            only_trainable: 是否只计算可训练参数

        Returns:
            参数数量
        """
        return sum(p.numel() for p in self.parameters() if not only_trainable or p.requires_grad)

    def reset_data(self) -> None:
        """清除数据生成器并重置电路。"""
        self._data_generator = None
        self._circuit = None

    def to(self, device: Union[str, torch.device]) -> 'QRENNCircuit':
        """
        将模型移动到指定设备。

        Args:
            device: 目标设备

        Returns:
            self（支持链式调用）
        """
        super().to(device)
        return self
    
    def __repr__(self) -> str:
        data_status = "set" if self.is_data_set else "not set"
        return (
            f"QRENNCircuit(num_trainable={self._num_trainable}, "
            f"num_control={self._num_control}, "
            f"num_layers={self._num_layers}, "
            f"use_conjugate={self._use_conjugate}, "
            f"data={data_status})"
        )

# =============================================================================
# Main Entry Point for Testing
# =============================================================================

# if __name__ == "__main__":
#     pass