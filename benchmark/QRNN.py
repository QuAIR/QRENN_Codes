import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 导入load_data函数
from load_data import load_data


# ============== 量子分类器 ==============
class QuantumBinaryClassifier:
    """基于对角编码和数据重上传的量子二分类器"""
    
    def __init__(self, n_qubits=2, n_data_layers=4, n_param_layers=5):
        """
        初始化量子分类器
        
        参数:
        -------
        n_qubits : int
            量子比特数 (特征数 = 2^n_qubits)
        n_data_layers : int
            数据重上传次数
        n_param_layers : int
            参数层数量 (通常比数据层多1)
        """
        self.n_qubits = n_qubits
        self.n_features = 2 ** n_qubits  # 对角编码需要的特征数
        self.n_data_layers = n_data_layers
        self.n_param_layers = n_param_layers
        
        # 创建量子设备
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 初始化权重和历史记录
        self.weights = None
        self.history = {"loss": [], "train_acc": [], "test_acc": []}
        
        print(f"\n{'='*70}")
        print("量子分类器初始化")
        print(f"{'='*70}")
        print(f"量子比特数: {n_qubits}")
        print(f"所需特征数: {self.n_features} (2^{n_qubits})")
        print(f"数据层数: {n_data_layers}")
        print(f"参数层数: {n_param_layers}")
        print(f"{'='*70}\n")
    
    def diagonal_encoding(self, x, wires):
        """
        对角编码: 将数据编码到对角酉矩阵
        diagonal = [exp(i*x[0]), exp(i*x[1]), exp(i*x[2]), exp(i*x[3])]
        """
        diagonal = np.exp(1j * x)
        qml.DiagonalQubitUnitary(diagonal, wires=wires)
    
    def variational_layer(self, weights, wires):
        """可训练的变分层: RX-RY-RZ + CNOT"""
        for i, wire in enumerate(wires):
            qml.RX(weights[i, 0], wires=wire)
            qml.RY(weights[i, 1], wires=wire)
            qml.RZ(weights[i, 2], wires=wire)
        
        # CNOT entanglement
        for i in range(len(wires)):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])
    
    def create_circuit(self):
        """
        创建量子电路
        结构: U(θ₀) - U(data) - U(θ₁) - U(data) - ... - U(θₙ)
        """
        @qml.qnode(self.dev)
        def circuit(x, weights):
            wires = range(self.n_qubits)
            
            # 第一个参数层 U(θ₀)
            self.variational_layer(weights[0], wires)
            
            # 交替数据层和参数层
            for i in range(self.n_data_layers):
                self.diagonal_encoding(x, wires)      # U(data)
                self.variational_layer(weights[i+1], wires)  # U(θᵢ₊₁)
            
            # 测量第一个量子比特的 Pauli-Z
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def cost(self, weights, X, y):
        """
        投影测量损失函数
        
        M_0 = (I+Z)/2  →  P(|0⟩) = (1 + ⟨Z⟩) / 2
        M_1 = (I-Z)/2  →  P(|1⟩) = (1 - ⟨Z⟩) / 2
        
        损失 = 测量到错误类别的概率
        """
        circuit = self.create_circuit()
        loss = 0.0
        
        for i in range(len(X)):
            z_exp = circuit(X[i], weights)
            
            if y[i] == 0:
                # 标签为0，希望测到 |0⟩，损失 = P(|1⟩)
                loss = loss + (1 - z_exp) / 2
            else:
                # 标签为1，希望测到 |1⟩，损失 = P(|0⟩)
                loss = loss + (1 + z_exp) / 2
        
        return loss / len(X)
    
    def predict(self, x, weights):
        """
        预测单个样本
        ⟨Z⟩ > 0  →  预测类别 0
        ⟨Z⟩ < 0  →  预测类别 1
        """
        circuit = self.create_circuit()
        z_exp = float(circuit(x, weights))
        return 0 if z_exp > 0 else 1
    
    def get_probabilities(self, x, weights):
        """获取测量概率 P(|0⟩) 和 P(|1⟩)"""
        circuit = self.create_circuit()
        z_exp = float(circuit(x, weights))
        p0 = (1 + z_exp) / 2  # P(|0⟩)
        p1 = (1 - z_exp) / 2  # P(|1⟩)
        return p0, p1
    
    def accuracy(self, weights, X, y):
        """计算准确率"""
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i], weights)
            if pred == y[i]:
                correct += 1
        return correct / len(X)
    
    def prepare_data(self, dataset_name, use_pca=True, n_train=None, n_test=None, random_state=42):
        """
        准备数据并自动调整维度
        
        参数:
        ------
        dataset_name : str
            数据集名称: 'mnist', 'iris', 'breast_cancer', 'ionosphere'
        use_pca : bool, default=True
            是否使用PCA降维。如果为False且数据维度不等于2^n_qubits，将报错
        n_train : int, optional
            训练样本数
        n_test : int, optional
            测试样本数
        random_state : int
            随机种子
        
        返回:
        ------
        X_train, X_test, y_train, y_test, info
        """
        # 从load_data.py加载数据
        if use_pca:
            # 使用PCA降维到所需特征数
            X_train, X_test, y_train, y_test, info = load_data(
                dataset_name=dataset_name,
                use_pca=True,
                pca_dim=self.n_features,  # 自动降维到 2^n_qubits
                n_train=n_train,
                n_test=n_test,
                random_state=random_state
            )
        else:
            # 不使用PCA，直接加载原始数据
            X_train, X_test, y_train, y_test, info = load_data(
                dataset_name=dataset_name,
                use_pca=False,
                n_train=n_train,
                n_test=n_test,
                random_state=random_state
            )
            
            # 检查维度是否匹配
            if X_train.shape[1] != self.n_features:
                raise ValueError(
                    f"数据维度({X_train.shape[1]})与量子比特数不匹配(需要{self.n_features}维)。\n"
                    f"请设置use_pca=True进行自动降维，或调整n_qubits参数。"
                )
        
        # 归一化到 [0, π] 以适配量子编码
        print("归一化特征到 [0, 2π]...")
        scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 转换为PennyLane数组
        X_train = np.array(X_train, requires_grad=False)
        X_test = np.array(X_test, requires_grad=False)
        y_train = np.array(y_train, requires_grad=False)
        y_test = np.array(y_test, requires_grad=False)
        
        print(f"特征形状: {X_train.shape[1]} 维 (匹配 2^{self.n_qubits} = {self.n_features})")
        
        return X_train, X_test, y_train, y_test, info
    
    def train(self, dataset_name, use_pca=True, n_epochs=100, learning_rate=0.1, batch_size=10,
              n_train=None, n_test=None, random_state=42, verbose=True):
        """
        训练量子分类器
        
        参数:
        ------
        dataset_name : str
            数据集名称: 'mnist', 'iris', 'breast_cancer', 'ionosphere'
        use_pca : bool, default=True
            是否使用PCA降维
        n_epochs : int
            训练轮数
        learning_rate : float
            学习率
        batch_size : int
            批次大小
        n_train, n_test : int, optional
            训练/测试样本数
        random_state : int
            随机种子
        verbose : bool
            是否打印详细训练信息
        
        返回:
        ------
        weights : ndarray
            训练好的权重
        history : dict
            训练历史
        data : tuple
            (X_train, X_test, y_train, y_test, info)
        """
        # 准备数据
        X_train, X_test, y_train, y_test, info = self.prepare_data(
            dataset_name, use_pca, n_train, n_test, random_state
        )
        
        # 初始化权重
        np.random.seed(random_state)
        self.weights = np.random.uniform(
            -np.pi, np.pi, 
            (self.n_param_layers, self.n_qubits, 3), 
            requires_grad=True
        )
        
        # 优化器
        optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        
        # 打印训练信息
        if verbose:
            print("\n" + "=" * 70)
            print("开始训练量子分类器")
            print("=" * 70)
            print(f"数据集: {dataset_name.upper()}")
            print(f"PCA降维: {'是' if use_pca else '否'}")
            print(f"电路结构: U(θ₀)", end="")
            for i in range(self.n_data_layers):
                print(f"-U(data)-U(θ{i+1})", end="")
            print()
            print(f"编码方式: 对角编码 (Diagonal Encoding)")
            print(f"损失函数: 投影测量 L = ⟨M_{{1-y}}⟩")
            print(f"参数数量: {self.weights.size}")
            print(f"学习率: {learning_rate}, 批次大小: {batch_size}")
            print("=" * 70)
        
        # 重置历史记录
        self.history = {"loss": [], "train_acc": [], "test_acc": []}
        
        # 训练循环
        for epoch in range(n_epochs):
            # 随机采样批次
            batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # 优化步骤
            self.weights, loss_val = optimizer.step_and_cost(
                lambda w: self.cost(w, X_batch, y_batch), self.weights
            )
            
            # 记录和打印
            if verbose and (epoch + 1) % 10 == 0:
                train_acc = self.accuracy(self.weights, X_train, y_train)
                test_acc = self.accuracy(self.weights, X_test, y_test)
                
                self.history["loss"].append(float(loss_val))
                self.history["train_acc"].append(train_acc)
                self.history["test_acc"].append(test_acc)
                
                print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Loss: {loss_val:.4f} | "
                      f"Train Acc: {train_acc:.2%} | "
                      f"Test Acc: {test_acc:.2%}")
        
        if verbose:
            print("=" * 70)
            final_acc = self.accuracy(self.weights, X_test, y_test)
            print(f"最终测试准确率: {final_acc:.2%}")
            print("=" * 70)
        
        return self.weights, self.history, (X_train, X_test, y_train, y_test, info)
    
    def visualize_results(self, data, history):
        """可视化训练结果"""
        X_train, X_test, y_train, y_test, info = data
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. 损失曲线
        ax = axes[0]
        epochs = range(10, 10 * len(history["loss"]) + 1, 10)
        ax.plot(epochs, history["loss"], "b-", linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Projection Measurement Loss", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax = axes[1]
        ax.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2, marker="o", markersize=4)
        ax.plot(epochs, history["test_acc"], "r--", label="Test", linewidth=2, marker="s", markersize=4)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"Accuracy - {info['dataset_name'].upper()}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # 3. 测试集概率分布
        ax = axes[2]
        probs_class0, probs_class1 = [], []
        labels = []
        for i in range(len(X_test)):
            p0, p1 = self.get_probabilities(X_test[i], self.weights)
            probs_class0.append(p0)
            probs_class1.append(p1)
            labels.append(y_test[i])
        
        labels = np.array(labels)
        colors = ["blue" if l == 0 else "red" for l in labels]
        ax.scatter(probs_class0, probs_class1, c=colors, alpha=0.7, s=60)
        ax.plot([0, 1], [1, 0], "k--", alpha=0.5, label="P(|0⟩) + P(|1⟩) = 1")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("P(|0⟩)", fontsize=11)
        ax.set_ylabel("P(|1⟩)", fontsize=11)
        ax.set_title("Test Set Probabilities", fontsize=12)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pca_suffix = "_pca" if info.get('use_pca', False) else ""
        filename = f"quantum_classifier_{info['dataset_name']}{pca_suffix}.png"
        plt.savefig(filename, dpi=150)
        print(f"\n可视化结果已保存到: {filename}")
        plt.show()
    
    def print_predictions(self, data, n_samples=10):
        """打印测试集预测结果"""
        X_test, y_test = data[1], data[3]
        
        print("\n" + "=" * 70)
        print("测试样本预测结果")
        print("=" * 70)
        print(f"{'样本':<8} {'真实':<8} {'预测':<8} {'P(|0⟩)':<12} {'P(|1⟩)':<12} {'结果':<6}")
        print("-" * 70)
        
        correct = 0
        for i in range(min(n_samples, len(X_test))):
            pred = self.predict(X_test[i], self.weights)
            p0, p1 = self.get_probabilities(X_test[i], self.weights)
            true_label = int(y_test[i])
            
            status = "✓" if pred == true_label else "✗"
            if pred == true_label:
                correct += 1
            
            print(f"{i+1:<8} {true_label:<8} {pred:<8} {p0:<12.4f} {p1:<12.4f} {status:<6}")
        
        print("-" * 70)
        print(f"前 {min(n_samples, len(X_test))} 个样本准确率: {correct}/{min(n_samples, len(X_test))} = {correct/min(n_samples, len(X_test)):.2%}")
        print("=" * 70)
    
    def print_circuit(self, x_sample):
        """打印量子电路结构"""
        circuit = self.create_circuit()
        print("\n" + "=" * 70)
        print("量子电路结构")
        print("=" * 70)
        print(qml.draw(circuit)(x_sample, self.weights))
        print("=" * 70)


# ============== 多次运行实验函数 ==============
def run_multiple_experiments(dataset_name, use_pca, n_qubits, n_data_layers, n_param_layers,
                             n_epochs, learning_rate, batch_size, n_train, n_test, 
                             n_runs=10, base_seed=42, show_details=False):
    """
    多次运行实验并统计结果
    
    参数:
    ------
    dataset_name : str
        数据集名称
    use_pca : bool
        是否使用PCA
    n_qubits, n_data_layers, n_param_layers : int
        量子电路参数
    n_epochs, learning_rate, batch_size : float/int
        训练参数
    n_train, n_test : int
        样本数
    n_runs : int
        运行次数
    base_seed : int
        基础随机种子
    show_details : bool
        是否显示每次运行的详细信息
    
    返回:
    ------
    results : dict
        包含所有运行结果的字典
    """
    print("\n" + "=" * 80)
    print(f"多次实验: {dataset_name.upper()} 数据集 (共 {n_runs} 次运行)")
    print("=" * 80)
    
    test_accuracies = []
    train_accuracies = []
    all_histories = []

    
    for run in range(n_runs):
        seed = base_seed + run
        print(f"\n{'='*80}")
        print(f"运行 {run + 1}/{n_runs} (随机种子: {seed})")
        print(f"{'='*80}")
        
        # 创建分类器
        qc = QuantumBinaryClassifier(
            n_qubits=n_qubits,
            n_data_layers=n_data_layers,
            n_param_layers=n_param_layers
        )
        
        # 训练（verbose控制是否打印详细信息）
        weights, history, data = qc.train(
            dataset_name=dataset_name,
            use_pca=use_pca,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_train=n_train,
            n_test=n_test,
            random_state=seed,
            verbose=show_details  # 只在show_details=True时打印详细训练过程
        )
        
        # 计算准确率
        X_train, X_test, y_train, y_test, info = data
        train_acc = qc.accuracy(weights, X_train, y_train)
        test_acc = qc.accuracy(weights, X_test, y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        all_histories.append(history)
        
        print(f"运行 {run + 1} 结果: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    
    # 计算统计结果
    test_accuracies = np.array(test_accuracies)
    train_accuracies = np.array(train_accuracies)
    
    test_mean = np.mean(test_accuracies)
    test_std = np.std(test_accuracies)
    train_mean = np.mean(train_accuracies)
    train_std = np.std(train_accuracies)
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("最终统计结果")
    print("=" * 80)
    print(f"数据集: {dataset_name.upper()}")
    print(f"PCA降维: {'是' if use_pca else '否'}")
    print(f"量子比特: {n_qubits}, 数据层: {n_data_layers}, 参数层: {n_param_layers}")
    print(f"训练参数: Epochs={n_epochs}, LR={learning_rate}, Batch={batch_size}")
    print(f"样本数: Train={n_train}, Test={n_test}")
    print("-" * 80)
    print(f"运行次数: {n_runs}")
    print(f"训练集准确率: {train_mean:.4f} ± {train_std:.4f}")
    print(f"测试集准确率: {test_mean:.4f} ± {test_std:.4f}")
    print("-" * 80)
    print(f"所有测试准确率: {[f'{acc:.4f}' for acc in test_accuracies]}")
    print(f"最大值: {np.max(test_accuracies):.4f}")
    print(f"最小值: {np.min(test_accuracies):.4f}")
    print(f"中位数: {np.median(test_accuracies):.4f}")
    print("=" * 80)
    
    results = {
        'test_accuracies': test_accuracies,
        'train_accuracies': train_accuracies,
        'test_mean': test_mean,
        'test_std': test_std,
        'train_mean': train_mean,
        'train_std': train_std,
        'histories': all_histories,
        'data': data,
        'info': info
    }
    
    return results


# ============== 主程序 ==============
if __name__ == "__main__":
    
    # ========== 数据集配置 ==========
    datasets_config = [
        {'name': 'mnist', 'n_train': 1000, 'n_test': 200, 'use_pca': True, 'pca_dim': 16},
        {'name': 'iris', 'n_train': 80, 'n_test': 20, 'use_pca': True, 'pca_dim': 4},
        {'name': 'breast_cancer', 'n_train': 400, 'n_test': 100, 'use_pca': True, 'pca_dim': 16},
        {'name': 'ionosphere', 'n_train': 280, 'n_test': 60, 'use_pca': True, 'pca_dim': 16}
    ]
    
    # ========== 通用训练参数 ==========
    N_DATA_LAYERS = 3             # 数据重上传次数
    N_PARAM_LAYERS = 4            # 参数层数量
    
    N_EPOCHS = 1000               # 训练轮数
    LEARNING_RATE = 0.1           # 学习率
    BATCH_SIZE = 20               # 批次大小
    
    N_RUNS = 10                   # 重复运行次数
    BASE_SEED = 42                # 基础随机种子
    SHOW_DETAILS = False          # 是否显示每次运行的详细训练过程
    
    VISUALIZE_LAST = False         # 是否可视化最后一次运行的结果
    SHOW_PREDICTIONS = False      # 是否显示预测结果
    N_PREDICTION_SAMPLES = 15     # 打印多少个预测样本
    
    
    # ========== 存储所有数据集的结果 ==========
    all_results = {}
    
    
    # ========== 遍历每个数据集 ==========
    for config in datasets_config:
        dataset_name = config['name']
        n_train = config['n_train']
        n_test = config['n_test']
        use_pca = config['use_pca']
        pca_dim = config['pca_dim']
        
        # 根据pca_dim计算量子比特数 (pca_dim = 2^n_qubits)
        n_qubits = int(np.log2(pca_dim))
        
        print("\n" + "█" * 80)
        print(f"开始处理数据集: {dataset_name.upper()}")
        print(f"配置: n_train={n_train}, n_test={n_test}, use_pca={use_pca}, pca_dim={pca_dim}, n_qubits={n_qubits}")
        print("█" * 80)
        
        
        # ========== 运行多次实验 ==========
        results = run_multiple_experiments(
            dataset_name=dataset_name,
            use_pca=use_pca,
            n_qubits=n_qubits,
            n_data_layers=N_DATA_LAYERS,
            n_param_layers=N_PARAM_LAYERS,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            n_train=n_train,
            n_test=n_test,
            n_runs=N_RUNS,
            base_seed=BASE_SEED,
            show_details=SHOW_DETAILS
        )
        
        all_results[dataset_name] = results
        
        
        # ========== 可选: 可视化最后一次运行的结果 ==========
        if VISUALIZE_LAST:
            print(f"\n正在生成 {dataset_name} 最后一次运行的可视化结果...")
            qc_last = QuantumBinaryClassifier(
                n_qubits=n_qubits,
                n_data_layers=N_DATA_LAYERS,
                n_param_layers=N_PARAM_LAYERS
            )
            
            # 使用最后一次的数据和历史
            last_history = results['histories'][-1]
            data = results['data']
            
            # 重新训练最后一次（以获取权重）
            seed = BASE_SEED + N_RUNS - 1
            weights, _, _ = qc_last.train(
                dataset_name=dataset_name,
                use_pca=use_pca,
                n_epochs=N_EPOCHS,
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                n_train=n_train,
                n_test=n_test,
                random_state=seed,
                verbose=False
            )
            
            qc_last.visualize_results(data, last_history)
            
            if SHOW_PREDICTIONS:
                qc_last.print_predictions(data, n_samples=N_PREDICTION_SAMPLES)
    
    
    # ========== 输出所有数据集的汇总结果 ==========
    print("\n" + "█" * 80)
    print("【所有数据集最终结果汇总】")
    print("█" * 80)
    print(f"{'数据集':<20} {'训练样本':<12} {'测试样本':<12} {'PCA维度':<10} {'测试准确率':<20}")
    print("-" * 80)
    
    for dataset_name, results in all_results.items():
        config = next(c for c in datasets_config if c['name'] == dataset_name)
        print(f"{dataset_name.upper():<20} {config['n_train']:<12} {config['n_test']:<12} "
              f"{config['pca_dim']:<10} {results['test_mean']:.2%} ± {results['test_std']:.2%}")
    
    print("█" * 80 + "\n")
    
    
    # ========== 可选: 保存结果到文件 ==========
    import json
    results_summary = {}
    for dataset_name, results in all_results.items():
        results_summary[dataset_name] = {
            'test_mean': float(results['test_mean']),
            'test_std': float(results['test_std']),
            'train_mean': float(results['train_mean']),
            'train_std': float(results['train_std']),
            'test_accuracies': [float(acc) for acc in results['test_accuracies']],
            'train_accuracies': [float(acc) for acc in results['train_accuracies']]
        }
    
    with open('quantum_classifier_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("结果已保存到: quantum_classifier_results.json\n")
