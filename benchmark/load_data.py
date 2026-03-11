import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from quairkit import State, Circuit
from quairkit.database import zero_state
import warnings
warnings.filterwarnings('ignore')

def load_data(dataset_name, use_pca=False, pca_dim=None, 
              n_train=None, n_test=None, random_state=42):
    """
    加载并预处理二分类数据集
    
    参数:
    --------
    dataset_name : str
        数据集名称，可选 'mnist', 'iris', 'breast_cancer', 'ionosphere'
    use_pca : bool, default=False
        是否使用PCA降维
    pca_dim : int, optional
        PCA降维后的维度。如果为None且use_pca=True，则保留95%方差
    n_train : int, optional
        训练集样本数量。如果为None，则使用默认划分
    n_test : int, optional
        测试集样本数量。如果为None，则使用默认划分
    random_state : int, default=42
        随机种子
        
    返回:
    --------
    X_train : ndarray, shape (n_train, n_features)
        训练集特征
    X_test : ndarray, shape (n_test, n_features)
        测试集特征
    y_train : ndarray, shape (n_train,)
        训练集标签 (0或1)
    y_test : ndarray, shape (n_test,)
        测试集标签 (0或1)
    info : dict
        数据集信息，包含原始维度、PCA后维度、样本数等
    """
    
    # ==================== 1. 加载原始数据 ====================
    if dataset_name.lower() == 'mnist':
        print("正在加载MNIST数据集...")
        # 加载MNIST (使用0和1两个数字进行二分类)
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = np.array(mnist.data, dtype=np.float64)
        y = np.array(mnist.target, dtype=int)
        
        # 只取0和1两类
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        y = (y == 1).astype(int)  # 转换为0/1标签
        
        # 归一化到[0, 1]
        X = X / 255.0
      
    elif dataset_name.lower() in ['fashion_mnist', 'fashionmnist']:
        print("正在加载Fashion-MNIST数据集...")
        fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
        X = np.array(fmnist.data, dtype=np.float64)
        y = np.array(fmnist.target, dtype=int)
        
        # ================= 修改这里 =================
        # 定义你想要进行二分类的两个类别
        class_neg = 0  # 负类 (最终映射为 0)
        class_pos = 1  # 正类 (最终映射为 1)
        
        # 1. 过滤出这两个类别
        mask = (y == class_neg) | (y == class_pos)
        X, y = X[mask], y[mask]
        
        # 2. 重新映射为 0 和 1
        y = (y == class_pos).astype(int) 
        # ==========================================
        
        # 归一化到[0, 1]
        X = X / 255.0
        
    elif dataset_name.lower() == 'iris':
        print("正在加载Iris数据集...")
        # 加载Iris (使用前两类：Setosa vs Versicolor)
        data = load_iris()
        X, y = data.data, data.target
        
        # 只取前两类
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        
    elif dataset_name.lower() == 'breast_cancer':
        print("正在加载Breast Cancer数据集...")
        # 加载乳腺癌数据集 (恶性 vs 良性)
        data = load_breast_cancer()
        X, y = data.data, data.target
        
    elif dataset_name.lower() == 'ionosphere':
        print("正在加载Ionosphere数据集...")
        # 加载电离层数据集 (雷达信号分类：好 vs 坏)
        iono = fetch_openml('ionosphere', version=1, as_frame=False)
        X = np.array(iono.data, dtype=np.float64)
        y_raw = iono.target
        
        # 将标签转换为0/1 ('g'=good=1, 'b'=bad=0)
        y = np.array([1 if label == 'g' else 0 for label in y_raw])
    
    elif dataset_name.lower() == 'banknote':
        dataset = fetch_openml(name='banknote-authentication', version=1, as_frame=True, parser='auto')
        X = dataset.data.values
        y = dataset.target.astype(int).values

    elif dataset_name.lower() == 'heart_disease':
        print("正在加载Heart Disease数据集...")
        from sklearn.preprocessing import LabelEncoder
        
        # 加载 Statlog Heart 数据集
        heart = fetch_openml('heart-statlog', version=1, as_frame=False, parser='auto')
        X = np.array(heart.data, dtype=np.float64)
        y_raw = heart.target

        # 使用 LabelEncoder 自动将两类标签转换为 0 和 1
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

    elif dataset_name.lower() == 'moon':
        print("正在生成Moon数据集...")
        # 手动生成双月形状的二维数据集
        # n_samples: 总样本数 (800训练 + 200测试 = 1000)
        # noise: 噪声标准差，控制两个月亮的交叉重叠程度 (0.2 是比较经典的难度)
        # random_state: 固定随机种子，保证每次生成的数据完全一样
        X_raw, y_raw = make_moons(n_samples=1000, noise=0.2, random_state=42)
        
        X = np.array(X_raw, dtype=np.float64)
        y = np.array(y_raw, dtype=int)  # 标签本身就是 0 和 1
                
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}。"
                        f"可选: 'mnist', 'iris', 'breast_cancer', 'ionosphere', 'banknote'")
    
    original_dim = X.shape[1]
    total_samples = X.shape[0]
    
    print(f"原始数据: {total_samples} 个样本, {original_dim} 维特征")
    print(f"类别分布: 类0={np.sum(y==0)}, 类1={np.sum(y==1)}")
    
    # ==================== 2. 数据标准化 ====================
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # ==================== 3. PCA降维 (可选) ====================
    if use_pca:
        if pca_dim is None:
            # 保留95%的方差
            pca = PCA(n_components=0.95, random_state=random_state)
            X = pca.fit_transform(X)
            pca_dim = X.shape[1]
            print(f"PCA降维: {original_dim}D → {pca_dim}D (保留95%方差)")
        else:
            # 降维到指定维度
            pca_dim = min(pca_dim, X.shape[1], X.shape[0])
            pca = PCA(n_components=pca_dim, random_state=random_state)
            X = pca.fit_transform(X)
            explained_var = np.sum(pca.explained_variance_ratio_) * 100
            print(f"PCA降维: {original_dim}D → {pca_dim}D (保留{explained_var:.2f}%方差)")
    else:
        pca_dim = original_dim
    
    # ==================== 4. 划分训练集和测试集 ====================
    if n_train is None and n_test is None:
        # 默认划分：80% 训练，20% 测试
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # 按指定数量划分
        if n_train is None:
            n_train = total_samples - n_test
        if n_test is None:
            n_test = total_samples - n_train
            
        # 确保不超过总样本数
        if n_train + n_test > total_samples:
            raise ValueError(f"n_train({n_train}) + n_test({n_test}) > 总样本数({total_samples})")
        
        # 如果指定数量小于总数，先采样再划分
        if n_train + n_test < total_samples:
            X, _, y, _ = train_test_split(
                X, y, train_size=n_train + n_test, 
                random_state=random_state, stratify=y
            )
        
        # 划分训练集和测试集
        test_ratio = n_test / (n_train + n_test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=y
        )
    
    # ==================== 5. 整理输出信息 ====================
    info = {
        'dataset_name': dataset_name,
        'original_dim': original_dim,
        'final_dim': pca_dim,
        'use_pca': use_pca,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_total': len(X_train) + len(X_test),
        'train_class_0': np.sum(y_train == 0),
        'train_class_1': np.sum(y_train == 1),
        'test_class_0': np.sum(y_test == 0),
        'test_class_1': np.sum(y_test == 1),
    }
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"特征维度: {original_dim} → {pca_dim} (PCA: {use_pca})")
    print(f"训练集: {len(X_train)} (类0: {info['train_class_0']}, 类1: {info['train_class_1']})")
    print(f"测试集: {len(X_test)} (类0: {info['test_class_0']}, 类1: {info['test_class_1']})")
    print(f"{'='*60}\n")
    
    return X_train, X_test, y_train, y_test, info


class StateLoader:
    r'''use for loading empirical initial state for QRENN.

    '''
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def load_state(self, data_name:str) -> State:
        if data_name == 'mnist':
            return zero_state(self.num_qubits)
        elif data_name == 'iris':
            return zero_state(self.num_qubits)
        elif data_name == 'breast_cancer':
            return zero_state(self.num_qubits)
        elif data_name == 'ionosphere':
            cir = Circuit(self.num_qubits)
            cir.x(qubits_idx=[2])
            return cir()
        else:
            raise ValueError(f"不支持的数据集: {data_name}。"
                        f"可选: 'mnist', 'iris', 'breast_cancer', 'ionosphere'")



# ==================== 使用示例 ====================
if __name__ == "__main__":
    
    # 示例1: MNIST，使用PCA降到10维
    X_train, X_test, y_train, y_test, info = load_data(
        dataset_name='mnist',
        use_pca=True,
        pca_dim=10,
        n_train=1000,
        n_test=200
    )
    
    # 示例2: Iris，不使用PCA
    X_train, X_test, y_train, y_test, info = load_data(
        dataset_name='iris',
        use_pca=False,
        n_train=80,
        n_test=20
    )
    
    # 示例3: Breast Cancer，自动选择PCA维度
    X_train, X_test, y_train, y_test, info = load_data(
        dataset_name='breast_cancer',
        use_pca=True,
        pca_dim=None,  # 自动保留95%方差
        n_train=400,
        n_test=100
    )
    
    # 示例4: Ionosphere，不使用PCA
    X_train, X_test, y_train, y_test, info = load_data(
        dataset_name='ionosphere',
        use_pca=False,
        n_train=240,
        n_test=60
    )
    
    # 打印返回的info字典
    print("Info字典内容:")
    for key, value in info.items():
        print(f"  {key}: {value}")
