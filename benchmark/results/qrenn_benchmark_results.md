# QRENN Benchmark Results

## Experimental Settings

- **Model**: QRENN (1 trainable qubit, variable control qubits, 6 layers)
- **Training**: 500 iterations, Adam optimizer, lr=0.1
- **Rounds**: 10 rounds per dataset

## Results Summary

| Dataset | Train | Test | Components | Control Qubits | Avg Accuracy | Std Dev | Min | Max |
|---------|-------|------|------------|----------------|--------------|---------|-----|-----|
| mnist | 1000 | 200 | 16 | 4 | 0.9950 | 0.0000 | 0.9950 | 0.9950 |
| iris | 80 | 20 | 4 | 2 | 0.9500 | 0.0000 | 0.9500 | 0.9500 |
| breast_cancer | 400 | 100 | 16 | 4 | 0.9300 | 0.0000 | 0.9300 | 0.9300 |
| ionosphere | 280 | 60 | 16 | 4 | 0.7082 | 0.0451 | 0.6230 | 0.7705 |

## Detailed Results

### MNIST

- **Training samples**: 1000
- **Test samples**: 200
- **PCA components**: 16
- **Control qubits**: 4
- **Average accuracy**: 0.9950 ± 0.0000
- **Min accuracy**: 0.9950
- **Max accuracy**: 0.9950
- **Individual accuracies**: 0.9950, 0.9950, 0.9950, 0.9950, 0.9950, 0.9950, 0.9950, 0.9950, 0.9950, 0.9950

### IRIS

- **Training samples**: 80
- **Test samples**: 20
- **PCA components**: 4
- **Control qubits**: 2
- **Average accuracy**: 0.9500 ± 0.0000
- **Min accuracy**: 0.9500
- **Max accuracy**: 0.9500
- **Individual accuracies**: 0.9500, 0.9500, 0.9500, 0.9500, 0.9500, 0.9500, 0.9500, 0.9500, 0.9500, 0.9500

### BREAST_CANCER

- **Training samples**: 400
- **Test samples**: 100
- **PCA components**: 16
- **Control qubits**: 4
- **Average accuracy**: 0.9300 ± 0.0000
- **Min accuracy**: 0.9300
- **Max accuracy**: 0.9300
- **Individual accuracies**: 0.9300, 0.9300, 0.9300, 0.9300, 0.9300, 0.9300, 0.9300, 0.9300, 0.9300, 0.9300

### IONOSPHERE

- **Training samples**: 280
- **Test samples**: 60
- **PCA components**: 16
- **Control qubits**: 4
- **Average accuracy**: 0.7082 ± 0.0451
- **Min accuracy**: 0.6230
- **Max accuracy**: 0.7705
- **Individual accuracies**: 0.6557, 0.7377, 0.7213, 0.7541, 0.6230, 0.7705, 0.7213, 0.7213, 0.7213, 0.6557

