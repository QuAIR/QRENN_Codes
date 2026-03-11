# QSVM Benchmark Results

## Experimental Settings

- **Model**: QSVM (ZZ Feature Map)
- **C**: 1.0 (SVM regularization parameter)
- **Depth**: 10 (feature map depth)
- **Rounds**: 10 rounds per dataset

## Results Summary

| Dataset | Train | Test | Components | Qubits | Depth | C | Avg Accuracy | Std Dev | Min | Max |
|---------|-------|------|------------|---------|-------|---|--------------|---------|-----|-----|
| mnist | 100 | 20 | 4 | 2 | 10 | 1.0 | 0.6500 | 0.0000 | 0.6500 | 0.6500 |
| iris | 80 | 20 | 4 | 2 | 10 | 1.0 | 0.5500 | 0.0000 | 0.5500 | 0.5500 |
| breast_cancer | 400 | 100 | 4 | 2 | 10 | 1.0 | 0.7000 | 0.0000 | 0.7000 | 0.7000 |
| ionosphere | 280 | 60 | 4 | 2 | 10 | 1.0 | 0.6393 | 0.0000 | 0.6393 | 0.6393 |

## Detailed Results

### MNIST

- **Training samples**: 100
- **Test samples**: 20
- **PCA components**: 4
- **Qubits**: 2
- **Feature map depth**: 10
- **C**: 1.0
- **Average accuracy**: 0.6500 ¡À 0.0000
- **Min accuracy**: 0.6500
- **Max accuracy**: 0.6500
- **Individual accuracies**: 0.6500, 0.6500, 0.6500, 0.6500, 0.6500, 0.6500, 0.6500, 0.6500, 0.6500, 0.6500

### IRIS

- **Training samples**: 80
- **Test samples**: 20
- **PCA components**: 4
- **Qubits**: 2
- **Feature map depth**: 10
- **C**: 1.0
- **Average accuracy**: 0.5500 ¡À 0.0000
- **Min accuracy**: 0.5500
- **Max accuracy**: 0.5500
- **Individual accuracies**: 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500

### BREAST_CANCER

- **Training samples**: 400
- **Test samples**: 100
- **PCA components**: 4
- **Qubits**: 2
- **Feature map depth**: 10
- **C**: 1.0
- **Average accuracy**: 0.7000 ¡À 0.0000
- **Min accuracy**: 0.7000
- **Max accuracy**: 0.7000
- **Individual accuracies**: 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000

### IONOSPHERE

- **Training samples**: 280
- **Test samples**: 60
- **PCA components**: 4
- **Qubits**: 2
- **Feature map depth**: 10
- **C**: 1.0
- **Average accuracy**: 0.6393 ¡À 0.0000
- **Min accuracy**: 0.6393
- **Max accuracy**: 0.6393
- **Individual accuracies**: 0.6393, 0.6393, 0.6393, 0.6393, 0.6393, 0.6393, 0.6393, 0.6393, 0.6393, 0.6393

