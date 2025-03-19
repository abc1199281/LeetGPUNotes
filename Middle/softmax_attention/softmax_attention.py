import numpy as np
import math

def solve_cpu(Q, K, V, output):
    """
    Computes the softmax attention operation using CPU.

    Args:
        Q: Query matrix of size M x d.
        K: Key matrix of size N x d.
        V: Value matrix of size N x d.
        output: Output matrix to store the result.
    """

    M, d = Q.shape
    N, _ = K.shape

    # Q * K^T
    QK_T = np.matmul(Q, K.T)

    # Scale by sqrt(d)
    QK_T_scaled = QK_T / math.sqrt(d)

    # Softmax (row-wise)
    attention_weights = np.zeros_like(QK_T_scaled)
    for i in range(M):
        row = QK_T_scaled[i]
        row_max = np.max(row)
        row_exp = np.exp(row - row_max)  # Subtract max for numerical stability
        row_sum = np.sum(row_exp)
        attention_weights[i] = row_exp / row_sum

    # Attention weights * V
    output[:] = np.matmul(attention_weights, V)

if __name__ == '__main__':
    # Example 1
    Q1 = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])
    K1 = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])
    V1 = np.array([[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0],
                   [9.0, 10.0, 11.0, 12.0]])
    output1 = np.zeros((Q1.shape[0], V1.shape[1]))
    solve_cpu(Q1, K1, V1, output1)
    print("Output 1:")
    print(output1)

    # Example 2
    Q2 = np.array([[1.0, 2.0]])
    K2 = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
    V2 = np.array([[3.0, 4.0],
                   [5.0, 6.0]])
    output2 = np.zeros((Q2.shape[0], V2.shape[1]))
    solve_cpu(Q2, K2, V2, output2)
    print("\nOutput 2:")
    print(output2)