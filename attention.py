import numpy as np


def softmax(x):
    """Calculates the softmax for a given array"""
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def attention(Q, K, V):
    """Calculates the self-attention for the given Query,Key,Value"""
    return softmax(np.dot(Q, np.transpose(K)) / np.sqrt(K.shape[1])) @ V


def multi_head_attn(Q, K, V, heads):
    """Calculates the multi-head attention based on the number of heads"""
    Q_ = np.hsplit(Q, heads)
    K_ = np.hsplit(K, heads)
    V_ = np.hsplit(V, heads)
    mh_attn = [attention(q, k, v) for q, k, v in zip(Q_, K_, V_)]
    return np.concatenate(mh_attn, axis=-1)


# Generates a random array
x = np.random.rand(3, 8)
Q = np.copy(x)
K = np.copy(x)
V = np.copy(x)


attn_val = attention(Q, K, V)
multihead_attn_val = multi_head_attn(Q, K, V, 4)

print(x, "\n")
print(attn_val, "\n")
print(multihead_attn_val, "\n")
