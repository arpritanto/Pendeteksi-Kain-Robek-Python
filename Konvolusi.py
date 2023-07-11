import numpy as np
def conv(X, F):
    X_height = X.shape[0]
    X_width = X.shape[1]

    F_height = F.shape[0]
    F_width = F.shape[1]

    H = (F_height) // 2
    W = (F_width) // 2

    out = np.zeros((X_height, X_width))

    for i in np.arange(H + 1, X_height - H):
        for j in np.arange(W + 1, X_width - W):
            sum = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum += (w * a)
            out[i, j] = sum
    return out

def conv2(X, F):
    X_Height = X.shape[0]
    X_Width = X.shape[1]
    F_Height = F.shape[0]
    F_Width = F.shape[1]
    H = 0
    W = 0
    batas = (F_Height) // 2
    out = np.zeros((X_Height, X_Width))
    for i in np.arange(H, X_Height - batas):
        for j in np.arange(W, X_Width - batas):
            sum = 0
            for k in np.arange(H, F_Height):
                for l in np.arange(W, F_Width):
                    a = X[i + k, j + l]
                    w = F[H + k, W + l]
                    sum += (w * a)
            out[i, j] = sum
    return out