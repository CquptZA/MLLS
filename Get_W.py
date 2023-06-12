import numpy as np
from scipy.spatial.distance import pdist, squareform
optmParameter = {
    'alpha': 2**(-10),  # 2.^[-10:10] % label correlation
    'beta': 2**(-4),  # 2.^[-10:10] % label specific feature 
    'gamma': 2**3,  # {0.1, 1, 10} % initialization for W
    'lamda': 2**(-8),  # instance correlation
    'lamda2': 2**(-4),  # common features
    'maxIter': 30,  # 最大迭代次数
    'minimumLossMargin': 0.0001,  # 两次迭代的最小损失间距 0.0001
    'bQuiet': 1
}
def CLML(X, Y, optmParameter):
    # optimization parameters
    alpha = optmParameter['alpha']
    beta = optmParameter['beta']
    gamma = optmParameter['gamma']
    lamda = optmParameter['lamda']
    lamda2 = optmParameter['lamda2']
    maxIter = optmParameter['maxIter']
    miniLossMargin = optmParameter['minimumLossMargin']
    
    # initialization
    num_dim = X.shape[1]
    XT = X.T
    XTX = np.dot(XT, X)
    XTY = np.dot(XT, Y)
    W_s = np.linalg.solve(XTX + gamma*np.eye(num_dim), XTY)
    W_s_1 = W_s
    
    # label correlation
    R = pdist(Y.T+np.finfo(float).eps, metric='cosine')
    R = 1 - squareform(R)
    C = R.reshape(Y.shape[1], Y.shape[1])
    L1 = np.diag(np.sum(C, axis=1)) - C
    
    # instance correlation
#     S = ins_similarity(X, 10)
#     L2 = np.diag(np.sum(S, axis=1)) - S
    
    iter = 1
    oldloss = 0
    bk = 1
    bk_1 = 1
    
    # compute LIP
    A = gradL21(W_s)
#     Lip = np.sqrt(4*(np.linalg.norm(XTX)**2 + np.linalg.norm(alpha*XTX)**2 * np.linalg.norm(L1)**2 
#                   + np.linalg.norm(lamda*XT.dot(L2).dot(X))**2) + np.linalg.norm(lamda2*A)**2)
    Lip = np.sqrt(4*(np.linalg.norm(XTX)**2 + np.linalg.norm(alpha*XTX)**2 * np.linalg.norm(L1)**2) + np.linalg.norm(lamda2*A)**2)   
    # proximal gradient
    while iter <= maxIter:
        A = gradL21(W_s)
        W_s_k = W_s + (bk_1 - 1)/bk * (W_s - W_s_1)
#         gradF = np.dot(XTX, W_s_k) - XTY + alpha * np.dot(XTX, W_s_k).dot(L1) + lamda * XT.dot(L2).dot(X).dot(W_s_k) + lamda2*A.dot(W_s_k)
        gradF = np.dot(XTX, W_s_k) - XTY + alpha * np.dot(XTX, W_s_k).dot(L1)  + lamda2*A.dot(W_s_k)
        Gw_s_k = W_s_k - 1/Lip *(gradF)
        # update b, W
        bk_1 = bk
        bk = (1 + np.sqrt(4*bk**2 + 1))/2
        W_s_1 = W_s
        W_s = softthres(Gw_s_k, beta/Lip)
        
        # compute loss function
        predictionLoss = np.trace(np.dot(X.dot(W_s) - Y, (X.dot(W_s) - Y).T))
        F = X.dot(W_s)
        correlation = np.trace(F.dot(L1).dot(F.T))
#         In_correlation = np.trace(F.T.dot(L2).dot(F))
        sparsity = np.sum(W_s != 0)
        sparsity2 = np.trace(np.dot(W_s.T, A).dot(W_s))
        totalloss = predictionLoss + alpha*correlation + beta*sparsity  + lamda2*sparsity2
#         totalloss = predictionLoss + alpha*correlation + beta*sparsity + lamda*In_correlation + lamda2*sparsity2
        if abs(oldloss - totalloss) <= miniLossMargin or totalloss <= 0:
            break
        else:
            oldloss = totalloss
        iter += 1
    
    model_NewLLSF = W_s
    return model_NewLLSF
def softthres(W_t, lambda_):
    return np.maximum(W_t-lambda_, 0) - np.maximum(-W_t-lambda_, 0)

def ins_similarity(X, K):
    A = squareform(pdist(X))
    num_dim = A.shape[0]
    for i in range(num_dim):
        temp = A[i,:]
        As = np.sort(temp)
        temp = (temp <= As[K])
        A[i,:] = temp
    return A
def label_similarity(Y):
    m = Y.shape[1]
    cos_sim = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                cos_sim[i, j] = 1.0
            else:
                cos_sim[i, j] = np.dot(Y[:, i], Y[:, j]) / (np.linalg.norm(Y[:, i]) * np.linalg.norm(Y[:, j]))
    return cos_sim
def gradL21(W):
    num = W.shape[0]
    D = np.zeros((num, num))
    for i in range(num):
        temp = np.linalg.norm(W[i,:], 2)
        if temp != 0:
            D[i,i] = 1/temp
        else:
            D[i,i] = 0
    return D