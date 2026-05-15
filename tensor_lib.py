import torch
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
def unfold(X, Nway, dim):
    dim2 = list(range(len(Nway))) + list(range(len(Nway)))
    permute_dim = dim2[dim - 1:len(Nway) + dim - 1]
    Y = X.permute(permute_dim)
    Y = Y.reshape(Nway[dim - 1], int(torch.prod(torch.tensor(Nway)) // Nway[dim - 1]))
    return Y

def fold(X, Nway, dim):
    dim2 = list(range(len(Nway))) + list(range(len(Nway)))
    permute_dim = dim2[dim - 1:len(Nway) + dim - 1]
    Nway2 = Nway + Nway
    X = X.reshape(Nway2[dim - 1:len(Nway) + dim - 1])
    reverse_permute_dim = sorted(range(len(permute_dim)), key=lambda k: permute_dim[k])
    Y = X.permute(reverse_permute_dim)
    return Y

def tmprod(X, U, dim):
    Nway = list(X.shape)
    Nway[dim - 1] = U.shape[0]
    Y = fold(torch.matmul(U, unfold(X, X.shape, dim)), Nway, dim)
    return Y


def dft_torch(num,device1,dtype1):
    from scipy.linalg import dft
    return torch.tensor(dft(num), dtype=dtype1,device=device1)/ torch.sqrt(torch.tensor(num, dtype=torch.float32,device=device1))

def dct_torch(num,device1,dtype1):
    import numpy as np
    from scipy.fftpack import dct
    identity_matrix = np.eye(num)
    dct_matrix = dct(identity_matrix, type=2, norm='ortho')
    return torch.tensor(dct_matrix, dtype=dtype1,device=device1)


def prox_TNN_my(Y,rho):
    n1, n2,n3 = Y.shape
    # n12 = min(n1,n2)
    # dtype2 = torch.cuda.ComplexFloat
    Yf = torch.fft.fft(Y,None,2,"ortho")
    X = torch.zeros(n1,n2,n3,device=Yf.device,dtype=Yf.dtype)
    for i in range(n3):
        U1,S1,V1 = torch.svd(Yf[:,:,i],)
        S=torch.clamp(S1 - rho, min=0)
        X[:,:,i] = torch.matmul(U1, torch.matmul(torch.diag(S).to(dtype=U1.dtype), V1.conj().t()))
    return torch.fft.ifft(X,None,2,"ortho").real


def prox_TNN_dct(Y,rho):
    n1, n2,n3 = Y.shape
    device1=Y.device
    dtype1=Y.dtype
    # dft_matrix = dft_torch(n3,device1,dtype1)
    dct_matrix = dct_torch(n3,device1,dtype1)
    Yf = tmprod(Y,dct_matrix,3)
    # Uf = torch.zeros(n1,n12,n3,device=Y.device,dtype=dtype2)
    # Sf = torch.zeros(n12,n12,n3,device=Y.device,dtype=dtype2)
    # Vf = torch.zeros(n2,n12,n3,device=Y.device,dtype=dtype2)
    X = torch.zeros(n1,n2,n3,device=Yf.device,dtype=Yf.dtype)
    for i in range(n3):
        U1,S1,V1 = torch.svd(Yf[:,:,i],)
        S=torch.clamp(S1 - rho, min=0)
        # S =torch.diag(S)
        X[:,:,i] = torch.matmul(U1, torch.matmul(torch.diag(S).to(dtype=U1.dtype), V1.t()))
    # return torch.fft.ifft(X,None,2,"ortho").real
    return tmprod(X,dct_matrix.t(),3)

def LRTC_TNN(B, mask, rho, beta, Xture):
    # 参数
    tol = 1e-4
    maxit = 150
    max_beta = 1e8
    # mask_bool = mask.bool()
    mask2 = 1 - mask
    n1, n2, n3 = B.shape
    X = B
    Y = torch.zeros(X.shape, device=X.device, dtype=X.dtype)
    M = Y
    for iter in range(maxit):
        Xold = X.clone()
        ##
        Y = prox_TNN_my(X - M / beta, 1 / beta)

        ##
        X = Y + M / beta

        X = X * mask2 + B * mask
        # X = X.real
        ##
        res = torch.norm(X - Xold, 'fro') / torch.norm(Xold, 'fro')
        resT = torch.norm(X - Xture, 'fro') / torch.norm(Xture, 'fro')
        ps = peak_signal_noise_ratio(np.clip(X.cpu().detach().numpy(), 0, 1), Xture.cpu().detach().numpy(),
                                     data_range=1.0)
        if iter % 10 == 0:
            print('iteration:', iter, 'PSNR', ps, 'res', res, 'resT', resT)
        if res < tol:
            break
        M = M + beta * (Y - X)
        beta = beta * rho
    return X


def LRTC_TNN_DFT(B, mask, opts):
    # 参数
    tol = opts.get('tol')
    maxit = opts.get('maxit')
    max_beta = opts.get('max_beta')
    rho = opts.get('rho')
    beta = opts.get('beta')
    Xture = opts.get('Xture')
    # mask_bool = mask.bool()
    mask2 = 1 - mask
    #
    n1, n2, n3 = B.shape
    X = B
    Y = torch.zeros(X.shape, device=X.device, dtype=X.dtype)
    M = Y

    Out = {'Res': [], 'ResT': [], 'PSNR': [], 'iter': []}
    iter_stop = 0
    for iter in range(maxit):
        Xold = X.clone()
        ##
        Y = prox_TNN_my(X - M / beta, 1 / beta)

        ##
        X = Y + M / beta

        X = X * mask2 + B * mask
        # X = X.real
        ##
        res = (torch.norm(X - Xold, 'fro') / torch.norm(Xold, 'fro')).cpu().detach().numpy()
        resT = (torch.norm(X - Xture, 'fro') / torch.norm(Xture, 'fro')).cpu().detach().numpy()
        ps = peak_signal_noise_ratio(np.clip(X.cpu().detach().numpy(), 0, 1), Xture.cpu().detach().numpy(),
                                     data_range=1.0)
        if iter % 2 == 0:
            print('DFT: iteration:', iter, 'PSNR', ps, 'res', res, 'resT', resT)
        Out['Res'].append(res)
        Out['ResT'].append(resT)
        Out['PSNR'].append(ps)
        if res < tol:
            break

        M = M + beta * (Y - X)
        # beta =max(beta*rho,max_beta)
        beta = beta * rho
        #
        iter_stop = iter
    Out['iter'].append(iter_stop+1)
    return X, Out


def LRTC_TNN_DCT(B, mask, opts):
    # 参数
    tol = opts.get('tol')
    maxit = opts.get('maxit')
    max_beta = opts.get('max_beta')
    rho = opts.get('rho')
    beta = opts.get('beta')
    Xture = opts.get('Xture')
    # mask_bool = mask.bool()
    mask2 = 1 - mask
    #
    n1, n2, n3 = B.shape
    X = B
    Y = torch.zeros(X.shape, device=X.device, dtype=X.dtype)
    M = Y

    Out = {'Res': [], 'ResT': [], 'PSNR': [], 'iter': []}
    iter_stop = 0
    for iter in range(maxit):
        Xold = X.clone()
        ##
        Y = prox_TNN_dct(X - M / beta, 1 / beta)
        ##
        X = Y + M / beta
        X = X * mask2 + B * mask
        # X = X.real
        ##
        res = (torch.norm(X - Xold, 'fro') / torch.norm(Xold, 'fro')).cpu().detach().numpy()
        resT = (torch.norm(X - Xture, 'fro') / torch.norm(Xture, 'fro')).cpu().detach().numpy()
        ps = peak_signal_noise_ratio(np.clip(X.cpu().detach().numpy(), 0, 1), Xture.cpu().detach().numpy(),
                                     data_range=1.0)
        if iter % 2 == 0:
            print('DCT: iteration:', iter, 'PSNR', ps, 'res', res, 'resT', resT)
        Out['Res'].append(res)
        Out['ResT'].append(resT)
        Out['PSNR'].append(ps)
        if res < tol:
            break

        M = M + beta * (Y - X)
        # beta =max(beta*rho,max_beta)
        beta = beta * rho
        #
        iter_stop = iter

    Out['iter'].append(iter_stop+1)
    return X, Out