import scipy.io
import numpy as np
from tensor_lib import *
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math
import torch.nn.init as init
import scipy.io as sio
dtype = torch.cuda.FloatTensor



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def parameter_output(params):
    output_dir = 'result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_file_path = os.path.join(output_dir, 'NICoTR_Parameters.txt')
    with open(txt_file_path, 'a') as txt_file:
        line = ', '.join([f"{key}: {value}" for key, value in params.items()])
        txt_file.write(line + "\n")

    df = pd.DataFrame([params])

    excel_file_path = os.path.join(output_dir, 'NICoTR_Parameters.xlsx')
    if not os.path.isfile(excel_file_path):
        df.to_excel(excel_file_path, index=False)
    else:
        with pd.ExcelWriter(excel_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)


class NICoTR(nn.Module):
    def __init__(self, Nway, R0, R_Align):
        super(NICoTR, self).__init__()
        N1, N2, N3 = Nway
        RA_1, RA_2,RA_3 = R_Align
        R0_1, R0_2, R0_3 = R0
        self.Nway = Nway
        self.R_Align = R_Align
        self.R0 = R0
        self.G1 = nn.Parameter(torch.rand(R0_1, N1, R0_2, requires_grad=True))
        self.G2 = nn.Parameter(torch.rand(RA_2, N2, R0_3, requires_grad=True))
        self.G3 = nn.Parameter(torch.rand(RA_3, N3, RA_1, requires_grad=True))
        self.R2_net = nn.Sequential(nn.Linear(R0_2, RA_2, bias=False),
                                     nn.LeakyReLU(),
                                     )
        self.R31_net = nn.Sequential(nn.Linear(R0_3*R0_1, RA_3*RA_1, bias=False),
                                     nn.LeakyReLU(),
                                   )
        self.reset_parameters()
    def reset_parameters(self):
        for param in [self.G1, self.G2, self.G3]:
            nn.init.kaiming_normal_(param, mode='fan_out')
        for net in [self.R2_net, self.R31_net]:
            for module in net:
                if hasattr(module, 'weight') and module.weight is not None:
                    init.kaiming_normal_(module.weight, mode='fan_out')

    def forward(self):
        N1, N2, N3 =self.Nway
        RA_1, RA_2,RA_3 = self.R_Align
        R0_1, R0_2, R0_3 = self.R0
        G1, G2, G3 = self.G1, self.G2, self.G3
        G1 = self.R2_net(G1)

        G1_reshaped = G1.reshape(R0_1 * N1, RA_2)
        G2_reshaped = G2.permute(2, 1, 0).reshape(RA_2, N2 * R0_3)
        M = torch.matmul(G1_reshaped, G2_reshaped)
        M = M.reshape(R0_1, N1, N2, R0_3)

        M_rotated = M.permute(1, 2, 3, 0).reshape(N1 * N2, R0_1 * R0_3)
        M_rotated = self.R31_net(M_rotated)
        G3_rotated = G3.permute(1, 0, 2).reshape(RA_1 * RA_3, N3)
        result_matrix = torch.matmul(M_rotated, G3_rotated)

        result_tensor = result_matrix.reshape(N1, N2, N3)
        return result_tensor




max_iter = 15001

for SR in ['10']:
    for data in ['MSI_toy']:
        for R_1 in [30]:
            for R_2 in [16]:
                for R_3 in [30]:
                    R_Align = [R_1, R_2, R_3]
                    R_0 = np.array(R_Align) + 2
                    file_name2 = 'data/' + data + SR + '.mat'
                    mat = scipy.io.loadmat(file_name2)
                    X_np = mat["X"]
                    XT = torch.from_numpy(X_np).type(dtype).cuda()
                    X0_np = mat["X0"]
                    X0 = torch.from_numpy(X0_np).type(dtype).cuda()
                    mask_np = mat["mask"]
                    mask = torch.from_numpy(mask_np).type(dtype).cuda()
                    X_observation = XT * mask
                    Xture = XT.clone()





                    Nway = X_observation.shape
                    model = NICoTR(Nway, R_0, R_Align).type(dtype)
                    params = []
                    params += [x for x in model.parameters()]

                    s = sum([np.prod(list(p.size())) for p in params]);
                    print('Number of params: %d' % s)
                    optimizier = optim.Adam(params, lr=0.002, weight_decay=1e-7 )

                    start_time = time.perf_counter()
                    F_norm = nn.MSELoss()
                    X_Out_real = X_observation
                    psnr_log = 0
                    ssim_log = 0
                    resT_log = 0
                    iter_log = 0
                    for iter in range(max_iter):
                        Xold = X_Out_real
                        X_Out_real = model()
                        loss = F_norm(X_Out_real * mask, XT * mask)
                        X_Out_real[mask == 1] = XT[mask == 1]

                        optimizier.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizier.step()
                        if iter % 500 == 0:
                            ps = peak_signal_noise_ratio(np.clip(X_Out_real.cpu().detach().numpy(), 0, 1), XT.cpu().detach().numpy(),
                                                         data_range=1.0)
                            ss = structural_similarity(np.clip(X_Out_real.cpu().detach().numpy(), 0, 1),
                                                       XT.cpu().detach().numpy(),
                                                       data_range=1.0,
                                                       channel_axis=2)
                            res = torch.norm(X_Out_real- Xold, 'fro') / torch.norm(Xold, 'fro')
                            resT = torch.norm(X_Out_real - XT, 'fro') / torch.norm(XT, 'fro')
                            if ps > psnr_log:
                                psnr_log = ps
                                ssim_log = ss
                                resT_log = resT
                                iter_log = iter
                            print(f"iteration：{iter},PSNR：{ps:.4f},SSIM：{ss:.4f},res：{res:.5f},resT：{resT:.5f} ")




                    # X_IrTR = X_Out_real.detach().cpu().numpy()
                    # filename = f"result_MSI/{data}_{SR}_IrTR.mat"
                    # sio.savemat(filename, {"X_Out_real": X_IrTR})

                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    print(f"PSNR：{psnr_log:.4f},iter：{iter_log},Time：{execution_time:.3f}s ")

                    params = {
                        'PSNR': 0,
                    }
                    params['PSNR'] = psnr_log
                    params['SSIM'] = ssim_log
                    params['resT'] = resT_log.cpu().detach().numpy()
                    params['iter'] = iter_log
                    params['Time'] = execution_time
                    params['data'] = data
                    params['SR'] = SR
                    params['R1'] = R_Align[0]
                    params['R2'] = R_Align[1]
                    params['R3'] = R_Align[2]
                    params['res'] = res
                    params['params'] = s
                    parameter_output(params)
                    imshow_flag = 0
                    if imshow_flag ==1:
                        show = [10, 20, 30]
                        # plt.figure(figsize=(11, 33))
                        plt.subplot(131)
                        plt.imshow(np.clip(np.stack((X_observation[:, :, show[0]].cpu().detach().numpy(),
                                                     X_observation[:, :, show[1]].cpu().detach().numpy(),
                                                     X_observation[:, :, show[2]].cpu().detach().numpy()), 2), 0, 1))
                        plt.title('Observed')
                        plt.subplot(132)
                        plt.imshow(np.clip(np.stack((X_Out_real[:, :, show[0]].cpu().detach().numpy(),
                                                     X_Out_real[:, :, show[1]].cpu().detach().numpy(),
                                                     X_Out_real[:, :, show[2]].cpu().detach().numpy()), 2), 0, 1))
                        plt.title('Recovered')

                        plt.subplot(133)
                        plt.imshow(np.clip(np.stack((XT[:, :, show[0]].cpu().detach().numpy(),
                                                     XT[:, :, show[1]].cpu().detach().numpy(),
                                                     XT[:, :, show[2]].cpu().detach().numpy()), 2), 0, 1))
                        plt.title('Ground truth')
                        plt.show()
