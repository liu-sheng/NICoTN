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
import scipy.io as sio
import torch.nn.init as init

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        os.makedirs(output_dir)  # 如果 result 文件夹不存在，则创建


    txt_file_path = os.path.join(output_dir, 'NICoTN_order4.txt')
    with open(txt_file_path, 'a') as txt_file:
        line = ', '.join([f"{key}: {value}" for key, value in params.items()])
        txt_file.write(line + "\n")


    df = pd.DataFrame([params])


    excel_file_path = os.path.join(output_dir, 'NICoTN_order4.xlsx')
    if not os.path.isfile(excel_file_path):
        df.to_excel(excel_file_path, index=False)
    else:
        with pd.ExcelWriter(excel_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)



class NICoTN(nn.Module):
    def __init__(self, Nway, RA, R0):
        super(NICoTN, self).__init__()
        N1, N2, N3,N4 = Nway
        RA_12 = RA[0,1]
        RA_13 = RA[0, 2]
        RA_14 = RA[0, 3]
        RA_23 = RA[1, 2]
        RA_24 = RA[1, 3]
        RA_34 = RA[2, 3]

        R0_12 = R0[0, 1]
        R0_13 = R0[0, 2]
        R0_14 = R0[0, 3]
        R0_23 = R0[1, 2]
        R0_24 = R0[1, 3]
        R0_34 = R0[2, 3]

        self.Nway = Nway
        self.RA = RA
        self.R0 = R0
        self.G1 = nn.Parameter(torch.rand(N1, R0_12,R0_13,R0_14, requires_grad=True))
        self.G2 = nn.Parameter(torch.rand(RA_12, N2, R0_23,R0_24, requires_grad=True))
        self.G3 = nn.Parameter(torch.rand(RA_13,RA_23, N3, R0_34, requires_grad=True))
        self.G4 = nn.Parameter(torch.rand(RA_14, RA_24, RA_34, N4, requires_grad=True))

        self.R12_net = nn.Sequential(nn.Linear(R0_12, RA_12, bias=False),
                                    nn.GELU(),
                                    # nn.LeakyReLU(),
                                    )

        self.R13_23_net = nn.Sequential(nn.Linear(R0_13*R0_23, RA_13*RA_23, bias=False),
                                     nn.GELU(),
                                     # nn.LeakyReLU(),
                                     )

        self.R14_24_34_net = nn.Sequential(nn.Linear(R0_14 * R0_24 *R0_34, RA_14 * RA_24 *RA_34, bias=False),
                                        nn.GELU(),
                                        # nn.LeakyReLU(),
                                        )

        self.reset_parameters()
    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.G1.size(0))
        stdv = 1. / math.sqrt(256)
        for param in [self.G1, self.G2, self.G3, self.G4]:
            param.data.uniform_(-stdv, stdv)
        for net in [self.R12_net, self.R13_23_net,self.R14_24_34_net]:
            for module in net:
                if hasattr(module, 'weight') and module.weight is not None:
                    init.kaiming_normal_(module.weight, mode='fan_out')


    def forward(self):
        N1, N2, N3,N4 =self.Nway
        RA = self.RA
        R0 = self.R0
        # N4 = 1
        RA_12 = RA[0, 1]
        RA_13 = RA[0, 2]
        RA_14 = RA[0, 3]
        RA_23 = RA[1, 2]
        RA_24 = RA[1, 3]
        RA_34 = RA[2, 3]

        R0_12 = R0[0, 1]
        R0_13 = R0[0, 2]
        R0_14 = R0[0, 3]
        R0_23 = R0[1, 2]
        R0_24 = R0[1, 3]
        R0_34 = R0[2, 3]

        G1_reshaped = self.G1.permute(2, 3, 0,1).reshape(R0_13*R0_14*N1, R0_12)
        G1_reshaped = self.R12_net(G1_reshaped)

        G2_reshaped = self.G2.reshape(RA_12, N2 * R0_23*R0_24)
        M = torch.matmul(G1_reshaped, G2_reshaped)
        M = M.reshape(R0_13,R0_14,N1, N2, R0_23,R0_24)


        M_rotated = M.permute(5,1, 2, 3,4, 0).reshape(R0_24*R0_14*N1*N2, R0_13 * R0_23)
        M_rotated = self.R13_23_net(M_rotated)

        G3_rotated = self.G3.reshape(RA_13 * RA_23, N3*R0_34)
        M2 = torch.matmul(M_rotated, G3_rotated)
        M2 = M2.reshape(R0_24,R0_14,N1,N2,N3,R0_34 )

        M2_rotated = M2.permute( 2, 3, 4, 5,0,1).reshape(N1 * N2*N3, R0_24* R0_14 * R0_34)
        M2_rotated = self.R14_24_34_net(M2_rotated)
        G4_rotated = self.G4.reshape(RA_14 * RA_24*RA_34, N4)
        result_matrix = torch.matmul(M2_rotated, G4_rotated)

        result_tensor = result_matrix.reshape(N1, N2, N3,N4)
        return result_tensor



max_iter = 15001
for SR in ['10' ]:
    for data in ['akiyo']:
        for R12 in [15]:
            for R13 in [8]:
                for R14 in [10]:
                    for R24 in [R14]:
                        for weight_decay in [3e-7]:
                            file_name2 = 'video144176/' + data + SR + '.mat'
                            print("data =", data, "| SR =", SR)
                            mat = scipy.io.loadmat(file_name2)
                            X_np = mat["X"]
                            XT = torch.from_numpy(X_np).type(dtype).cuda()
                            X0_np = mat["X0"]
                            X0 = torch.from_numpy(X0_np).type(dtype).cuda()
                            mask_np = mat["mask"]
                            mask = torch.from_numpy(mask_np).type(dtype).cuda()
                            X_observation = XT * mask
                            Xture = XT.clone()

                            # opts['weight_decay']=weight_decay
                            R_Align = np.array([
                                [0, R12, R13, R14],
                                [0, 0, R13, R24],
                                [0, 0, 0, R13],
                                [0, 0, 0, 0]
                            ])

                            R_0 = np.array(R_Align) + 1



                            Nway = X_observation.shape
                            model = NICoTN(Nway,R_Align,R_0 ).type(dtype)
                            params = []
                            params += [x for x in model.parameters()]

                            s = sum([np.prod(list(p.size())) for p in params]);
                            print('Number of params: %d' % s)
                            optimizier = optim.Adam(params, lr=0.002, weight_decay=3e-7)

                            start_time = time.perf_counter()
                            # max_iter = opts['maxit']
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




                            end_time = time.perf_counter()
                            execution_time = end_time - start_time
                            # print(f"PSNR：{ps:.4f},iter：{iter},Time：{execution_time:.3f}s ")
                            print(f"PSNR：{psnr_log:.4f},iter：{iter},Time：{execution_time:.3f}s ")

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
                            params['R12'] = R_Align[0,1]
                            params['R13'] = R_Align[0, 2]
                            params['R14'] = R_Align[0, 3]
                            params['R23'] = R_Align[1,2]
                            params['R24'] = R_Align[1, 3]
                            params['R34'] = R_Align[2,3]
                            params['res'] = res.cpu().detach().numpy()
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
