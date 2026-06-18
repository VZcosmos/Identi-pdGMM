from itertools import combinations
import torch
import numpy as np
import argparse
import invertible_network_utils
import random
import os
import encoders
import csv
from torch import nn
from evaluation import MCC, reorder, elementwise_r2
import cooper
import utils_latent as ut
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
import torch.distributions as D
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
import disentanglement_utils
import wandb
from utils_plot import plot_latent_histograms_untrimmed

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("device:", device)


def main():
    args, parser = parse_args()

    args.x_n = args.z_n
    args.nn = args.z_n

    if args.n_mixing_layer == 1:
        mix_type = 'linear'
        setting_name = 'Simple'
    else:
        mix_type = 'pw'
        setting_name = 'Complicate'
    # args.model_dir = os.path.join("Outputs", args.noise_type)
    args.model_dir = os.path.join(
        "Outputs",
        "Ours_BN", # model
        setting_name,
        f"{args.noise_type}"
    )
    os.makedirs(args.model_dir, exist_ok=True)
    heat_path_est = os.path.join(args.model_dir, 'heatmaps', 'est')
    heat_path_true = os.path.join(args.model_dir, 'heatmaps', 'true')
    heat_path_indep = os.path.join(args.model_dir, 'heatmaps', 'indep')

    if not os.path.exists(heat_path_est):
        os.makedirs(heat_path_est)
    if not os.path.exists(heat_path_true):
        os.makedirs(heat_path_true)
    if not os.path.exists(heat_path_indep):
        os.makedirs(heat_path_indep)

    mcc_scores = []
    mcc_indep_scores = []
    R2 = []
    MCC_stage1 = []

    for args.seed in args.seeds:
        wandb.init(
            project="thesis_final",
            name=f"Ours_BN_{setting_name}_{args.noise_type}_seed[{args.seed}]",
            config=vars(args),
            reinit=True
        )

        if wandb.config:
            for key, value in wandb.config.items():
                safe_key = key.replace("-", "_") 
                if hasattr(args, safe_key):
                    setattr(args, safe_key, value)

        wandb.define_metric("stage1/*", step_metric="stage1/step")
        wandb.define_metric("stage2/*", step_metric="stage2/step")

        # By default set the dimension of representations to be the same as z
        if args.nn == None:
            args.nn = args.z_n

        args.save_dir = os.path.join(args.model_dir,
                                     f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.x_n}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_theta{args.rotation}')
        load_slopes = args.load_f
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        results_file = os.path.join(args.save_dir, 'results.csv')

        B_file = os.path.join(args.save_dir, 'B.csv')
        W_file = os.path.join(args.save_dir, 'W.csv')
        mask_values_file = os.path.join(args.save_dir, 'mask_values.csv')
        Corr_file = os.path.join(args.save_dir, 'Corr_est.csv')
        Corr_true = os.path.join(args.save_dir, 'Corr_true.csv')
        heatmap_file_est = os.path.join(heat_path_est,
                                        f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.nn}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_theta{args.rotation}_Corr_heatmap.pdf')
        heatmap_file_true = os.path.join(heat_path_true,
                                         f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.nn}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_theta{args.rotation}_Corr_heatmap.pdf')
        heatmap_file_indep = os.path.join(heat_path_indep,
                                          f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_nn{args.nn}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_theta{args.rotation}_Corr_heatmap.pdf')
        if args.evaluate_redu:
            args.load_redu = os.path.join(args.save_dir, 'linearredu.pth')
            args.load_f = os.path.join(args.save_dir, 'f.npz')
            load_slopes = os.path.join(args.save_dir, 'slopes.csv')
        if args.evaluate or args.resume:
            args.load_g = os.path.join(args.save_dir, 'g.pth')
            args.load_f_hat = os.path.join(args.save_dir, 'f_hat.pth')
            if args.evaluate:
                args.n_steps = 1

        global device
        if args.no_cuda:
            device = "cpu"
            print("Using cpu")
        if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)

        if not args.evaluate:
            B_ori = ut.simulate_dag(args.z_n, args.z_n * args.DAG_dense, args.graph_type)
            np.savetxt(B_file, B_ori, delimiter=',')
            W_ori = ut.simulate_parameter(B_ori)
            np.savetxt(W_file, W_ori, delimiter=',')

            # def print_true_scm(W):
            #     d = W.shape[0]

            #     print("\n========== True linear SCM ==========")
            #     print("W matrix:")
            #     print(W)

            #     for j in range(d):
            #         parents = np.where(W[:, j] != 0)[0]

            #         terms = []
            #         for i in parents:
            #             terms.append(f"({W[i, j]:+.4f}) * z{i+1}")

            #         noise = f"ε{j+1}"
            #         if len(terms) == 0:
            #             equation = f"z{j+1} = {noise}"
            #         else:
            #             equation = f"z{j+1} = " + " + ".join(terms) + f" + {noise}"

            #         print(equation)

            #     print("=====================================\n")


            # print_true_scm(W_ori)
        else:
            B_ori = np.loadtxt(B_file, delimiter=',')
            W_ori = np.loadtxt(W_file, delimiter=',')

        if not args.evaluate:
            if args.noise_type == 'cauchy':
                mask_values = np.zeros(args.z_n)
                np.savetxt(mask_values_file, mask_values, delimiter=',')
            else:
                z = ut.simulate_linear_sem(W_ori, 5000, args.noise_type, args.noise_scale)
                Sigma_z = np.cov(z.T)
                Mean = np.mean(z, axis=0)
                sigma = np.sqrt(Sigma_z.diagonal())
                mask_values = args.distance * sigma + Mean
                np.savetxt(mask_values_file, mask_values, delimiter=',')
        else:
            mask_values = np.loadtxt(mask_values_file, delimiter=',')

        def generate_rhohot_batch(batch_size, vector_dimension, rho):
            if vector_dimension < 2:
                raise ValueError("Vector dimension must be at least 2.")

            # Create a batch array with zeros
            batch_data = np.zeros((batch_size, vector_dimension), dtype=int)

            for i in range(batch_size):
                # Generate indices and shuffle them
                indices = np.arange(vector_dimension)
                np.random.shuffle(indices)
                # Set the first rho*n indices to 1
                batch_data[i, indices[:rho]] = 1

            return batch_data # [batch_size, vector_dimension]

        if args.mask_dense == 1:
            ac = 1
        elif args.mask_dense == 50:
            ac = int(args.z_n / 2)
        elif args.mask_dense == 75:
            ac = int(args.z_n * 0.75)
        elif args.mask_dense == 100:
            ac = int(args.z_n)

        if args.mask_size > 1:
            # [args.mask_size * args.z_n, args.z_n]
            masks = generate_rhohot_batch(args.mask_size * args.z_n, args.z_n, ac)
        else:
            # when mask size is relatively low, artificially design masks to ensure sufficient index variability assumption
            masks = np.ones(args.z_n)
            masks = (np.tril(masks, -args.z_n - 1 + ac) + np.tril(np.triu(masks), ac - 1)).tolist()

        masks = np.unique(masks, axis=0)  # unifying the repeat masks
        num_unique_masks = masks.shape[0] # [num_unique_masks, args.z_n]
        masks = masks.tolist()


        def rotation_in_plane(n, i, j, theta):
            #Create an n-dimensional identity matrix and apply a 2D rotation in plane (i, j)
            R = np.eye(n)
            c, s = np.cos(theta), np.sin(theta)
            R[i, i] = c
            R[j, j] = c
            R[i, j] = -s
            R[j, i] = s
            return R

        def full_n_dim_rotation_matrix(n, angles=None):

            # Generate a full n-D rotation matrix by applying rotations in all 45 planes.


            R = np.eye(n)
            plane_indices = list(combinations(range(n), 2))  # All 45 plane index pairs

            if angles is None:
                # if angle is none, generate random roation
                angles = np.random.uniform(0, 2 * np.pi, len(plane_indices))
            else:
                angles = np.random.uniform(0, 2 * np.pi, len(plane_indices))*0+angles

            for (i, j), theta in zip(plane_indices, angles):
                R_ij = rotation_in_plane(n, i, j, theta)
                R = R_ij @ R  # Left-multiply to accumulate the rotation

            return R



        R_matrix = full_n_dim_rotation_matrix(n=args.z_n, angles=np.radians(args.rotation))

        def sample_whole_latent(size, indep=False, Mask=True, device=device, return_raw=False):

            if indep:
                Diag_B = ut.simulate_dag(args.z_n, 0, args.graph_type)
                z = ut.simulate_linear_sem(Diag_B, size, args.noise_type, args.noise_scale)
            else:
                z = ut.simulate_linear_sem(W_ori, size, args.noise_type, args.noise_scale)

            z_raw = torch.tensor(z).float().to(device)

            if not Mask:
                z = torch.tensor(z)
                z = z.float()
                z = z.to(device)
                return z

            z = torch.tensor(z) # [num of samples, z_n]

            # divide [num of samples, z_n] into k groups of [mini_batch, z_n], each group uses the same mask
            mini_batch = size // num_unique_masks
            for k in range(num_unique_masks):
                mask = np.array(masks[k])
                if k == num_unique_masks - 1:
                    z[k * mini_batch:, :] = z[k * mini_batch:, :] * mask
                else:

                    z[k * mini_batch:(k + 1) * mini_batch, :] = z[k * mini_batch:(k + 1) * mini_batch, :] * mask

            for i in range(args.z_n):
                z[z[:, i] == 0, i] = mask_values[i]

            # rotation to create non-standard basis
            z = z @ R_matrix

            z = z.float()
            z = z.to(device)

            if return_raw:
                return z, z_raw

            # combined_z = torch.cat([z, z_raw], dim=0)
            # return combined_z
            return z

        f = invertible_network_utils.get_decoder(args.x_n, args.z_n, args.seed, args.n_mixing_layer, args.load_f,
                                                 load_slopes, args.save_dir, smooth=False)


        def log_elementwise_r2_heatmap(r2_matrix, title="Element-wise Pairwise R2 Matrix"):
            """
            Plots the R2 matrix as a heatmap and logs it to W&B.
            """
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # R^2 is between 0 and 1, so vmin=0 and vmax=1 is perfect
            sns.heatmap(r2_matrix, annot=True, fmt=".2f", cmap="Blues", 
                        vmin=0, vmax=1, ax=ax)
            
            ax.set_title(title)
            ax.set_ylabel("Ground Truth Latents (z)")
            ax.set_xlabel("Learned Latents (hz)")
            
            # Log to wandb
            wandb.log({title: wandb.Image(fig)})
            
            # Close figure to prevent memory leaks
            plt.close(fig)

        class LinearRedu(nn.Module):
            def __init__(self):
                super(LinearRedu, self).__init__()

                # Encoder
                self.encoder = encoders.get_mlp(
                    n_in=args.x_n,
                    n_out=args.z_n,
                    layers=[

                        (args.nn) * 50,
                        (args.nn) * 100,
                        (args.nn) * 100,
                        (args.nn) * 50,

                    ],
                    output_normalization="bn",
                    
                    # linear=True
                )

                # Decoder
                self.decoder = encoders.get_mlp(
                    n_in=args.z_n,
                    n_out=args.x_n,
                    layers=[

                        (args.nn) * 50,
                        (args.nn) * 100,
                        (args.nn) * 100,
                        (args.nn) * 50,

                    ],
                    
                    # linear=True
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        # criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
        class CauchyNLLLoss(nn.Module):
            def __init__(self, gamma=1.0):
                super(CauchyNLLLoss, self).__init__()
                self.gamma = gamma

            def forward(self, x_hat, x):
                return torch.log1p(((x - x_hat) / self.gamma) ** 2).mean()
        # class SymlogMSELoss(nn.Module):
        #     def __init__(self):
        #         super(SymlogMSELoss, self).__init__()

        #     def symlog(self, tensor):
        #         return torch.sign(tensor) * torch.log1p(torch.abs(tensor))

        #     def forward(self, x_hat, x):
        #         # 1. Compress both target and prediction into symlog space
        #         sym_x = self.symlog(x)
        #         sym_x_hat = self.symlog(x_hat)
                
        #         # 2. Calculate standard MSE on the compressed values
        #         # This is essentially calculating the error of the orders-of-magnitude
        #         loss = torch.nn.functional.mse_loss(sym_x_hat, sym_x)
                
        #         return loss
        
        if args.noise_type == 'gauss':
            # criterion = nn.MSELoss()
            criterion = CauchyNLLLoss()
        elif args.noise_type == 'cauchy':
            # criterion = nn.MSELoss()
            criterion = CauchyNLLLoss()
        else:
            # criterion = nn.MSELoss()
            criterion = CauchyNLLLoss()
        linearredu = LinearRedu().to(device)

        optimizer = optim.Adam(linearredu.parameters(), lr=args.lr_redu_linear)

        def train_linearredu(model, criterion, optimizer, num_epochs=20):
            model.train()

            for epoch in range(num_epochs):
                total_loss = 0

                # Forward pass
                data_z = sample_whole_latent(size=args.batch_size) # [bs, z_n]

                data = f(data_z)

                # Forward pass
                reconstructed = model(data)
                z_hat = model.encoder(data)

                loss_rec = criterion(reconstructed, data)

                # dimension alignment
                # mvn = D.multivariate_normal.MultivariateNormal(torch.zeros(args.z_n).to(device),
                #                                                torch.eye(args.z_n).to(device))
                # loss_prior = mvn.log_prob(z_hat).mean()
                # loss = loss_rec - loss_prior
                # loss = loss_rec
                if args.loss_prior_stage1: # if use loss_prior for stage1
                    if args.noise_type == 'gauss': # mean=0, cov=I, indep
                        prior = D.multivariate_normal.MultivariateNormal(
                            torch.zeros(args.z_n).to(device),
                            torch.eye(args.z_n).to(device)
                        )
                        loss_prior = prior.log_prob(z_hat).mean()

                    elif args.noise_type == 'cauchy': # loc=0, scale=1, assume indep components, thus can sample from univa package, but also expect this loss to be bad for performance
                        prior = D.cauchy.Cauchy(
                            torch.zeros(args.z_n).to(device),
                            torch.ones(args.z_n).to(device)
                        )
                        loss_prior = prior.log_prob(z_hat).sum(dim=-1).mean()

                    elif args.noise_type == 'exp': # rate=1
                        prior = D.exponential.Exponential(
                            torch.ones(args.z_n).to(device)
                        )
                        loss_prior = prior.log_prob(z_hat).sum(dim=-1).mean()

                    elif args.noise_type == 'gumbel': # loc=0, scale=1
                        prior = D.gumbel.Gumbel(
                            torch.zeros(args.z_n).to(device),
                            torch.ones(args.z_n).to(device)
                        )
                        loss_prior = prior.log_prob(z_hat).sum(dim=-1).mean()

                    else:
                        raise ValueError(f"Unsupported noise_type: {args.noise_type}")
                    
                    loss = loss_rec - loss_prior
                else:
                    loss = loss_rec

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # log grad
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item()
                
                # clip grad
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                wandb.log({
                    "stage1/step": epoch,
                    "stage1/grad_norm": total_norm
                })

                optimizer.step()
                wandb.log({
                    "stage1/step": epoch,
                    "stage1/loss_rec": loss_rec.item(),
                    "stage1/x_hat_max_abs": reconstructed.abs().max().item(),
                    "stage1/x_max_abs": data.abs().max().item()
                })

                total_loss += loss.item()

                if epoch % 250 == 1:
                    print('loss_rec', loss_rec)
                    if args.loss_prior_stage1:
                        print('loss_prior', loss_prior)
                    mcc, cor_m = MCC(z_hat, data_z, args.z_n, False, args.use_floc)
                    mcc = mcc / args.z_n
                    mcc_s, cor_m_s = MCC(z_hat, data_z, args.z_n, True, args.use_floc)
                    mcc_s = mcc_s / args.z_n
                    # print('mcc:', mcc)

                    ###R2###
                    data_z = sample_whole_latent(size=args.batch_size)
                    data = f(data_z)
                    z_hat = linearredu.encoder(data)
                    data_z1 = sample_whole_latent(size=args.batch_size)
                    data1 = f(data_z1)
                    z_hat1 = linearredu.encoder(data1)

                    scaler_x = StandardScaler()
                    scaler_x_ro = RobustScaler()

                    x_train = scaler_x.fit_transform(data_z.detach().cpu().numpy())
                    x_test = scaler_x.fit_transform(data_z1.detach().cpu().numpy())
                    x_train_ro = scaler_x_ro.fit_transform(data_z.detach().cpu().numpy())
                    x_test_ro = scaler_x_ro.fit_transform(data_z1.detach().cpu().numpy())

                    scaler_y = StandardScaler()
                    scaler_y_ro = RobustScaler()

                    y_train = scaler_y.fit_transform(z_hat.detach().cpu().numpy())
                    y_test = scaler_y.fit_transform(z_hat1.detach().cpu().numpy())
                    y_train_ro = scaler_y_ro.fit_transform(z_hat.detach().cpu().numpy())
                    y_test_ro = scaler_y_ro.fit_transform(z_hat1.detach().cpu().numpy())

                    linear_model = disentanglement_utils.linear_disentanglement(y_train, x_train, train_mode=True)
                    linear_model_ro = disentanglement_utils.linear_disentanglement(y_train_ro, x_train_ro, train_mode=True)

                    # log_affine_matrix_to_wandb(linear_model, title="Affine Matrix (Standard Scaler)")
                    # log_affine_matrix_to_wandb(linear_model_ro, title="Affine Matrix (Robust Scaler)")

                    ((linear_disentanglement_score, linear_disentanglement_score_raw), _), _ = (
                        disentanglement_utils.linear_disentanglement(
                            y_test, x_test, mode="r2", model=linear_model
                        ))
                    ((linear_disentanglement_score_ro, linear_disentanglement_score_ro_raw), _), _ = (
                        disentanglement_utils.linear_disentanglement(
                            y_test_ro, x_test_ro, mode="r2", model=linear_model_ro
                        ))

                    wandb.log({
                        "stage1/step": epoch,
                        "stage1/z_hat_max_abs": z_hat.abs().max().item(),
                        "stage1/z_max_abs": data_z.abs().max().item(),
                        "stage1/mcc_pearson": mcc,
                        "stage1/mcc_spearman": mcc_s,
                        "stage1/r2_standard": linear_disentanglement_score,
                        "stage1/r2_robust": linear_disentanglement_score_ro
                    })

                    save_path = os.path.join(args.save_dir, 'linearredu.pth')
                    torch.save(model.state_dict(), save_path)

                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}")

        if args.load_redu is not None:
            linearredu.load_state_dict(torch.load(args.load_redu, map_location=torch.device(device)))

        if not args.evaluate_redu:
            train_linearredu(linearredu, criterion, optimizer, num_epochs=args.n_steps_redulinear)

        linearredu.eval()
        data_z, data_z_raw = sample_whole_latent(size=args.batch_size, return_raw=True)
        data = f(data_z)
        z_hat = linearredu.encoder(data)
        data_z1, data_z1_raw = sample_whole_latent(size=args.batch_size, return_raw=True)
        data1 = f(data_z1)
        z_hat1 = linearredu.encoder(data1)

        ### plot data distributions ###
        plot_latent_histograms_untrimmed(z_raw=data_z_raw, z_true=data_z, z_hat=z_hat, save_dir=args.save_dir, filename=f"stage1_r2_train_histograms_seed_{args.seed}_scale_{args.noise_scale}.png", max_dims=args.z_n)
        plot_latent_histograms_untrimmed(z_raw=data_z1_raw, z_true=data_z1, z_hat=z_hat1, save_dir=args.save_dir, filename=f"stage1_r2_test_histograms_seed_{args.seed}_scale_{args.noise_scale}.png", max_dims=args.z_n)

        for i in range(1): # number for testing

            # x_train = data_z
            scaler_x = StandardScaler()
            scaler_x_ro = RobustScaler()

            x_train = scaler_x.fit_transform(data_z.detach().cpu().numpy())
            x_test = scaler_x.fit_transform(data_z1.detach().cpu().numpy())
            x_train_ro = scaler_x_ro.fit_transform(data_z.detach().cpu().numpy())
            x_test_ro = scaler_x_ro.fit_transform(data_z1.detach().cpu().numpy())
            for j in range(1):
                # y_train = z_hat
                scaler_y = StandardScaler()
                scaler_y_ro = RobustScaler()

                y_train = scaler_y.fit_transform(z_hat.detach().cpu().numpy())
                y_test = scaler_y.fit_transform(z_hat1.detach().cpu().numpy())
                y_train_ro = scaler_y_ro.fit_transform(z_hat.detach().cpu().numpy())
                y_test_ro = scaler_y_ro.fit_transform(z_hat1.detach().cpu().numpy())

                linear_model = disentanglement_utils.linear_disentanglement(y_train, x_train, train_mode=True)
                linear_model_ro = disentanglement_utils.linear_disentanglement(y_train_ro, x_train_ro, train_mode=True)

                linear_model = disentanglement_utils.linear_disentanglement(y_train, x_train, train_mode=True)
                linear_model_ro = disentanglement_utils.linear_disentanglement(y_train_ro, x_train_ro, train_mode=True)

                ((linear_disentanglement_score, linear_disentanglement_score_raw), _), _ = (
                    disentanglement_utils.linear_disentanglement(
                        y_test, x_test, mode="r2", model=linear_model
                    ))
                ((linear_disentanglement_score_ro, linear_disentanglement_score_ro_raw), _), _ = (
                    disentanglement_utils.linear_disentanglement(
                        y_test_ro, x_test_ro, mode="r2", model=linear_model_ro
                    ))

                # print('after stage 1 R2:', linear_disentanglement_score)

        R2.append(linear_disentanglement_score)
        csv_path = os.path.join(args.model_dir, "stage1_r2.csv")
        fileobj = open(csv_path, "a+")
        # fileobj = open(args.model_dir + '_stage1.csv', 'a+') # stage1 R2 of one seed
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed,
               linear_disentanglement_score]
        writer.writerow(wri)
        fileobj.close()

        mcc, cor_m = MCC(z_hat, data_z, args.z_n, False, args.use_floc)
        mcc = mcc / args.z_n
        mcc_s, cor_m_s = MCC(z_hat, data_z, args.z_n, True, args.use_floc)
        mcc_s = mcc_s / args.z_n
        print('after stage 1 mcc:', mcc)
        MCC_stage1.append(mcc)
        csv_path = os.path.join(args.model_dir, "stage1_mcc.csv")
        fileobj = open(csv_path, "a+")
        # fileobj = open(args.model_dir + '_stage1_mcc.csv', 'a+') # stage1 MCC of one seed
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed,
               mcc]
        writer.writerow(wri)
        fileobj.close()

        wandb.log({
            "stage1/eval_r2_standard": linear_disentanglement_score,
            "stage1/eval_r2_robust": linear_disentanglement_score_ro,
            "stage1/eval_mcc_pearson": mcc,
            "stage1/eval_mcc_spearman": mcc_s
        })
        # print('Stage 1 R2 (Standard) per dimension: ', linear_disentanglement_score_raw.tolist())
        # print('Stage 1 R2 (Robust) per dimension: ', linear_disentanglement_score_ro_raw.tolist())

        class Constrained_DE(cooper.ConstrainedMinimizationProblem):
            def __init__(self):
                # self.criterion = nn.MSELoss(reduction='mean')
                if args.noise_type == 'gauss':
                    # self.criterion = nn.MSELoss(reduction='mean')
                    self.criterion = CauchyNLLLoss()
                elif args.noise_type == 'cauchy':
                    # self.criterion = nn.MSELoss(reduction='mean')
                    self.criterion = CauchyNLLLoss()
                else:
                    # self.criterion = nn.MSELoss(reduction='mean')
                    self.criterion = CauchyNLLLoss()

                super().__init__(is_constrained=True)

            # Vectorized version, should be faster
            def compute_loss(self, x, x_hat, z_hat):
                loss = self.criterion(x_hat, x)

                if not args.loss_prior_stage2:
                    return loss

                B, z_n = z_hat.shape
                K = len(masks)
                mini_batch = B // K

                # [K * mini_batch, z_n]
                z_hat = z_hat[: K * mini_batch]

                # [K, mini_batch, z_n]
                z_hat_g = z_hat.view(K, mini_batch, z_n)

                if args.noise_type == 'cauchy':
                    medians = torch.median(z_hat_g, dim=1).values  # [K, z_n]
                    abs_diffs = torch.abs(z_hat_g - medians.unsqueeze(1))  # [K, mini_batch, z_n]
                    mads = torch.median(abs_diffs, dim=1).values  # [K, z_n]

                    if args.oracle:
                        # 1. Load the true masks [K, z_n]
                        # mask=1 means active, mask=0 means masked
                        mask_prior = torch.tensor(masks, device=z_hat.device, dtype=torch.float32)
                        
                        # 2. Invert the mask: 1 for masked dimensions, 0 for active dimensions
                        penalty_mask = 1.0 - mask_prior
                        
                        # 3. Calculate loss ONLY on the masked dimensions
                        # Since your masked values are exactly 0, target median and mad are 0.
                        # Active dimensions are multiplied by 0, so the network is free to learn their true varying scales!
                        stat_loss = (
                            torch.mean(torch.abs(medians) * penalty_mask) +
                            torch.mean(torch.abs(mads) * penalty_mask)
                        )
                        
                        # Note: We use a multiplier (e.g., 10.0) to ensure the penalty is strong enough
                        stat_loss = stat_loss * 10.0 
                        
                    else:
                        # If you are NOT using the oracle, you cannot easily enforce MADs 
                        # because every dimension has a different scale.
                        # The safest option is to rely solely on your CauchyNLL reconstruction loss 
                        # and the L1 sparsity penalty you defined later in the code.
                        stat_loss = 0.0
                    
                else:
                    # mu_prior = 2: shape [1, 1, z_n]
                    mu_prior = torch.ones(1, 1, z_n, device=z_hat.device) * 2

                    diffs = z_hat_g - mu_prior  # [K, mini_batch, z_n]

                    var = torch.mean(diffs ** 2, dim=(1, 2), keepdim=True)  # [K,1,1]
                    std = torch.sqrt(var + 1e-9)

                    zscores = diffs / std  # [K, mini_batch, z_n]

                    skews = torch.mean(zscores ** 3, dim=1)       # [K, z_n]
                    kurtoses = torch.mean(zscores ** 4, dim=1)    # [K, z_n]

                    if args.noise_type == 'gauss':
                        target_skew, target_kurt = 0.0, 0.0
                    elif args.noise_type == 'exp':
                        target_skew, target_kurt = 2.0, 6.0
                    elif args.noise_type == 'gumbel':
                        target_skew, target_kurt = 1.14, 2.4
                    else:
                        raise ValueError

                    # instead of doing mean on [z_n] then sum k groups, do mean on [k, z_n]
                    stat_loss = (
                        torch.mean(torch.abs(skews - target_skew)) +
                        torch.mean(torch.abs(kurtoses - target_kurt))
                    )
                    # stat_loss = (
                    #     torch.mean(torch.abs(skews - target_skew), dim=1).sum() +
                    #     torch.mean(torch.abs(kurtoses - target_kurt), dim=1).sum()
                    # )

                return loss + stat_loss

            def closure(self, inputs):
                z_hat = g(inputs)
                x_hat = f_hat(z_hat)

                # loss = self.criterion(x_hat, inputs) ### Eq4 ###
                loss = self.compute_loss(inputs, x_hat, z_hat)

                # log sparse penalty
                # if args.noise_type == 'cauchy':
                #     sparse_penalty = torch.mean(torch.log1p(torch.abs(z_hat)))
                # else:
                #     sparse_penalty = torch.sum(torch.abs(z_hat)) / args.batch_size / args.z_n
                # ineq_defect = sparse_penalty - args.sparse_level
                
                # normal sparse penalty
                ineq_defect = torch.sum(torch.abs(z_hat)) / args.batch_size / args.z_n - args.sparse_level ### Eq5 ###


                return cooper.CMPState(loss=loss, ineq_defect=ineq_defect, eq_defect=None)

        g = encoders.get_mlp(
            n_in=args.x_n,
            n_out=args.nn,
            layers=[

                (args.nn) * 10,
                (args.nn) * 50,
                (args.nn) * 50,
                (args.nn) * 50,
                (args.nn) * 50,
                (args.nn) * 10,

            ],
            output_normalization="bn",
            
            linear=True
        )

        f_hat = encoders.get_mlp(
            n_in=(args.nn),
            n_out=args.x_n,
            layers=[

                (args.nn) * 10,
                (args.nn) * 50,
                (args.nn) * 50,
                (args.nn) * 50,
                (args.nn) * 50,
                (args.nn) * 10,

            ],
            
            linear=True
        )

        if args.load_g is not None:
            g.load_state_dict(torch.load(args.load_g, map_location=torch.device(device)))
            # g.load_state_dict(torch.load(args.load_g))

        if args.load_f_hat is not None:
            f_hat.load_state_dict(torch.load(args.load_f_hat, map_location=torch.device(device)))

        g = g.to(device)
        f_hat = f_hat.to(device)

        params = list(g.parameters()) + list(f_hat.parameters())

        ############### Constraint optimization with sparsity regularization ###################
        cmp_vade = Constrained_DE()
        formulation = cooper.LagrangianFormulation(cmp_vade, args.aug_lag_coefficient)
        primal_optimizer = cooper.optim.ExtraAdam(params, lr=args.lr) #loss
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=args.lr / 2) #sparse constraint

        coop_vade = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

        total_loss_values = []
        global_step = len(total_loss_values) + 1
        mcc_log = []
        step_log = []

        while (
                global_step <= args.n_steps
        ):
            if not args.evaluate:
                g.train()
                f_hat.train()

                # data = linearredu.encoder(f(z_masked))
                data = linearredu.encoder(f(sample_whole_latent(size=args.batch_size))) #z_hat_stage1
                data = data.clone().detach().requires_grad_(True).to(device)

                coop_vade.zero_grad()
                lagrangian = formulation.composite_objective(
                    cmp_vade.closure, data
                )
                formulation.custom_backward(lagrangian)
                # log grad
                total_norm = 0
                for p in params:
                    if p.grad is not None:
                        total_norm += p.grad.norm().item()
                # clip grad
                # torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

                coop_vade.step(cmp_vade.closure, data)

                with torch.no_grad():
                    z_hat_train = g(data) #z_hat_stage2?
                    x_hat_train = f_hat(z_hat_train) # reconstructed z_hat_stage1

                    loss_rec = cmp_vade.criterion(x_hat_train, data) # reconstructed z_hat_stage1 and z_hat_stage1?

                    if args.noise_type == 'cauchy':
                        sparse_penalty = torch.mean(torch.log1p(torch.abs(z_hat_train)))
                    else:
                        sparse_penalty = torch.sum(torch.abs(z_hat_train)) / args.batch_size / args.z_n

                    wandb.log({
                        "stage2/step": global_step,
                        "stage2/loss_rec": loss_rec.item(),
                        "stage2/z_hat_stage1_max_abs": data.abs().max().item(),
                        "stage2/recon_z_hat_stage1_hat_max_abs": x_hat_train.abs().max().item(),
                        "stage2/lagrangian": lagrangian.item(),
                        "stage2/sparse_penalty": sparse_penalty.item(),
                        "stage2/grad_norm": total_norm
                    })

            if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
                f_hat.eval()
                g.eval()

                z_disentanglement = sample_whole_latent(5000)

                hz_disentanglement = linearredu.encoder(f(z_disentanglement)) #z_hat_stage1

                hz_disentanglement = g(hz_disentanglement) #z_hat_stage2
                mcc, cor_m = MCC(z_disentanglement, hz_disentanglement, args.z_n, False, args.use_floc) # z and z_hat_stage2
                mcc = mcc / args.z_n
                mcc_s, cor_m_s = MCC(z_disentanglement, hz_disentanglement, args.z_n, True, args.use_floc) # z and z_hat_stage2
                mcc_s = mcc_s / args.z_n
                # mind = linear_sum_assignment(-1 * cor_m)[1]

                mean_r2, r2_matrix, optimal_matches = elementwise_r2(z_disentanglement, hz_disentanglement)
                log_elementwise_r2_heatmap(r2_matrix, title="Stage 2: Element-wise R2 Disentanglement")

                rank_mean_r2, rank_r2_matrix, rank_optimal_matches = elementwise_r2(z_disentanglement, hz_disentanglement, True)
                log_elementwise_r2_heatmap(rank_r2_matrix, title="Stage 2: Element-wise R2_Rank Disentanglement")

                wandb.log({
                    "stage2/step": global_step,
                    "stage2/mcc_pearson": mcc,
                    "stage2/mcc_spearman": mcc_s,
                    "stage2/r2": mean_r2,
                    "stage2/r2_rank": rank_mean_r2,
                    "stage2/z_max_abs": z_disentanglement.abs().max().item(),
                    "stage2/z_hat_stage2_max_abs": hz_disentanglement.abs().max().item()
                })

                if not args.evaluate:
                    fileobj = open(results_file, 'a+')
                    writer = csv.writer(fileobj)
                    wri = ['MCC', mcc]
                    writer.writerow(wri)
                    print(global_step)
                    print('estimate_mcc')
                    print(mcc)
                    mcc_log.append(mcc)
                    step_log.append(global_step)

                    save_path = os.path.join(args.save_dir, 'g.pth')
                    torch.save(g.state_dict(), save_path)
                    save_path = os.path.join(args.save_dir, 'f_hat.pth')
                    torch.save(f_hat.state_dict(), save_path)

            global_step += 1
            # if mcc > 0.995:
            #     break

        # draw MCC-step fig
        # plt.figure()
        # plt.plot(step_log, mcc_log)
        # plt.xlabel("Global Steps")
        # plt.ylabel("Estimated MCC")
        # plt.title("Stage 2: MCC vs Training Steps")

        # os.makedirs(args.model_dir, exist_ok=True)
        # fig_path = os.path.join(args.model_dir, "stage2_mcc_curve.png")
        # plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        # plt.close()

        # post-training eval
        z_true = sample_whole_latent(args.batch_size)
        x_batch = g(linearredu.encoder(f(z_true)))
        mcc, cor_m = MCC(z_true, x_batch, args.z_n, False, args.use_floc)
        mcc = mcc / args.z_n
        mcc_s, cor_m_s = MCC(z_true, x_batch, args.z_n, True, args.use_floc)
        mcc_s = mcc_s / args.z_n
        # print('After stage 2: ', mcc)
        mean_r2, r2_matrix, optimal_matches = elementwise_r2(z_true, x_batch)
        log_elementwise_r2_heatmap(r2_matrix, title="Stage 2: Element-wise R2 Disentanglement")

        rank_mean_r2, rank_r2_matrix, rank_optimal_matches = elementwise_r2(z_true, x_batch, True)
        log_elementwise_r2_heatmap(rank_r2_matrix, title="Stage 2: Element-wise R2_Rank Disentanglement")

        
        wandb.log({
            "stage2/eval_r2": mean_r2,
            "stage2/eval_mcc_pearson": mcc,
            "stage2/eval_mcc_spearman": mcc_s
        })
        # print('Stage 2 R2 (Standard) per dimension: ', stage2_r2_raw.tolist())
        # print('Stage 2 R2 (Robust) per dimension: ', stage2_r2_ro_raw.tolist())

        csv_path = os.path.join(args.model_dir, "stage2_mcc.csv")
        fileobj = open(csv_path, "a+")
        # fileobj = open(args.model_dir + '.csv', 'a+') # stage2 MCC of one seed
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed, mcc]
        writer.writerow(wri)
        fileobj.close()
        mcc_scores.append(mcc)

        mcc_true, cor_true = MCC(z_true, z_true, args.z_n, args.use_spearman, args.use_floc)
        np.savetxt(Corr_file, cor_m, delimiter=',')
        np.savetxt(Corr_true, cor_true, delimiter=',')

        # draw heatmaps for ground truth
        sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 900})
        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        gap1 = args.z_n // 10
        if gap1 == 0:
            gap1 = 1
        list1 = list(range(0, args.z_n, gap1)) + [args.z_n - 1]
        z_label = [''] * args.z_n
        for i in list1:
            kk = i + 1
            z_label[i] = r'$\mathbf{z}$' + f'$_{{{kk}}}$'
        cor_true = pd.DataFrame(cor_true, index=z_label, columns=z_label)
        sns.heatmap(cor_true, xticklabels=True, yticklabels=True, annot=False, cmap="Blues", ax=axes, cbar=False,
                    fmt=".2f", vmin=0, vmax=1)
        axes.set_title(
            fr'{mix_type} n={args.z_n} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense} $\theta$={args.rotation}',
            fontsize=15)
        plt.savefig(heatmap_file_true, format="pdf", bbox_inches='tight')

        # draw heatmaps for estimation
        cor_m = reorder(cor_m, args.z_n)
        sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 900})
        fig, axes = plt.subplots(1, 1, figsize=(4 * args.nn / args.z_n, 4))
        gap2 = args.nn // 10
        if gap2 == 0:
            gap2 = 1
        list2 = list(range(0, args.nn, gap1)) + [args.nn - 1]
        z_hat_label = [''] * args.nn
        for i in list2:
            kk = i + 1
            z_hat_label[i] = r'$\widehat{\mathbf{z}}$' + f'$_{{{kk}}}$'
        cor_m = pd.DataFrame(cor_m, index=z_label, columns=z_hat_label)
        sns.heatmap(cor_m, xticklabels=True, yticklabels=True, annot=False, cmap="Blues", ax=axes, cbar=False,
                    fmt=".2f", vmin=0, vmax=1)
        if args.nn == args.z_n:
            axes.set_title(
                fr'{mix_type} n={args.z_n} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense} $\theta$={args.rotation}',
                fontsize=15)
        else:
            axes.set_title(
                fr'{mix_type} n={args.z_n} nn={args.nn} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense} $\theta$={args.rotation}',
                fontsize=15)

        plt.savefig(heatmap_file_est, format="pdf", bbox_inches='tight')

        # testing on independent
        z_indep = sample_whole_latent(5000, indep=True)
        hz_indep = f(z_indep)
        hz_indep = g(linearredu.encoder(hz_indep))

        mcc_indep, cor_indep = MCC(z_indep, hz_indep, args.z_n, False, args.use_floc)
        mcc_indep = mcc_indep / args.z_n
        mcc_indep_s, cor_indep_s = MCC(z_indep, hz_indep, args.z_n, True, args.use_floc)
        mcc_indep_s = mcc_indep_s / args.z_n

        wandb.log({
            "stage2/eval_indep_mcc_pearson": mcc_indep,
            "stage2/eval_indep_mcc_spearman": mcc_indep_s
        })

        csv_path = os.path.join(args.model_dir, "stage2_mcc_indep.csv")
        fileobj = open(csv_path, "a+")
        # fileobj = open(args.model_dir + '_independent_test.csv', 'a+') # stage2 MCC_indep of one seed
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed,
               mcc_indep]
        writer.writerow(wri)
        fileobj.close()

        mcc_indep_scores.append(mcc_indep)

        # draw heatmaps for independent
        cor_indep = reorder(cor_indep, args.z_n)
        sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 900})
        fig, axes = plt.subplots(1, 1, figsize=(4 * args.nn / args.z_n, 4))
        cor_indep = pd.DataFrame(cor_indep, index=z_label, columns=z_hat_label)
        sns.heatmap(cor_indep, xticklabels=True, yticklabels=True, annot=False, cmap="Blues", ax=axes, cbar=False,
                    fmt=".2f", vmin=0, vmax=1)
        if args.nn == args.z_n:
            axes.set_title(
                fr'{mix_type} n={args.z_n} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense} $\theta$={args.rotation}',
                fontsize=15)
        else:
            axes.set_title(
                fr'{mix_type} n={args.z_n} nn={args.nn} m={args.n_mixing_layer} $\delta$={args.distance}$\sigma$ $\rho$={args.mask_dense} k={args.DAG_dense} $\theta$={args.rotation}',
                fontsize=15)

        plt.savefig(heatmap_file_indep, format="pdf", bbox_inches='tight')

        wandb.finish()
        print('finished one random seeds!')

    csv_path = os.path.join(args.model_dir, "SUM_stage2_mcc.csv")
    fileobj = open(csv_path, "a+")
    # fileobj = open('SUM_MCC_' + args.model_dir + '.csv', 'a+')
    writer = csv.writer(fileobj)
    wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(mcc_scores),
           np.std(mcc_scores)]
    writer.writerow(wri)
    fileobj.close()

    csv_path = os.path.join(args.model_dir, "SUM_stage1_r2.csv")
    fileobj = open(csv_path, "a+")
    # fileobj = open('SUM_R2_' + args.model_dir + '.csv', 'a+')
    writer = csv.writer(fileobj)
    wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(R2),
           np.std(R2)]
    writer.writerow(wri)
    fileobj.close()

    csv_path = os.path.join(args.model_dir, "SUM_stage2_mcc_indep.csv")
    fileobj = open(csv_path, "a+")
    # fileobj = open('SUM_INDE_' + args.model_dir + '.csv', 'a+')
    writer = csv.writer(fileobj)
    wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense,
           np.mean(mcc_indep_scores), np.std(mcc_indep_scores)]
    writer.writerow(wri)
    fileobj.close()

    csv_path = os.path.join(args.model_dir, "SUM_stage1_mcc.csv")
    fileobj = open(csv_path, "a+")
    # fileobj = open('SUM_MCC_stage1_' + args.model_dir + '.csv', 'a+')
    writer = csv.writer(fileobj)
    wri = [args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(MCC_stage1),
           np.std(MCC_stage1)]
    writer.writerow(wri)
    fileobj.close()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-n", type=int, default=10, choices=[3, 5, 10, 20, 40])
    parser.add_argument("--x-n", type=int, default=10, choices=[3, 5, 10, 20, 40])
    parser.add_argument("--rotation", type=float, default=0.0, choices=[0, 15, 30, 45])
    parser.add_argument("--distance", type=float, default=0.0, choices=[0, 1, 2, 3, 5, 10])
    parser.add_argument("--DAG-dense", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--mask-dense", type=int, default=50, choices=[1, 50, 75, 100])
    parser.add_argument("--n-mixing-layer", type=int, default=10, choices=[1, 3, 5, 10, 20])  # larger means more complicated piecewise
    parser.add_argument("--mask-size", type=int, default=5)
    parser.add_argument("--nn", type=int)
    parser.add_argument("--evaluate_redu", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--causal", action="store_false")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2])
    parser.add_argument("--scm-type", type=str, default='linear', choices=['linear', 'nonlinear'])
    parser.add_argument("--noise-type", type=str, default="gauss", choices=['gauss', 'cauchy', 'exp', 'gumbel'])
    parser.add_argument("--lr_redu_linear", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--n-log-steps", type=int, default=250)
    parser.add_argument("--n-steps", type=int, default=80001)
    parser.add_argument("--n-steps-redulinear", type=int, default=5001)
    parser.add_argument("--load-f", default=None)
    parser.add_argument("--load-g", default=None)
    parser.add_argument("--load-f-hat", default=None)
    parser.add_argument("--load-redu", default=None)
    parser.add_argument("--graph-type", type=str, default="ER")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--aug-lag-coefficient", type=float, default=0.00)
    parser.add_argument("--sparse-level", type=float, default=0.01)
    parser.add_argument("--loss_prior_stage1", action="store_true")
    parser.add_argument("--loss_prior_stage2", action="store_true")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--use_spearman", action="store_true")
    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--s1_gamma_nll", type=float, default=1.0)
    parser.add_argument("--s2_gamma_nll", type=float, default=1.0)
    parser.add_argument("--use_floc", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()

    return args, parser


if __name__ == "__main__":
    main()