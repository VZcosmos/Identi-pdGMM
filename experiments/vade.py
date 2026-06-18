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
import utils_latent as ut
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
import plotly.graph_objs as go
import torch.distributions as D
import torch.nn.functional as F
import itertools
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
import disentanglement_utils
from torch.nn.parameter import Parameter
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

    # Match Danru-style defaults / output organization.
    args.x_n = args.z_n
    args.nn = args.z_n
    args.use_bn = True

    if args.n_mixing_layer == 1:
        mix_type = 'linear'
        setting_name = 'Simple'
    else:
        mix_type = 'pw'
        setting_name = 'Complicate'

    args.model_dir = os.path.join(
        "Outputs",
        "VaDE",
        setting_name,
        f"{args.noise_type}"
    )
    os.makedirs(args.model_dir, exist_ok=True)
    heat_path_est = os.path.join(args.model_dir, 'heatmaps', 'est')
    heat_path_true = os.path.join(args.model_dir, 'heatmaps', 'true')
    heat_path_indep = os.path.join(args.model_dir, 'heatmaps', 'indep')
    c_path = os.path.join(args.model_dir, 'learn_graph', 'c')
    c_hat_path = os.path.join(args.model_dir, 'learn_graph', 'chat')

    if not os.path.exists(heat_path_est):
        os.makedirs(heat_path_est)
    if not os.path.exists(heat_path_true):
        os.makedirs(heat_path_true)
    if not os.path.exists(heat_path_indep):
        os.makedirs(heat_path_indep)
    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(c_hat_path):
        os.makedirs(c_hat_path)

    mcc_scores = []
    mcc_indep_scores = []
    R2 = []
    MCC_stage1 = []

    for args.seed in args.seeds:
        wandb.init(
            project="thesis_final",
            name=f"VaDE_{setting_name}_{args.noise_type}_seed[{args.seed}]",
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
        else:
            B_ori = np.loadtxt(B_file, delimiter=',')
            W_ori = np.loadtxt(W_file, delimiter=',')

        if not args.evaluate:
            if args.noise_type == 'cauchy':
                mask_values = np.zeros(args.z_n)
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
                # Set the first 5 indices to 1
                batch_data[i, indices[:rho]] = 1

            return batch_data

        if args.mask_dense == 1:
            ac = 1
        elif args.mask_dense == 50:
            ac = int(args.z_n / 2)
        elif args.mask_dense == 75:
            ac = int(args.z_n * 0.75)

        if args.mask_size > 1:
            masks = generate_rhohot_batch(args.mask_size * args.z_n, args.z_n, ac)
        else:
            # when mask size is relatively low, artificially design masks to ensure assumption 2.2
            masks = np.ones(args.z_n)
            masks = (np.tril(masks, -args.z_n - 1 + ac) + np.tril(np.triu(masks), ac - 1)).tolist()

        masks = np.unique(masks, axis=0)  # unifying the repeat masks
        num_unique_masks = masks.shape[0]
        masks = masks.tolist()

        # masks = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]




        def rotation_in_plane(n, i, j, theta):
            """
            Create an n-dimensional identity matrix and apply a 2D rotation in plane (i, j)
            """
            R = np.eye(n)
            c, s = np.cos(theta), np.sin(theta)
            R[i, i] = c
            R[j, j] = c
            R[i, j] = -s
            R[j, i] = s
            return R

        def full_n_dim_rotation_matrix(n, angles=None):
            """
            Generate a full 10D rotation matrix by applying rotations in all 45 planes.
            If angles is None, use random angles.
            """

            R = np.eye(n)
            plane_indices = list(combinations(range(n), 2))  # All 45 plane index pairs

            if angles is None:
                angles = np.random.uniform(0, 2 * np.pi, len(plane_indices))
            else:
                angles = np.random.uniform(0, 2 * np.pi, len(plane_indices))*0+angles

            for (i, j), theta in zip(plane_indices, angles):
                R_ij = rotation_in_plane(n, i, j, theta)
                R = R_ij @ R  # Left-multiply to accumulate the rotation

            return R



        R_matrix = full_n_dim_rotation_matrix(n=args.z_n, angles=np.radians(args.rotation))

        def sample_whole_latent(size, indep=False, Mask=True, device=device):

            if indep:
                Diag_B = ut.simulate_dag(args.z_n, 0, args.graph_type)
                z = ut.simulate_linear_sem(Diag_B, size, args.noise_type, args.noise_scale)
            else:
                z = ut.simulate_linear_sem(W_ori, size, args.noise_type, args.noise_scale)

            if not Mask:
                z = torch.tensor(z)
                z = z.float()
                z = z.to(device)
                return z

            z = torch.tensor(z)

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
            #z = torch.sigmoid(z)



            return z

        f = invertible_network_utils.get_decoder(args.x_n, args.z_n, args.seed, args.n_mixing_layer, args.load_f,
                                                 load_slopes, args.save_dir, smooth=False)

        def log_elementwise_r2_heatmap(r2_matrix, title="Element-wise Pairwise R2 Matrix"):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(r2_matrix, annot=True, fmt=".2f", cmap="Blues",
                        vmin=0, vmax=1, ax=ax)
            ax.set_title(title)
            ax.set_ylabel("Ground Truth Latents (z)")
            ax.set_xlabel("Learned Latents (hz)")
            wandb.log({title: wandb.Image(fig)})
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
                    output_normalization="bn" if args.use_bn else None,
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
                    #output_layer = "sigmoid",
                    # linear=True
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
        linearredu = LinearRedu().to(device)

        optimizer = optim.Adam(linearredu.parameters(), lr=args.lr_redu_linear)

        def train_linearredu(model, criterion, optimizer, num_epochs=20):
            model.train()

            for epoch in range(num_epochs):
                total_loss = 0

                # Forward pass
                data_z = sample_whole_latent(size=args.batch_size)

                data = f(data_z)

                # Forward pass
                reconstructed = model(data)
                z_hat = model.encoder(data)

                loss_rec = criterion(reconstructed, data)

                # dimension alignment
                mvn = D.multivariate_normal.MultivariateNormal(torch.zeros(args.z_n).to(device),
                                                               torch.eye(args.z_n).to(device))
                loss_prior = mvn.log_prob(z_hat).mean()
                loss = loss_rec - loss_prior
                # loss = loss_rec

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item()

                optimizer.step()

                wandb.log({
                    "stage1/step": epoch,
                    "stage1/loss_rec": loss_rec.item(),
                    "stage1/loss_prior": loss_prior.item(),
                    "stage1/loss": loss.item(),
                    "stage1/grad_norm": total_norm,
                    "stage1/x_hat_max_abs": reconstructed.abs().max().item(),
                    "stage1/x_max_abs": data.abs().max().item(),
                    "stage1/z_hat_max_abs": z_hat.abs().max().item(),
                    "stage1/z_max_abs": data_z.abs().max().item(),
                })

                total_loss += loss.item()

                if epoch % 250 == 1:
                    print('loss_rec', loss_rec)
                    print('loss_prior', loss_prior)
                    mcc, cor_m = MCC(z_hat, data_z, args.z_n, False, args.use_floc)
                    mcc = mcc / args.z_n
                    print('mcc:', mcc)
                    mcc_s, _ = MCC(z_hat, data_z, args.z_n, True, args.use_floc)
                    mcc_s = mcc_s / args.z_n

                    data_z_eval = sample_whole_latent(size=args.batch_size)
                    data_eval = f(data_z_eval)
                    z_hat_eval = model.encoder(data_eval)
                    mean_r2, _, _ = elementwise_r2(data_z_eval, z_hat_eval)
                    rank_mean_r2, _, _ = elementwise_r2(data_z_eval, z_hat_eval, True)

                    wandb.log({
                        "stage1/step": epoch,
                        "stage1/mcc_pearson": mcc,
                        "stage1/mcc_spearman": mcc_s,
                        "stage1/r2": mean_r2,
                        "stage1/r2_rank": rank_mean_r2,
                    })

                    save_path = os.path.join(args.save_dir, 'linearredu.pth')
                    torch.save(model.state_dict(), save_path)

                    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}")

        if args.load_redu is not None:
            linearredu.load_state_dict(torch.load(args.load_redu, map_location=torch.device(device)))

        if not args.evaluate_redu:
            train_linearredu(linearredu, criterion, optimizer, num_epochs=args.n_steps_redulinear)

        linearredu.eval()
        data_z = sample_whole_latent(size=args.batch_size)
        data = f(data_z)
        z_hat = linearredu.encoder(data)
        data_z1 = sample_whole_latent(size=args.batch_size)
        data1 = f(data_z1)
        z_hat1 = linearredu.encoder(data1)

        for i in range(1):

            x_train = data_z
            scaler_x = StandardScaler()
            x_train = scaler_x.fit_transform(x_train.detach().cpu().numpy())
            x_test = scaler_x.fit_transform(data_z1.detach().cpu().numpy())
            for j in range(1):
                # y_train = z_hat[:, j].reshape((-1, 1))
                y_train = z_hat
                scaler_y = StandardScaler()
                y_train = scaler_y.fit_transform(y_train.detach().cpu().numpy())
                y_test = scaler_y.fit_transform(z_hat1.detach().cpu().numpy())
                linear_model = disentanglement_utils.linear_disentanglement(y_train, x_train, train_mode=True)
                (linear_disentanglement_score, _,), _ = (
                    disentanglement_utils.linear_disentanglement(
                        y_test, x_test, mode="r2", model=linear_model
                    ))

                print('after stage 1 R2:', linear_disentanglement_score)

        R2.append(linear_disentanglement_score)
        fileobj = open(os.path.join(args.model_dir, 'stage1_r2.csv'), 'a+')
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed,
               linear_disentanglement_score]
        writer.writerow(wri)
        fileobj.close()

        mcc, cor_m = MCC(z_hat, data_z, args.z_n, False, args.use_floc)
        mcc = mcc / args.z_n
        print('after stage 1 mcc:', mcc)
        MCC_stage1.append(mcc)
        fileobj = open(os.path.join(args.model_dir, 'stage1_mcc.csv'), 'a+')
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed,
               mcc]
        writer.writerow(wri)
        fileobj.close()

        mcc_s, _ = MCC(z_hat, data_z, args.z_n, True, args.use_floc)
        mcc_s = mcc_s / args.z_n
        mean_r2_stage1, r2_matrix_stage1, _ = elementwise_r2(data_z, z_hat)
        rank_mean_r2_stage1, rank_r2_matrix_stage1, _ = elementwise_r2(data_z, z_hat, True)
        log_elementwise_r2_heatmap(r2_matrix_stage1, title="Stage 1: Element-wise R2 Disentanglement")
        log_elementwise_r2_heatmap(rank_r2_matrix_stage1, title="Stage 1: Element-wise R2_Rank Disentanglement")
        wandb.log({
            "stage1/eval_r2": mean_r2_stage1,
            "stage1/eval_r2_rank": rank_mean_r2_stage1,
            "stage1/eval_mcc_pearson": mcc,
            "stage1/eval_mcc_spearman": mcc_s,
        })

        ########### Stage 2 ############
        class VaDE(torch.nn.Module):
            def __init__(self, in_dim=args.x_n, latent_dim=args.z_n, n_classes=args.z_n*args.mask_size):
                super(VaDE, self).__init__()

                self.pi_prior = Parameter(torch.ones(n_classes) / n_classes)
                self.mu_prior = Parameter(torch.zeros(n_classes, latent_dim))
                self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim))

                self.fc1 = nn.Linear(in_dim, 512)  # Encoder
                self.fc2 = nn.Linear(512, 512)
                self.fc3 = nn.Linear(512, 2048)

                self.mu = nn.Linear(2048, latent_dim)  # Latent mu
                self.log_var = nn.Linear(2048, latent_dim)  # Latent logvar

                self.fc4 = nn.Linear(latent_dim, 2048)
                self.fc5 = nn.Linear(2048, 512)
                self.fc6 = nn.Linear(512, 512)
                self.fc7 = nn.Linear(512, in_dim)  # Decoder

            def encode(self, x):
                h = F.relu(self.fc1(x))
                h = F.relu(self.fc2(h))
                h = F.relu(self.fc3(h))
                return self.mu(h), self.log_var(h)

            def decode(self, z):
                h = F.relu(self.fc4(z))
                h = F.relu(self.fc5(h))
                h = F.relu(self.fc6(h))
                return F.sigmoid(self.fc7(h))

            def reparameterize(self, mu, log_var):
                std = torch.exp(log_var / 2)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                x_hat = self.decode(z)
                return x_hat, mu, log_var, z

        VaDE = VaDE().to(device)

        def compute_gamma(z, p_c):
            h = (z.unsqueeze(1) - VaDE.mu_prior).pow(2) / VaDE.log_var_prior.exp()
            h += VaDE.log_var_prior
            h += torch.Tensor([np.log(np.pi * 2)]).to(device)
            p_z_c = torch.exp(torch.log(p_c + 1e-5).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)) + 1e-5
            gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
            return gamma

        criterion = nn.MSELoss(reduction='sum')

        def compute_loss(x, x_hat, mu, log_var, z):
            p_c = VaDE.pi_prior
            gamma = compute_gamma(z, p_c)
            gamma = torch.clamp(gamma, min=1e-5)
            p_c = torch.clamp(p_c, min=1e-5)

            log_p_x_given_z = criterion(x_hat, x)
            h = log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - VaDE.mu_prior).pow(2)
            h = torch.sum(VaDE.log_var_prior + h / VaDE.log_var_prior.exp(), dim=2)
            log_p_z_given_c = 0.5 * torch.sum(gamma * h)
            log_p_c = torch.sum(gamma * torch.log(p_c + 1e-5))
            log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-5))
            log_q_z_given_x = 0.5 * torch.sum(1 + log_var)

            loss = log_p_x_given_z + log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x
            loss /= x.size(0)
            return loss



        total_loss = 0
        optimizer_vade = optim.Adam(VaDE.parameters(), lr=args.lr)
        global_step = 1

        while (
                global_step <= args.n_steps
        ):
            if not args.evaluate:
                VaDE.train()
                optimizer_vade.zero_grad()


                data = linearredu.encoder(f(sample_whole_latent(size=args.batch_size)))
                data = data.clone().detach().requires_grad_(True).to(device)


                x_hat, mu, log_var, z = VaDE(data)
                loss = compute_loss(data, x_hat, mu, log_var, z)
                loss.backward()

                total_norm = 0
                for p in VaDE.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item()

                optimizer_vade.step()
                total_loss += loss.item()

                with torch.no_grad():
                    loss_rec = criterion(x_hat, data) / data.size(0)
                    wandb.log({
                        "stage2/step": global_step,
                        "stage2/loss": loss.item(),
                        "stage2/loss_rec": loss_rec.item(),
                        "stage2/grad_norm": total_norm,
                        "stage2/z_hat_stage1_max_abs": data.abs().max().item(),
                        "stage2/recon_z_hat_stage1_hat_max_abs": x_hat.abs().max().item(),
                        "stage2/z_hat_stage2_max_abs": z.abs().max().item(),
                    })

            if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
                VaDE.eval()


                z_disentanglement = sample_whole_latent(4096)  # 4096

                hz_disentanglement = linearredu.encoder(f(z_disentanglement))

                hz_disentanglement,_ = VaDE.encode(hz_disentanglement)
                #print(hz_disentanglement)
                mcc, cor_m = MCC(z_disentanglement, hz_disentanglement, args.z_n, False, args.use_floc)
                mcc = mcc / args.z_n
                mcc_s, _ = MCC(z_disentanglement, hz_disentanglement, args.z_n, True, args.use_floc)
                mcc_s = mcc_s / args.z_n
                mean_r2, r2_matrix, _ = elementwise_r2(z_disentanglement, hz_disentanglement)
                rank_mean_r2, rank_r2_matrix, _ = elementwise_r2(z_disentanglement, hz_disentanglement, True)
                log_elementwise_r2_heatmap(r2_matrix, title="Stage 2: Element-wise R2 Disentanglement")
                log_elementwise_r2_heatmap(rank_r2_matrix, title="Stage 2: Element-wise R2_Rank Disentanglement")
                wandb.log({
                    "stage2/step": global_step,
                    "stage2/mcc_pearson": mcc,
                    "stage2/mcc_spearman": mcc_s,
                    "stage2/r2": mean_r2,
                    "stage2/r2_rank": rank_mean_r2,
                    "stage2/z_max_abs": z_disentanglement.abs().max().item(),
                    "stage2/z_hat_stage2_max_abs": hz_disentanglement.abs().max().item(),
                })
                mind = linear_sum_assignment(-1 * cor_m)[1]

                if not args.evaluate:
                    fileobj = open(results_file, 'a+')
                    writer = csv.writer(fileobj)
                    wri = ['MCC', mcc]
                    writer.writerow(wri)
                    print(global_step)
                    print('estimate_mcc')
                    print(mcc)

                    save_path = os.path.join(args.save_dir, 'VaDE.pth')
                    torch.save(VaDE.state_dict(), save_path)


            global_step += 1
            if mcc > 0.995:
                break



        fileobj = open(os.path.join(args.model_dir, 'stage2_mcc.csv'), 'a+')
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed, mcc]
        writer.writerow(wri)
        fileobj.close()
        mcc_scores.append(mcc)
        wandb.log({
            "stage2/eval_r2": mean_r2,
            "stage2/eval_r2_rank": rank_mean_r2,
            "stage2/eval_mcc_pearson": mcc,
            "stage2/eval_mcc_spearman": mcc_s,
        })

        z_true=z_disentanglement

        mcc_true, cor_true = MCC(z_true, z_true, args.z_n, args.use_spearman, args.use_floc)
        np.savetxt(Corr_file, cor_m, delimiter=',')
        np.savetxt(Corr_true, cor_true, delimiter=',')

        # draw heatmaps for truth
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

        # draw heatmaps for est
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

        # testing on indep
        z_indep = sample_whole_latent(5000, indep=True)
        hz_indep = f(z_indep)
        hz_indep,_ = VaDE.encode(linearredu.encoder(hz_indep))

        mcc_indep, cor_indep = MCC(z_indep, hz_indep, args.z_n, False, args.use_floc)
        mcc_indep = mcc_indep / args.z_n
        mcc_indep_s, _ = MCC(z_indep, hz_indep, args.z_n, True, args.use_floc)
        mcc_indep_s = mcc_indep_s / args.z_n
        wandb.log({
            "stage2/eval_indep_mcc_pearson": mcc_indep,
            "stage2/eval_indep_mcc_spearman": mcc_indep_s,
        })

        fileobj = open(os.path.join(args.model_dir, 'stage2_mcc_indep.csv'), 'a+')
        writer = csv.writer(fileobj)
        wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, args.seed,
               mcc_indep]
        writer.writerow(wri)
        fileobj.close()

        mcc_indep_scores.append(mcc_indep)

        # draw heatmaps for indep
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

        # store unmask z and z_hat for graph learning
        c = sample_whole_latent(size=5000, Mask=False)
        c_hat,_ = VaDE.encode(linearredu.encoder(f(c)))
        c_hat = c_hat.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        c_file = os.path.join(c_path,
                              f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_theta{args.rotation}')
        c_hat_file = os.path.join(c_hat_path,
                                  f'{mix_type}{args.n_mixing_layer}_d{args.distance}_n{args.z_n}_M{args.mask_dense}_G{args.DAG_dense}_rs{args.seed}_theta{args.rotation}')
        np.savetxt(c_hat_file + '.csv', c_hat[:, mind], delimiter=',')
        np.savetxt(c_file + '.csv', c, delimiter=',')
        print('finished one random seeds!')
        wandb.finish()

    fileobj = open(os.path.join(args.model_dir, 'SUM_stage2_mcc.csv'), 'a+')
    writer = csv.writer(fileobj)
    wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(mcc_scores),
           np.std(mcc_scores)]
    writer.writerow(wri)
    fileobj.close()

    fileobj = open(os.path.join(args.model_dir, 'SUM_stage1_r2.csv'), 'a+')
    writer = csv.writer(fileobj)
    wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(R2),
           np.std(R2)]
    writer.writerow(wri)
    fileobj.close()

    fileobj = open(os.path.join(args.model_dir, 'SUM_stage2_mcc_indep.csv'), 'a+')
    writer = csv.writer(fileobj)
    wri = [args.rotation, args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense,
           np.mean(mcc_indep_scores), np.std(mcc_indep_scores)]
    writer.writerow(wri)
    fileobj.close()

    fileobj = open(os.path.join(args.model_dir, 'SUM_stage1_mcc.csv'), 'a+')
    writer = csv.writer(fileobj)
    wri = [args.distance, args.n_mixing_layer, args.z_n, args.nn, args.mask_dense, args.DAG_dense, np.mean(MCC_stage1),
           np.std(MCC_stage1)]
    writer.writerow(wri)
    fileobj.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-n", type=int, default=5, choices=[3, 5, 10, 20, 40])
    parser.add_argument("--x-n", type=int, default=5, choices=[3, 5, 10, 20, 40])
    parser.add_argument("--rotation", type=float, default=0.0, choices=[0, 15, 30, 45])
    parser.add_argument("--distance", type=float, default=0.0, choices=[0, 1, 2, 3, 5, 10])
    parser.add_argument("--DAG-dense", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--mask-dense", type=int, default=50, choices=[1, 50, 75, 100])
    parser.add_argument("--n-mixing-layer", type=int, default=10,
                        choices=[1, 3, 10, 20])  # larger means more complicated piecewise
    parser.add_argument("--mask-size", type=int, default=5)
    parser.add_argument("--nn", type=int)
    parser.add_argument("--oracle", type=str, default='Given_masks', choices=['Given_masks', 'nonoracle'])

    parser.add_argument("--evaluate_redu", action='store_true')
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--causal", action="store_false")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2])

    parser.add_argument("--scm-type", type=str, default='linear', choices=['linear', 'nonlinear'])
    parser.add_argument("--noise-type", type=str, default="gauss", choices=['gauss', 'cauchy', 'exp', 'gumbel'])

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_redu_linear", type=float, default=1e-4)

    parser.add_argument("--batch-size", type=int, default=6144)
    parser.add_argument("--n-log-steps", type=int, default=250)
    parser.add_argument("--n-steps", type=int, default=5001)
    parser.add_argument("--n-steps-redulinear", type=int, default=5001)

    parser.add_argument("--load-f", default=None)
    parser.add_argument("--load-g", default=None)
    parser.add_argument("--load-f-hat", default=None)
    parser.add_argument("--load-nf", default=None)
    parser.add_argument("--load-redu", default=None)

    parser.add_argument("--graph-type", type=str, default="ER")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="")

    parser.add_argument("--aug-lag-coefficient", type=float, default=0.00)
    parser.add_argument("--sparse-level", type=float, default=0.01)

    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--use_spearman", action="store_true")
    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--use_floc", action="store_true")

    args = parser.parse_args()

    return args, parser


if __name__ == "__main__":
    main()