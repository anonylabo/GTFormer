import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_provider.create_od_matix import create_od_matrix
from data_provider.data_loader import data_provider
from exp.exp_basic import Exp_Basic
from model import AR, GEML, LSTM, CrowdNet, GTFormer
from utils.dataset_utils import get_matrix_mapping, restore_od_matrix, to_2D_map
from utils.exp_utils import EarlyStopping


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            "GTFormer": GTFormer,
            "CrowdNet": CrowdNet,
            "GEML": GEML,
            "LSTM": LSTM,
            "AR": AR,
        }
        model = model_dict[self.args.model].Model(self.args).to(self.args.dtype)

        return model

    def train(self):
        dataset_directory = os.path.join(self.args.path + "/data/" + self.args.city + "_" + self.args.data_type + "/")
        od_matrix, _, _, _, param = create_od_matrix(dataset_directory, self.args)
        train_loader = data_provider("train", self.args, od_matrix)
        vali_loader = data_provider("val", self.args, od_matrix)

        del od_matrix

        if self.args.model in ["CrowdNet", "GEML"]:
            param = torch.tensor(param).to(self.args.dtype).to(self.device)

        path = os.path.join(self.args.path + f"/checkpoints_{self.args.model}/")
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.model == "CrowdNet":
            model_optim = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        else:
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        mse_criterion = nn.MSELoss()
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model_optim, gamma=0.96)

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if self.args.save_attention:
                    outputs, _, _ = self.model(batch_x, param)
                else:
                    outputs = self.model(batch_x, param)

                loss = mse_criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, param)

            my_lr_scheduler.step()

            print(
                "Epoch: {}, cost time: {}, Steps: {} | Train Loss: {} Vali Loss: {}".format(
                    epoch + 1, time.time() - epoch_time, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def vali(self, vali_loader, param):
        total_loss = []
        mse_criterion = nn.MSELoss()
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if self.args.save_attention:
                    outputs, _, _ = self.model(batch_x, param)
                else:
                    outputs = self.model(batch_x, param)

                loss = mse_criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        return total_loss

    def test(self, itr):
        dataset_directory = os.path.join(self.args.path + "/data/" + self.args.city + "_" + self.args.data_type + "/")
        od_matrix, min_tile_id, empty_indices, scaler, param = create_od_matrix(dataset_directory, self.args)
        test_loader = data_provider("test", self.args, od_matrix)

        del od_matrix

        if self.args.model in ["CrowdNet", "GEML"]:
            param = torch.tensor(param).to(self.args.dtype).to(self.device)

        self.model.load_state_dict(
            torch.load(os.path.join(self.args.path + f"/checkpoints_{self.args.model}/" + "checkpoint.pth"))
        )

        preds = []
        trues = []
        if self.args.save_attention:
            A_temporals = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if self.args.save_attention:
                    outputs, A_temporal, A_spatial = self.model(batch_x, param)
                    A_temporals.append(A_temporal.cpu().float().detach().numpy())
                else:
                    outputs = self.model(batch_x, param)

                preds.append(outputs.cpu().float().detach().numpy())
                trues.append(batch_y.cpu().float().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if self.args.save_attention:
            A_temporals = np.concatenate(A_temporals, axis=0)

        if self.args.model == "GTFormer" and self.args.spatial_mode == "AFT-full":
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
            trues = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

        # Error of OD flow
        od_rmse_test = np.sqrt(mean_squared_error(trues.flatten().reshape(-1, 1), preds.flatten().reshape(-1, 1)))
        od_mae_test = mean_absolute_error(trues.flatten().reshape(-1, 1), preds.flatten().reshape(-1, 1))

        print("OD flow Prediction")
        print("RMSE Error test: ", od_rmse_test)
        print("MAE Error test: ", od_mae_test)

        # Restore ODmatrirx
        matrix_mapping, x_max, y_max = get_matrix_mapping(self.args)
        trues = restore_od_matrix(trues, empty_indices)
        preds = restore_od_matrix(preds, empty_indices)

        # Conversion OD matrix to IO flow tensor
        trues_map, preds_map = to_2D_map(trues, preds, matrix_mapping, min_tile_id, x_max, y_max, self.args)

        # Erro of IO flow
        io_rmse_test = np.sqrt(
            mean_squared_error(trues_map.flatten().reshape(-1, 1), preds_map.flatten().reshape(-1, 1))
        )
        io_mae_test = mean_absolute_error(trues_map.flatten().reshape(-1, 1), preds_map.flatten().reshape(-1, 1))

        print("In-Out Flow Prediction")
        print("RMSE Error test: ", io_rmse_test)
        print("MAE Error test: ", io_mae_test)
        print("")
        print("")

        # Write results
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path + "/")
        if not os.path.exists(self.args.save_path + "/results.csv"):
            results = pd.DataFrame()
        else:
            results = pd.read_csv(self.args.save_path + "/results.csv")
        result = {}
        result["city"] = self.args.city
        result["data_type"] = self.args.data_type
        result["tile_size"] = self.args.tile_size
        result["model"] = self.args.model
        result["itr"] = itr
        if self.args.model == "GTFormer":
            result["temporal_mode"] = self.args.temporal_mode
            result["spatial_mode"] = self.args.spatial_mode
            result["use_only"] = self.args.use_only
        else:
            result["temporal_mode"] = "-"
            result["spatial_mode"] = "-"
            result["use_only"] = "-"
        result["OD_RMSE"] = od_rmse_test
        result["OD_MAE"] = od_mae_test
        result["IO_RMSE"] = io_rmse_test
        result["IO_MAE"] = io_mae_test

        result = pd.DataFrame(result, index=[len(results)])
        results = pd.concat([results, result], axis=0)
        results.to_csv(self.args.save_path + "/results.csv", index=False)

        # save predictions and true values
        save_path = self.args.save_path + f"/{self.args.city}_{self.args.data_type}/{self.args.model}/"
        if self.args.model == "GTFormer":
            if self.args.use_only == "temporal":
                mode = f"{self.args.use_only}_{self.args.temporal_mode}/"
            elif self.args.use_only == "spatial":
                mode = f"{self.args.use_only}_{self.args.spatial_mode}/"
            else:
                mode = f"{self.args.temporal_mode}_{self.args.spatial_mode}/"
            save_path = save_path + mode

        if self.args.save_attention:
            if not os.path.exists(save_path + f"/{itr}"):
                os.makedirs(save_path + f"/{itr}")
            np.save(save_path + f"/{itr}/" + "A_temporal.npy", A_temporals)
            if not self.args.spatial_mode == "AFT-simple":
                np.save(save_path + f"/{itr}/" + "A_spatial.npy", A_spatial.cpu().float().detach().numpy())

        if self.args.save_outputs:
            if not os.path.exists(save_path + f"/{itr}"):
                os.makedirs(save_path + f"/{itr}")
            np.save(save_path + f"/{itr}/" + "trues.npy", trues)
            np.save(save_path + f"/{itr}/" + "preds.npy", preds)
            np.save(save_path + f"/{itr}/" + "trues_map.npy", trues_map)
            np.save(save_path + f"/{itr}/" + "preds_map.npy", preds_map)

        return
