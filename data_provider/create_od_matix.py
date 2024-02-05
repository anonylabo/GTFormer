import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.dataset_utils import get_normalized_adj


def create_od_matrix(dataset_directory, args):
    df = pd.read_csv(dataset_directory + "df_grouped_" + args.tile_size + "_" + args.sample_time + ".csv")
    tessellation = pd.read_csv(dataset_directory + "Tessellation_" + args.tile_size + "_" + args.city + ".csv")

    minites = {"60min": 60, "45min": 45, "30min": 30, "15min": 15}
    minite = minites[args.sample_time]

    # Calculate how many time intervals each time is counted from the beginning
    df["start"] = [df["starttime"][0] for _ in range(len(df))]
    df["start"] = pd.to_datetime(df["start"])
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["dif"] = df["starttime"] - df["start"]
    df["dif"] = df["dif"].dt.total_seconds().round() // (minite * 60)
    df["dif"] = df["dif"].astype("int")

    min_tile_id = min(df["tile_ID_origin"].min(), df["tile_ID_destination"].min())
    max_tile_id = max(int(df["tile_ID_origin"].max() + 1), int(df["tile_ID_destination"].max() + 1))
    df["tile_ID_origin"] -= min_tile_id
    df["tile_ID_destination"] -= min_tile_id

    # Create an empty ODmatrix
    _axis = max(int(df["tile_ID_origin"].max()) + 1, int(df["tile_ID_destination"].max()) + 1)
    od_matrix = np.zeros([df["dif"].max() + 1, _axis, _axis])

    # Substitute each flow
    for row in df.itertuples():
        od_matrix[row.dif, int(row.tile_ID_origin), int(row.tile_ID_destination)] = row.flow

    del df

    # The diagonal component is not regarded as flow, so it is set to 0
    for i in range(od_matrix.shape[0]):
        np.fill_diagonal(od_matrix[i, :, :], 0)

    # Remove origin-destination pairs whose flow is 0 at all times to make the calculation lighter
    od_sum = np.sum(od_matrix, axis=0)
    od_matrix = od_matrix[:, ~(od_sum == 0).all(1), :]
    od_matrix = od_matrix[:, :, ~(od_sum == 0).all(1)]

    if args.model == "GTFormer" and args.spatial_mode == "AFT-full":
        scaler = MinMaxScaler()
        od_matrix = scaler.fit_transform(od_matrix.reshape(-1, 1)).reshape(od_matrix.shape)

    # Get adjacency matrix for CrowdNet
    elif args.model == "CrowdNet":
        A = od_matrix.sum(0)
        A_hat = get_normalized_adj(A)

    elif args.model == "GEML":
        dis_matrix = np.zeros([max_tile_id - min_tile_id, max_tile_id - min_tile_id])
        tessellation["position"] = tessellation["position"].apply(lambda x: ast.literal_eval(x))
        for i in range(min_tile_id, max_tile_id):
            for j in range(min_tile_id, max_tile_id):
                i_pos = tessellation["position"][i]
                j_pos = tessellation["position"][j]
                dis_matrix[i - min_tile_id, j - min_tile_id] = (i_pos[0] - j_pos[0]) ** 2 + (i_pos[1] - j_pos[1]) ** 2

        dis_matrix = dis_matrix[~(od_sum == 0).all(1), :]
        dis_matrix = dis_matrix[:, ~(od_sum == 0).all(1)]

    # For restore ODmatrix
    empty_indices = [i for i, x in enumerate((od_sum == 0).all(1)) if x]

    if args.model == "GTFormer":
        if args.spatial_mode == "AFT-full":
            return od_matrix, min_tile_id, empty_indices, scaler, None
        else:
            return od_matrix, min_tile_id, empty_indices, None, None
    elif args.model == "CrowdNet":
        return od_matrix, min_tile_id, empty_indices, None, A_hat
    elif args.model == "GEML":
        return od_matrix, min_tile_id, empty_indices, None, dis_matrix
    else:
        return od_matrix, min_tile_id, empty_indices, None, None
