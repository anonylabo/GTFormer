import ast
import json
import os
from operator import itemgetter
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkt


def load_dataset(city, data_type, tile_size, sample_time, dataset_directory):
    # Unzip dataset and convert to a dataframe
    print("data preprocessing...")
    if city == "NYC" and data_type == "Bike":
        zip_files = [f for f in os.listdir(dataset_directory) if f.endswith(".zip")]
        data = [pd.read_csv(dataset_directory + file_name) for file_name in zip_files]
        df = pd.concat(data)
        df = df.drop(
            [
                "tripduration",
                "start station id",
                "start station name",
                "end station id",
                "end station name",
                "bikeid",
                "usertype",
                "birth year",
                "gender",
            ],
            axis=1,
        )

    elif city == "DC" and data_type == "Bike":
        zip_files = [f for f in os.listdir(dataset_directory) if f.endswith(".zip")]
        data = [pd.read_csv(dataset_directory + file_name) for file_name in zip_files]
        df = pd.concat(data)

        # Since there is no latitude and longitude information of each station in the Captial Bikeshare dataset,
        # download station information and combine
        url = "https://gbfs.lyft.com/gbfs/2.3/dca-cabi/en/station_information.json"
        response = urlopen(url)
        station_information = json.load(response)

        lat = []
        lon = []
        short_name = []
        for i in range(len(station_information["data"]["stations"])):
            lat.append(station_information["data"]["stations"][i]["lat"])
            lon.append(station_information["data"]["stations"][i]["lon"])
            short_name.append(int(station_information["data"]["stations"][i]["short_name"]))

        station_df = pd.DataFrame({"lat": lat, "lon": lon, "short_name": short_name})
        station_df_sta = station_df.rename(
            {"lat": "start station latitude", "lon": "start station longitude", "short_name": "Start station number"},
            axis=1,
        )
        station_df_end = station_df.rename(
            {"lat": "end station latitude", "lon": "end station longitude", "short_name": "End station number"}, axis=1
        )

        df = pd.merge(df, station_df_sta, on="Start station number", how="left")
        df = pd.merge(df, station_df_end, on="End station number", how="left")

        df = df.drop(
            [
                "Duration",
                "Start station number",
                "Start station",
                "End station number",
                "End station",
                "Bike number",
                "Member type",
            ],
            axis=1,
        )
        df = df.rename({"Start date": "starttime", "End date": "stoptime"}, axis=1)
        df = df.dropna()

    elif city == "NYC" and data_type == "Taxi":
        csv_files = [f for f in os.listdir(dataset_directory) if f.startswith("yellow")]
        data = [pd.read_csv(dataset_directory + file_name) for file_name in csv_files]
        df = pd.concat(data)

        df = df.rename(
            {
                "tpep_pickup_datetime": "starttime",
                "tpep_dropoff_datetime": "stoptime",
                "pickup_longitude": "start station longitude",
                "pickup_latitude": "start station latitude",
                "dropoff_longitude": "end station longitude",
                "dropoff_latitude": "end station latitude",
            },
            axis=1,
        )
        df = df.drop(
            [
                "VendorID",
                "passenger_count",
                "trip_distance",
                "RatecodeID",
                "store_and_fwd_flag",
                "payment_type",
                "fare_amount",
                "extra",
                "mta_tax",
                "tip_amount",
                "tolls_amount",
                "improvement_surcharge",
                "total_amount",
            ],
            axis=1,
        )

    elif city == "BJ" and data_type == "Taxi":
        zip_files = [f for f in os.listdir(dataset_directory + "/taxi_log_2008_by_id/") if f.endswith(".txt")]
        data = [
            pd.read_table(file_name, sep=",", header=None, names=["ID", "datetime", "longitude", "latitude"])
            for file_name in zip_files
        ]
        data = list(map(up_off, data))
        df = pd.concat(data)
        df = df.rename(
            {
                "pickup_datetime": "starttime",
                "dropoff_datetime": "stoptime",
                "pickup_longitude": "start station longitude",
                "pickup_latitude": "start station latitude",
                "dropoff_longitude": "end station longitude",
                "dropoff_latitude": "end station latitude",
            },
            axis=1,
        )
        df = df.drop("ID", axis=1)

    print("load tessellation")
    # Load tile information
    tessellation = pd.read_csv(dataset_directory + f"Tessellation_{tile_size}_{city}.csv")
    tessellation["geometry"] = [shapely.wkt.loads(el) for el in tessellation.geometry]
    tessellation = gpd.GeoDataFrame(tessellation, geometry="geometry")

    # tessellation['position'] contains che position in a matrix
    # The origin is located on the bottom left corner
    # We need to locate it on the top left corner
    list_positions = np.array([ast.literal_eval(el) for el in tessellation["position"]])
    max_y = list_positions[:, 1].max()
    for i, y in enumerate(list_positions[:, 1]):
        list_positions[i, 1] = max_y - y
    tessellation["positions"] = list(sorted(list_positions, key=itemgetter(0)))

    print("dataframe preprocessing")
    # Filtering the dataset using the relevant features
    gdf_in = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["start station longitude"], df["start station latitude"]), crs="epsg:4326"
    )
    del df

    gdf_in_join = gpd.sjoin(gdf_in, tessellation)
    del gdf_in

    gdf_in_join = gdf_in_join[["starttime", "end station latitude", "end station longitude", "stoptime", "tile_ID"]]

    gdf_final = gpd.GeoDataFrame(
        gdf_in_join,
        geometry=gpd.points_from_xy(gdf_in_join["end station longitude"], gdf_in_join["end station latitude"]),
        crs="epsg:4326",
    )
    del gdf_in_join

    gdf_final_join = gpd.sjoin(gdf_final, tessellation)
    del gdf_final

    gdf_final_join = gdf_final_join[["starttime", "stoptime", "tile_ID_left", "tile_ID_right"]]

    gdf_final_join = gdf_final_join.rename(
        columns={"tile_ID_left": "tile_ID_origin", "tile_ID_right": "tile_ID_destination"}
    )
    gdf_final_join["starttime"] = pd.to_datetime(gdf_final_join["starttime"])
    gdf_final_join = gdf_final_join.sort_values(by="starttime")

    gdf_final_join["flow"] = 1
    gdf = gdf_final_join[["starttime", "tile_ID_origin", "tile_ID_destination", "flow"]]

    gdf_grouped = gdf.groupby(
        [pd.Grouper(key="starttime", freq=sample_time), "tile_ID_origin", "tile_ID_destination"]
    ).sum()

    # Saving geodataframe
    gdf_grouped.to_csv(dataset_directory + "df_grouped_" + tile_size + "_" + sample_time + ".csv")


def up_off(df):
    pickup = df
    pickup["datetime"] = pd.to_datetime(pickup["datetime"])
    pickup.rename(
        columns={"datetime": "pickup_datetime", "longitude": "pickup_longitude", "latitude": "pickup_latitude"},
        inplace=True,
    )

    dropoff = pickup.shift(-1)
    dropoff.rename(
        columns={
            "pickup_datetime": "dropoff_datetime",
            "pickup_longitude": "dropoff_longitude",
            "pickup_latitude": "dropoff_latitude",
        },
        inplace=True,
    )
    up_off = pd.concat([pickup.iloc[:-1], dropoff.iloc[:-1, 1:]], axis=1)
    return up_off
