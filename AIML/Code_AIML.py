import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from scipy.ndimage import median_filter
from datetime import datetime
import warnings
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # (1)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # (2,3)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR  # Import SVR from scikit-learn  #(4)
from sklearn.neighbors import KNeighborsRegressor # Import KNeighborsRegressor from scikit-learn  #(5)
from xgboost import XGBRegressor  # Import XGBRegressor from xgboost #(6)

# (7)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf

# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

#Global Variables
models = {}
choice = 0
data_records = [] # Make sure this initialization is outside the loop to avoid resetting it on each iteration


# Ignore all warnings
warnings.filterwarnings('ignore')


def extract_date_from_path(file_path):
    folder_name = os.path.basename(os.path.dirname(file_path))
    date = datetime.strptime(folder_name, "%Y%m%d")
    return date

def parse_mtl(mtl_path: str) -> Dict[str, str]:
    metadata = {}
    with open(mtl_path, 'r') as file:
        for line in file:
            parts = line.strip().split('=')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip('"')
                metadata[key] = value
    return metadata


# # Define data preprocessing functions
# def impute_missing_values(data, strategy='mean'):
#     imputer = SimpleImputer(strategy=strategy)
#     data_imputed = imputer.fit_transform(data)
#     return pd.DataFrame(data_imputed, columns=data.columns)

# def scale_features(data, scaler=StandardScaler()):
#     scaled_data = scaler.fit_transform(data)
#     return pd.DataFrame(scaled_data, columns=data.columns)


def sun_correction(raster_band: np.ndarray, sun_elevation_angle: float) -> np.ndarray:
    sun_elevation_rad = np.radians(sun_elevation_angle)
    standard_sun_elevation_rad = np.pi / 2
    correction_factor = np.sin(
        standard_sun_elevation_rad) / np.sin(sun_elevation_rad)
    corrected_band = raster_band * correction_factor
    return corrected_band


def process_images(mtl_path: str) -> Dict[str, Dict[str, float]]:
    folder_path = os.path.dirname(mtl_path)
    all_files = os.listdir(folder_path)
    band3_file = next(
        (file for file in all_files if "_B3.TIF" in file.upper()), None)
    band7_file = next(
        (file for file in all_files if "_B7.TIF" in file.upper()), None)

    metadata = parse_mtl(mtl_path)
    sun_elevation_angle = float(metadata.get("SUN_ELEVATION", 0))

    with rasterio.open(os.path.join(folder_path, band3_file)) as src:
        band3_data = src.read(1)
        band3_data = sun_correction(band3_data, sun_elevation_angle)

    with rasterio.open(os.path.join(folder_path, band7_file)) as src:
        band7_data = src.read(1)
        band7_data = sun_correction(band7_data, sun_elevation_angle)

    mndwi = (band3_data - band7_data) / \
        (band3_data + band7_data + np.finfo(float).eps)
    mndwi_filtered = median_filter(mndwi, size=(5, 5))
    binary_image = (mndwi_filtered > 0).astype(np.uint8)

    pixel_area = abs(src.transform[0] * src.transform[4])
    water_pixel_count = np.sum(binary_image)
    land_pixel_count = binary_image.size - water_pixel_count
    water_area = water_pixel_count * pixel_area
    land_area = land_pixel_count * pixel_area

    return {
        "binary_image": binary_image,
        "water_area": water_area,
        "land_area": land_area
    }


def process_and_show_images(mtl_path: str):
    results = process_images(mtl_path) # Capture the returned dictionary
    if results:
        # tabular_data = pd.DataFrame({'water_area': [results['water_area']], 'land_area': [results['land_area']]})
        # tabular_data = impute_missing_values(tabular_data)
        # tabular_data = scale_features(tabular_data)
        # binary_image = results["binary_image"]
        # plt.imshow(binary_image.squeeze(), cmap="gray")
        # plt.title(f"Processed Image for {mtl_path[-31:-15]}")
        # plt.colorbar()
        # plt.show()

        print(f"Results for {mtl_path}:")
        print(f"{mtl_path[-31:-15]}: ")
        print(f"Water Area: {results['water_area']} square meters")
        print(f"Land Area: {results['land_area']} square meters")
        print(f"Water Percentage: {(results['water_area']/(results['water_area']+results['land_area']))*100} %")
        print(f"Land Percentage: {(results['land_area']/(results['water_area']+results['land_area']))*100} %")
        print("-" * 50)
    
    date = extract_date_from_path(mtl_path)
    water_area = results['water_area']  # Access water area from the dictionary
    land_area = results['land_area']  # Access land area from the dictionary
    data_records.append(
        {"date": date, "water_area": water_area, "land_area": land_area})


mtl_paths = [
    # "D:\\DatasetUSGS\\2023\\20230124\\LC08_L2SP_147047_20230124_20230207_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20230225\\LC08_L2SP_147047_20230225_20230301_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20230321\\LC09_L2SP_147047_20230321_20230323_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20230422\\LC09_L2SP_147047_20230422_20230424_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20230524\\LC09_L2SP_147047_20230524_20230601_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20230625\\LC09_L2SP_147047_20230625_20230627_02_T2_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20230921\\LC08_L2SP_147047_20230921_20230926_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20231023\\LC08_L2SP_147047_20231023_20231031_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20231124\\LC08_L2SP_147047_20231124_20231129_02_T1_MTL.txt",
    # "D:\\DatasetUSGS\\2023\\20231226\\LC08_L2SP_147047_20231226_20240104_02_T1_MTL.txt",

    "D:\\DatasetUSGS\\2022\\20220129\\LC09_L2SP_147047_20220129_20230430_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20220222\\LC08_L2SP_147047_20220222_20220301_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20220326\\LC08_L2SP_147047_20220326_20220330_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20220427\\LC08_L2SP_147047_20220427_20220503_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20220529\\LC08_L2SP_147047_20220529_20220603_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20220630\\LC08_L2SP_147047_20220630_20220708_02_T2_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20220926\\LC09_L2SP_147047_20220926_20230327_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20221028\\LC09_L2SP_147047_20221028_20230324_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20221129\\LC09_L2SP_147047_20221129_20230320_02_T1_MTL.txt",
    "D:\\DatasetUSGS\\2022\\20221231\\LC09_L2SP_147047_20221231_20230315_02_T1_MTL.txt"

#     "D:\\DatasetUSGS\\2021\\20210118\\LC08_L2SP_147047_20210118_20210306_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20210219\\LC08_L2SP_147047_20210219_20210302_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20210323\\LC08_L2SP_147047_20210323_20210402_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20210424\\LC08_L2SP_147047_20210424_20210501_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20210526\\LC08_L2SP_147047_20210526_20210529_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20210627\\LC08_L2SP_147047_20210627_20210707_02_T2_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20210915\\LC08_L2SP_147047_20210915_20210924_02_T2_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20211017\\LC08_L2SP_147047_20211017_20211026_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20211118\\LC08_L2SP_147047_20211118_20211125_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2021\\20211228\\LC09_L2SP_147047_20211228_20230503_02_T1_MTL.txt",

#     "D:\\DatasetUSGS\\2020\\20200116\\LC08_L2SP_147047_20200116_20200823_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20200217\\LC08_L2SP_147047_20200217_20200823_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20200320\\LC08_L2SP_147047_20200320_20200822_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20200421\\LC08_L2SP_147047_20200421_20200822_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20200523\\LC08_L2SP_147047_20200523_20200820_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20200624\\LC08_L2SP_147047_20200624_20200823_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20200928\\LC08_L2SP_147047_20200928_20201006_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20201030\\LC08_L2SP_147047_20201030_20201106_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20201115\\LC08_L2SP_147047_20201115_20210315_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2020\\20201217\\LC08_L2SP_147047_20201217_20210309_02_T1_MTL.txt",

#     "D:\\DatasetUSGS\\2019\\20190129\\LC08_L2SP_147047_20190129_20200830_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20190214\\LC08_L2SP_147047_20190214_20200829_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20190318\\LC08_L2SP_147047_20190318_20200829_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20190419\\LC08_L2SP_147047_20190419_20200828_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20190521\\LC08_L2SP_147047_20190521_20200828_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20190622\\LC08_L2SP_147047_20190622_20200827_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20190926\\LC08_L2SP_147047_20190926_20200825_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20191028\\LC08_L2SP_147047_20191028_20200825_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20191129\\LC08_L2SP_147047_20191129_20200825_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2019\\20191231\\LC08_L2SP_147047_20191231_20200824_02_T1_MTL.txt",

#     "D:\\DatasetUSGS\\2018\\20180126\\LC08_L2SP_147047_20180126_20200902_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20180227\\LC08_L2SP_147047_20180227_20200902_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20180331\\LC08_L2SP_147047_20180331_20200901_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20180416\\LC08_L2SP_147047_20180416_20200901_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20180518\\LC08_L2SP_147047_20180518_20200901_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20180619\\LC08_L2SP_147047_20180619_20200831_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20180923\\LC08_L2SP_147047_20180923_20200830_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20181025\\LC08_L2SP_147047_20181025_20200830_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20181126\\LC08_L2SP_147047_20181126_20200830_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2018\\20181228\\LC08_L2SP_147047_20181228_20200830_02_T1_MTL.txt",

#     "D:\\DatasetUSGS\\2017\\20170123\\LC08_L2SP_147047_20170123_20200905_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20170224\\LC08_L2SP_147047_20170224_20200905_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20170328\\LC08_L2SP_147047_20170328_20200904_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20170429\\LC08_L2SP_147047_20170429_20200904_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20170531\\LC08_L2SP_147047_20170531_20200903_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20170616\\LC08_L2SP_147047_20170616_20200903_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20170920\\LC08_L2SP_147047_20170920_20200903_02_T2_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20171022\\LC08_L2SP_147047_20171022_20200902_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20171123\\LC08_L2SP_147047_20171123_20200902_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2017\\20171225\\LC08_L2SP_147047_20171225_20200902_02_T1_MTL.txt",

#     "D:\\DatasetUSGS\\2016\\20160121\\LC08_L2SP_147047_20160121_20200907_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20160222\\LC08_L2SP_147047_20160222_20200907_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20160325\\LC08_L2SP_147047_20160325_20200907_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20160426\\LC08_L2SP_147047_20160426_20200907_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20160528\\LC08_L2SP_147047_20160528_20200906_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20160629\\LC08_L2SP_147047_20160629_20200906_02_T2_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20160917\\LC08_L2SP_147047_20160917_20200906_02_T2_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20161019\\LC08_L2SP_147047_20161019_20200905_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20161120\\LC08_L2SP_147047_20161120_20200905_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2016\\20161222\\LC08_L2SP_147047_20161222_20200905_02_T1_MTL.txt",

#     "D:\\DatasetUSGS\\2015\\20150118\\LC08_L2SP_147047_20150118_20200910_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20150219\\LC08_L2SP_147047_20150219_20200909_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20150323\\LC08_L2SP_147047_20150323_20200909_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20150424\\LC08_L2SP_147047_20150424_20200909_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20150526\\LC08_L2SP_147047_20150526_20200909_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20150627\\LC08_L2SP_147047_20150627_20200909_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20150915\\LC08_L2SP_147047_20150915_20200908_02_T2_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20151017\\LC08_L2SP_147047_20151017_20200908_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20151118\\LC08_L2SP_147047_20151118_20200908_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2015\\20151220\\LC08_L2SP_147047_20151220_20200908_02_T1_MTL.txt",

#     "D:\\DatasetUSGS\\2014\\20140131\\LC08_L2SP_147047_20140131_20200912_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20140216\\LC08_L2SP_147047_20140216_20200911_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20140320\\LC08_L2SP_147047_20140320_20200911_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20140421\\LC08_L2SP_147047_20140421_20200911_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20140523\\LC08_L2SP_147047_20140523_20200911_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20140624\\LC08_L2SP_147047_20140624_20200911_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20140928\\LC08_L2SP_147047_20140928_20200910_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20141030\\LC08_L2SP_147047_20141030_20200910_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20141124\\LC08_L2SP_146047_20141124_20200910_02_T1_MTL.txt",
#     "D:\\DatasetUSGS\\2014\\20141217\\LC08_L2SP_147047_20141217_20200910_02_T1_MTL.txt"
]

def forecast(df, target):
    global choice
    X = df[['date_ordinal']]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        if choice == 7:
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=100, verbose=0)
            predictions = model.predict(X_test).flatten()
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"{name} - MSE: {mse}, R²: {r2}")

        future_dates = pd.date_range(
            start=df['date'].max(), periods=4, freq='M')[1:]
        future_ordinal = future_dates.map(datetime.toordinal)

        if choice == 7:
            future_predictions = model.predict(
                future_ordinal.to_numpy().reshape(-1, 1)).flatten()
        else:
            future_predictions = model.predict(
                future_ordinal.to_numpy().reshape(-1, 1))

        formatted_predictions = [f"{float(prediction):.1f} square meters" for prediction in future_predictions]
        print(f"Future predictions for {target} with {name}: {', '.join(formatted_predictions)}")
        print("-" * 50)

def main():
    global choice
    print("Starting")
    print("\n1.Linear Regression\n2.Random Forest\n3.Gradient Boosting\n4.Support Vector Machine\n5.K-Nearest Neighbors\n6.XGBoost\n7.Neural Network\n99.All\n100.Exit")
    choice = int(input(f"Enter Your Choice for Predicting: "))
    if (choice == 1):
        print("\nApplying Linear Regression\n")
        models["Linear Regression"] = LinearRegression()
    elif (choice == 2):
        print("\nApplying Random Forest\n")
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=100, random_state=42)
    elif (choice == 3):
        print("\nApplying Gradient Boosting\n")
        models["Gradient Boosting"] = GradientBoostingRegressor(
            n_estimators=100, random_state=42)
    elif (choice == 4):
        print("\nApplying Support Vector Machine\n")
        models["Support Vector Machine"] = SVR(kernel='rbf')
    elif (choice == 5):
        print("\nApplying K-Nearest Neighbors\n")
        models["K-Nearest Neighbors"] = KNeighborsRegressor(n_neighbors=5)
    elif (choice == 6):
        print("\nApplying XGBoost\n")
        models["XGBoost"] = XGBRegressor()
    elif (choice == 7):
        print("\nApplying Neural Network\n")
        models["Neural Network"] = Sequential(
            [Dense(10, activation='relu', input_shape=(1,)), Dense(1)])
    elif(choice==99):
        print("\nApplying Linear Regression\n")
        models["Linear Regression"] = LinearRegression()

        print("Applying Random Forest\n")
        models["Random Forest"] = RandomForestRegressor(
            n_estimators=100, random_state=42)

        print("Applying Gradient Boosting\n")
        models["Gradient Boosting"] = GradientBoostingRegressor(
            n_estimators=100, random_state=42)

        print("Applying Support Vector Machine\n")
        models["Support Vector Machine"] = SVR(kernel='rbf')

        print("Applying K-Nearest Neighbors\n")
        models["K-Nearest Neighbors"] = KNeighborsRegressor(n_neighbors=5)

        print("Applying XGBoost\n")
        models["XGBoost"] = XGBRegressor()

        print("Applying Neural Network\n")
        models["Neural Network"] = Sequential(
            [Dense(10, activation='relu', input_shape=(1,)), Dense(1)])
    elif(choice==100):
        print("Exiting")
        return
    else:
        print("Invalid Choice")
        return

    for mtl_path in mtl_paths:
        process_and_show_images(mtl_path)

    df = pd.DataFrame(data_records)
    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())


    # Call the forecast function with the dataset and target variable
    forecast(df, 'water_area')
    forecast(df, 'land_area')
    print("Done")


main()
