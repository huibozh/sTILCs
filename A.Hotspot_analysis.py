
"""

Generate pixel-level hotspot maps for deep learning

The Getis-Ord Gi* statistic is a spatial statistical method used to identify clusters 
of high or low values (hotspots and coldspots) in spatial data. 
It helps analyze the spatial distribution of values and 
determine whether a point and its neighboring points exhibit significant clustering 
of high or low values compared to the entire dataset. 

The Gi* statistic measures whether a given point and its neighboring points have attribute values 
significantly higher or lower than the overall average of the study area. 
The calculation is based on the spatial relationships between each point and its neighbors.

@author: Huibo Zhang
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from pandas.core.frame import DataFrame
#from matplotlib.pyplot import MultipleLocator
from shapely.geometry import Point
from pysal.lib import weights
#from pysal.explore import esda
import geopandas as gpd
from esda import G_Local



for directory_path in glob.glob("/*"):
    label1 = directory_path.split("/")[-1]
    print(label1)

    # read data "_predict_results.csv" ,which represents prediction results for all patches
    file_path = directory_path + "//" + "*_predict_results.csv"
    for file in glob.glob(file_path):
        df = pd.read_csv(file)
        # add new column 'Hot_Spot'
        df['Hot_Spot'] = (df['Positive'] > df['Negative']) & (df['Positive'] > df['Other'])
        df['Hot_Spot'] = df['Hot_Spot'].astype(int)  # transfer to 1 and 0
    
        df['Patch_id'] = df['Patch_id'].str[36:]
        df[['X', 'Y', 'id', 'norm']] = df['Patch_id'].str.split('_', expand=True)
        df['X'] = pd.to_numeric(df['X'])
        df['Y'] = pd.to_numeric(df['Y'])
        # delete unnecessary columns
        columns_to_drop = ['norm', 'Patch_id', 'Positive', 'Negative', 'Other','id']
        data2 = df.drop(columns_to_drop, axis=1)
        data2.index.name = 'No'
        output_filename = df['id'].iloc[0] + "_hot_pot_data.csv"  
        data2_output_file = directory_path + "//" + output_filename
        
        data2.to_csv(data2_output_file, index=False)


os.makedirs("./TCGA_cohort", exist_ok=True)

base_path = os.path.abspath("./")
print(f"Base path: {base_path}")

for directory_path in glob.glob("/*"):
    print(f"Directory: {directory_path}")
    for file_path in glob.glob(os.path.join(directory_path, "*_hot_pot_data.csv")): # TCGA: hot_pot_data, 
    #for file_path in glob.glob(os.path.join(directory_path, "*_hot_pot_original_data.csv")): # XY: hot_pot_original_data
        print(f"Processing file: {file_path}")

        label1 = os.path.basename(directory_path)[:12] #TCGA_cohort

        df = pd.read_csv(file_path)

        # Convert the DataFrame to a GeoDataFrame, with geometry column based on X, Y coordinates
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.X, df.Y)])
        gdf['Hot_Spot'] = pd.to_numeric(gdf['Hot_Spot'], downcast='float').astype(np.float64)

        # Create a spatial weights matrix based on distance (threshold = 2), neighbors within 2 units are considered
        w = weights.DistanceBand.from_dataframe(gdf, threshold=2, binary=True, silence_warnings=True)
        w.transform = 'r'

        # Perform Getis-Ord Gi* local statistic calculation with 999 permutations for significance
        gistar = G_Local(gdf['Hot_Spot'], w, star=True, permutations=999)

        # Store the Gi* statistic (Z-scores) in the GeoDataFrame for each point
        gdf['Gi*'] = gistar.Zs

        # Calculate the min/max values of the X and Y coordinates for creating the grid
        x_min, x_max = gdf.geometry.x.min(), gdf.geometry.x.max()
        y_min, y_max = gdf.geometry.y.min(), gdf.geometry.y.max()

        # Define the grid size (1 unit), resolution is calculated based on the coordinate range
        grid_size = 1  # pixel-level
        x_res = int((x_max - x_min) / grid_size) + 1
        y_res = int((y_max - y_min) / grid_size) + 1
        grid = np.full((y_res, x_res), np.nan)

        # Populate the grid with the Gi* scores based on spatial coordinates
        for index, row in gdf.iterrows():
            x_idx = int((row.geometry.x - x_min) / grid_size)
            y_idx = int((row.geometry.y - y_min) / grid_size)
            grid[y_idx, x_idx] = row['Gi*']

        # Create a figure for the heatmap, adjusting size based on coordinate range
        plt.figure(figsize=((max(gdf.X)-min(gdf.X))/50, (max(gdf.Y)-min(gdf.Y))/50), dpi=200) #
        ax = plt.gca()  

        
        # Plot the grid as an image, using a cool-warm color map (red for hotspots, blue for coldspots)
        im = plt.imshow(grid, cmap='coolwarm', origin='lower', extent=[x_min, x_max, y_min, y_max])
        ax.set_aspect('equal')
        #plt.colorbar(im, label='Gi* Score')
        #plt.title(label1 + '_TIL Hot')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks([])
        plt.yticks([])

        # Hide the borders (spines) around the plot for a cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.savefig("TCGA_cohort//" + label1 + '.png', dpi=200)
        plt.close()

print("All files processed.")