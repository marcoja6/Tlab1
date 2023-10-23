#first_tlab


#Big disclaimer for the future, all the commands need to run together not individually.
#No need to hastag after using command once

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("data/raw/realestate.xlsx")

#The commands above import info and gave new values (names) to make it easier to write name

print(df[0:5])

#command above provides x amount columns
 
print(df.describe())

#provide stastical information of numerical dataset (range, mean, min, max, IQR)

# EX1.  sns.histplot(df["Y house price of unit area"])
#plt.show()


#provides a histograph, need both commands to display graph




#don't run mulitple graphing command at the sametime, it will led to the terminal to freeze

#TODO 

#sns.histplot(df["X2 house age"])
#plt.show()

#sns.boxplot(df["X2 house age"])
#plt.show()

#sns.histplot(df["X4 number of convenience stores"])
#plt.show()

#sns.boxplot(df["X4 number of convenience stores"])
#plt.show()

#Example of scatterplot

#sns.scatterplot(data=df, x="X2 house age", y="Y house price of unit area")
#plt.show()

#TODO

#sns.scatterplot(data=df, x="X4 number of convenience stores", y="Y house price of unit area")
#plt.show()


#Example of Heatmap

#selected_cols = ["X2 house age", "X4 number of convenience stores", "Y house price of unit area"]

#mask = np.triu(np.ones_like(df[selected_cols].corr(), dtype=bool))

#sns.heatmap(df[selected_cols].corr(), annot=True, mask=mask)
#plt.show()



#CLEANING DATA SECTION

missing_value = df.isna()
print(missing_value)

new_dict = {
    "X2 house age": "house_age", 
    "X3 distance to the nearest MRT station": "distance_to_mrt", 
    "X4 number of convenience stores": "num_convenience_stores",
    "X5 latitude": "lat",
    "X6 longitude": "long",
    "Y house price of unit area": "price_unit_area"
}



new_df = df.rename(columns = new_dict, )

print(new_df)

selected = ["lat", "long", "No"]

revised_df = new_df.drop( selected, axis = 1)
print(revised_df)


revised_df["distance_to_mrt"] = revised_df["distance_to_mrt"].str.strip("\"")

print(revised_df["distance_to_mrt"])

revised_df["distance_to_mrt"] = revised_df["distance_to_mrt"].astype

print(revised_df["distance_to_mrt"].astype)

print(revised_df.shape)

#Exploring Again

print(revised_df.describe())

df_outl = revised_df[revised_df.num_convenience_stores >= 0]
df_outl.shape

df_outl = df_outl[df_outl.house_age != 410.3]
df_outl.shape

#Plotting 

#sns.histplot(df_outl["price_unit_area"])
#plt.show()

#sns.histplot(df_outl["house_age"])
#plt.show()

#sns.histplot(df_outl["num_convenience_stores"])
#plt.show()

#sns.histplot(df_outl["distance_to_mrt"])
#plt.show()



#sns.scatterplot(data=df_outl, x="house_age", y="price_unit_area")
#plt.show()

#sns.scatterplot(data=df_outl, x="num_convenice_stores", y="price_unit_area")
#plt.show()

#sns.scatterplot(data=df_outl, x="distance_to_mrt", y="price_unit_area")
#plt.show()

#selected_cols = ["house_age", "num_convenience_stores", "price_unit_area"]

#mask = np.triu(np.ones_like(df_outl[selected_cols].corr(), dtype=np.bool_()))

#sns.heatmap(df_outl[selected_cols].corr(), annot=True, mask=mask)
#plt.show()


#COME BACK TO THIS

#df_outl.to_csv("/Users/marcoalba/Downloads/rental-pricing-explore-normal/data", index=False)


#Linear Regression



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import pickle

import pickle




echo "# Tlab1" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/marcoja6/Tlab1.git
git push -u origin main