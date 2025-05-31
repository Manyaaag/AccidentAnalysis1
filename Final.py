# config
testing_mode = True      # True for fast debugging

# imports
import warnings; warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, confusion_matrix, ConfusionMatrixDisplay)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency

# loading datasets
BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets"

def load_csv(name):
    df = pd.read_csv(DATA / name, keep_default_na=False)
    if testing_mode and len(df) > 5_000:
        df = df.sample(5_000, random_state=42)
        print(f"[testing_mode] sampled 5 000 rows from {name}")
    return df

person   = load_csv("person.csv")
vehicle  = load_csv("filtered_vehicle.csv")
accident = load_csv("accident.csv")

# merging datasets
occupants = person[person["ROAD_USER_TYPE_DESC"].isin(["Drivers","Passengers"])]
person_vehicle  = occupants.merge(
        vehicle[[
            "ACCIDENT_NO","VEHICLE_ID","VEHICLE_TYPE_DESC",
            'NO_OF_WHEELS','NO_OF_CYLINDERS','SEATING_CAPACITY',
            'TARE_WEIGHT','TOTAL_NO_OCCUPANTS',
            'VEHICLE_YEAR_MANUF','VEHICLE_BODY_STYLE','VEHICLE_MAKE'
        ]],
        on=["ACCIDENT_NO","VEHICLE_ID"], how="inner")
person_vehicle_accident = person_vehicle.merge(accident[["ACCIDENT_NO","SPEED_ZONE"]], on="ACCIDENT_NO", how="left")
if testing_mode and len(person_vehicle_accident) > 5_000:
    person_vehicle_accident = person_vehicle_accident.sample(5_000, random_state=42)

# mapping
seatbelt_map = {
    1.0:"Belted",4.0:"Belted",
    2.0:"Unbelted",5.0:"Unbelted",8.0:"Unbelted",
    3.0:"Other",6.0:"Other",
    7.0:"Unknown",9.0:"Unknown","":"Unknown"
}
person_vehicle_accident["BeltCat"] = person_vehicle_accident["HELMET_BELT_WORN"].map(seatbelt_map)
person_vehicle_accident = person_vehicle_accident[person_vehicle_accident["BeltCat"] != "Other"]
person_vehicle_accident = person_vehicle_accident[person_vehicle_accident["BeltCat"] != "Unknown"]

# map to Front/Rear/Other
person_vehicle_accident["SEATING_POSITION"] = person_vehicle_accident["SEATING_POSITION"].replace(
    {'NK':"Other", 'NA':"Other", '':"Other"}
)
# map to Front/Rear/Other
seating_position_map = {
    'D':'Front','LF':'Front','CF':'Front','PL':'Front',
    'RR':'Rear','CR':'Rear','LR':'Rear','OR':'Rear'
}
# map else to other
person_vehicle_accident["SeatCat"] = person_vehicle_accident["SEATING_POSITION"].replace(seating_position_map).fillna("Other")

person_vehicle_accident = person_vehicle_accident[person_vehicle_accident["INJ_LEVEL"].isin([1,2,3,4])].copy()

person_vehicle_accident = person_vehicle_accident[person_vehicle_accident["SeatCat"] != "Other"]

# vehicle clustering: from assignment 1 task3
vehicle_features = ['NO_OF_WHEELS','NO_OF_CYLINDERS','SEATING_CAPACITY',
             'TARE_WEIGHT','TOTAL_NO_OCCUPANTS']
grouped = (vehicle
           .groupby(['VEHICLE_YEAR_MANUF','VEHICLE_BODY_STYLE','VEHICLE_MAKE'])[vehicle_features]
           .mean().reset_index())
X_scaled = MinMaxScaler().fit_transform(grouped[vehicle_features])

# Elbow curve
sse, k_range = [], range(1,11)
for k in k_range:
    sse.append(KMeans(n_clusters=k,n_init=10,random_state=42)
               .fit(X_scaled).inertia_)
plt.figure(figsize=(7,4))
plt.plot(k_range, sse, marker="o"); plt.xticks(k_range)
plt.xlabel("K"); plt.ylabel("SSE"); plt.title("Elbow – vehicle clustering")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(BASE/"elbow_vehicle.png"); plt.close()

K_CHOSEN = 3
kmeans_final = KMeans(n_clusters=K_CHOSEN,n_init=10,random_state=42)
grouped['VehCluster'] = kmeans_final.fit_predict(X_scaled)

person_vehicle_accident = person_vehicle_accident.merge(
    grouped[['VEHICLE_YEAR_MANUF','VEHICLE_BODY_STYLE','VEHICLE_MAKE','VehCluster']],
    on=['VEHICLE_YEAR_MANUF','VEHICLE_BODY_STYLE','VEHICLE_MAKE'],
    how='left'
)

# figures
all_charts = BASE / "all_charts"; all_charts.mkdir(exist_ok=True)

def make_charts(df, category, title, fname, order=None, drop=None):
    created_table = df.groupby(['INJ_LEVEL', category]).size().unstack(fill_value=0)
    if drop:
        created_table = created_table.drop(columns=[c for c in drop if c in created_table.columns], errors='ignore')
    if order:
        created_table = created_table[order]
    # bar
    fig,axes=plt.subplots(1, created_table.shape[1], figsize=(4*created_table.shape[1],4), sharey=True)
    for ax,col in zip(axes, created_table.columns):
        created_table[col].plot(kind="bar",ax=ax)
        ax.set_title(col)
    fig.suptitle(f"{title} – bar"); plt.tight_layout()
    fig.savefig(all_charts/f"{fname}_bar.png"); plt.close()
    # pie
    fig,axes=plt.subplots(1, created_table.shape[1], figsize=(4*created_table.shape[1],4))
    for ax,col in zip(axes, created_table.columns):
        ax.pie(created_table[col], labels=created_table.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(col)
    fig.suptitle(f"{title} – pie"); plt.tight_layout()
    fig.savefig(all_charts/f"{fname}_pie.png"); plt.close()

# vehicle cluster chart
make_charts(person_vehicle_accident, "VehCluster", "Injury by vehicle cluster", "cluster")

# seating position chart
make_charts(
    person_vehicle_accident,
    "SeatCat",
    "Injury by seating position",
    "seat",
    order=["Front","Rear"],
    drop=None
)

# seatbelt usage chart
make_charts(
    person_vehicle_accident,
    "BeltCat",
    "Injury by seatbelt usage",
    "belt usage",
    order=["Belted","Unbelted"],
    drop=None
)

# MI & chi square & cramers v calculations
mi_dataframe = person_vehicle_accident[['INJ_LEVEL','BeltCat','SeatCat','VEHICLE_TYPE_DESC']]
X_mi = pd.get_dummies(mi_dataframe[['BeltCat','SeatCat','VEHICLE_TYPE_DESC']], drop_first=True)
y_mi = mi_dataframe['INJ_LEVEL']
mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=True, random_state=42)
mi_df = pd.DataFrame({'Feature': X_mi.columns, 'MI Score': mi_scores})
mi_df['Parent'] = mi_df['Feature'].str.split('_').str[0]
mi_agg_norm = (mi_df.groupby('Parent')['MI Score'].sum() / np.log(4)).round(3)
print("\n=== Normalised MI (0–1 scale) ===")
print(mi_agg_norm)

def cramers_v(cm):
    chi2,_,_,_ = chi2_contingency(cm)
    n = cm.sum().sum(); r,k = cm.shape
    return np.sqrt(chi2 / (n*(min(r,k)-1)))

for var in ['BeltCat','SeatCat','VEHICLE_TYPE_DESC']:
    table = pd.crosstab(person_vehicle_accident[var], person_vehicle_accident['INJ_LEVEL'])
    chi2, p, _, _ = chi2_contingency(table)
    print(f"\n{var}: chi2={chi2:.1f}  p={p:.4g}  Cramér’s V={cramers_v(table):.3f}")

# supervised learning models
FEATURES = ['BeltCat','SeatCat','VEHICLE_TYPE_DESC','VehCluster','AGE_GROUP','SEX','SPEED_ZONE']
TARGET   = 'INJ_LEVEL'

train_df, test_df = train_test_split(
    person_vehicle_accident[FEATURES+[TARGET]],
    test_size=0.2, stratify=person_vehicle_accident[TARGET], random_state=42
)

preprocessing = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)],
    remainder="drop"
)

# Decision Tree
decision_tree = Pipeline([
    ("prep", preprocessing),
    ("clf", DecisionTreeClassifier(
        criterion="entropy",
        class_weight="balanced",
        min_samples_leaf=3,
        random_state=42
    ))
])
decision_tree.fit(train_df[FEATURES], train_df[TARGET])
y_dt = decision_tree.predict(test_df[FEATURES])

# K-NN
knn = Pipeline([
    ("prep", preprocessing),
    ("clf", KNeighborsClassifier(n_neighbors=3, n_jobs=-1))
])
knn.fit(train_df[FEATURES], train_df[TARGET])
y_knn = knn.predict(test_df[FEATURES])

print("\nDecisionTree macro-F1:", f1_score(test_df[TARGET], y_dt, average="macro"))
print("K-NN macro-F1:", f1_score(test_df[TARGET], y_knn, average="macro"))

# Confusion matrix
confusionmatrix_dt  = confusion_matrix(test_df[TARGET], y_dt,  labels=[1,2,3,4])
confusionmatrix_knn = confusion_matrix(test_df[TARGET], y_knn, labels=[1,2,3,4])
fig, axes = plt.subplots(1, 2, figsize=(10,4))
ConfusionMatrixDisplay(confusionmatrix_dt,  display_labels=[1,2,3,4]).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Decision Tree")
ConfusionMatrixDisplay(confusionmatrix_knn, display_labels=[1,2,3,4]).plot(ax=axes[1], colorbar=False)
axes[1].set_title("K-NN")
plt.tight_layout(); plt.savefig(BASE/"confusionmatrix_comparison.png");  plt.close()
print("confusionmatrix_comparison.png saved")

# save CSV
person_vehicle_accident.to_csv(BASE/"merged_datasets.csv", index=False)
print(f"merged_datasets.csv ({len(person_vehicle_accident):,} rows) saved")