import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv(
    r"C:/Users/gwm93/AppData/Local/Programs/Python/Python313/ML/Cognifyz/Task 1/Notebook/Data/Dataset.csv"
)
df.drop("Restaurant ID", axis=1, inplace=True)
df["Country Code"] = df["Country Code"].astype("object")

numerical_features = [c for c in df.columns if df[c].dtype != "object"]
categorical_features = [c for c in df.columns if df[c].dtype == "object"]

cat_label = [c for c in categorical_features if df[c].nunique() > 7]
cat_onehot = [c for c in categorical_features if c not in cat_label]

X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]


numerical_features.remove("Aggregate rating")


le = LabelEncoder()
for col in cat_label:
    X[col] = le.fit_transform(X[col])


oh = OneHotEncoder(drop="first", sparse_output=False)

preprocessor = ColumnTransformer(
    [
        ("oh", oh, cat_onehot),        
        ("sc", StandardScaler(), numerical_features),  
    ],
    remainder="passthrough",          
)


X_t = preprocessor.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_t, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    max_features=5,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_tr, y_tr)


def evaluate(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    return mae, rmse, r2

for split, X_split, y_split in [("Train", X_tr, y_tr)]:
    mae, rmse, r2 = evaluate(y_split, rf.predict(X_split))
    print("Random Forest Regressor\n")
    print("Train Data:\n\n")
    print("Mean Absolute Error:",mae)
    print("\nRoot Mean Squared Error:",rmse)
    print("\nR2_Score",r2)

print("=" * 40)

for split, X_split, y_split in [ ("Test", X_te, y_te)]:
    mae, rmse, r2 = evaluate(y_split, rf.predict(X_split))
    
    print("Test Data:\n\n")
    print("Mean Absolute Error:",mae)
    print("\nRoot Mean Squared Error:",rmse)
    print("\nR2_Score",r2)



oh_feature_names = preprocessor.named_transformers_["oh"].get_feature_names_out(cat_onehot)

oh_mapped = [name.split("_")[0] for name in oh_feature_names]


num_names = numerical_features


passthrough_names = cat_label  


all_mapped_names = (
    list(oh_mapped) +          
    list(num_names) +
    list(passthrough_names)
)
assert len(all_mapped_names) == len(rf.feature_importances_), "Name / importance length mismatch"


importances = (
    pd.Series(rf.feature_importances_, index=all_mapped_names)
      .groupby(level=0).sum()
      .sort_values(ascending=False)
)


TOP_K = 20
print(f"Top {TOP_K} original features by importance:\n")
for i, (feat, imp) in enumerate(importances.head(TOP_K).items(), 1):
    print(f"{i:>2}. {feat:<25} {imp:.4%}")
