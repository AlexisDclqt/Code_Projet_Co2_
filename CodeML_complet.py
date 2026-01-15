import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv(r"C:\Users\alexd\Desktop\WorkSpace\NoteBook_Jupyter\Projet_DA_DST_Co2\Data_Clean\Co2_sample_v3.csv")


data = data[["Constructeur","WLTP_poids","Co2_Emission(WLTP)","Type_Carburant","Puissance_KW","Fuel consumption","Pays"]]

data = data[data["Type_Carburant"] != "Electric"]


target = data["Co2_Emission(WLTP)"]
data = data.drop(columns="Co2_Emission(WLTP)")


corr_matrix = data.corr(numeric_only=True)

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()




num_features = make_column_selector(dtype_exclude=["category","object"])(data)
cat_features = make_column_selector(dtype_include=["category","object"])(data)

kf = KFold(n_splits = 10, shuffle=True, random_state=42)

num_prepocessor = Pipeline(
    steps = [
        ("impute",SimpleImputer(strategy="median")),
         ("scaler", StandardScaler())
    ]
)

cat_prepocessor = Pipeline(
    steps = [
        ("impute",SimpleImputer(strategy="most_frequent")),
         ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessors = ColumnTransformer(
    transformers = [
        ('num',num_prepocessor, num_features),
        ('cat',cat_prepocessor, cat_features)
        
    ]
)

model = Pipeline(
    steps = [
        ("preprocessors", preprocessors),
        ("linreg", LinearRegression())
    ]
)



mse_scores = cross_validate(model, data, target, cv = kf, scoring = "neg_mean_squared_error")
mean_mse = -mse_scores["test_score"].mean()


r2_scores = cross_validate(model, data, target, cv = kf, scoring = "r2", return_train_score=True)
test_mean_r2 = r2_scores["test_score"].mean()
train_mean_r2 = r2_scores["train_score"].mean()

print(f'MSE : {mean_mse}, R2 : {test_mean_r2}')
print(f'Train : {train_mean_r2}, Test : {test_mean_r2}')



train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=data,
    y=target,
    cv=kf,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)

test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_mean, label="Train R²")
plt.plot(train_sizes, test_mean, label="Validation R²")


plt.xlabel("Taille du jeu d'entraînement")
plt.ylabel("R²")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()



model_clean = model.fit(data,target)

train_preds = model_clean.predict(data)

plt.figure(figsize=(10,6))
plt.plot(target.values[:100], label="Vraies Emissions", marker='o')
plt.plot(train_preds[:100], label="Prédictions", marker='x')
plt.legend()
plt.title("Comparaison vraies vs prédites (extrait des 100 premières)")
plt.xlabel("Index")
plt.ylabel("Emissions Co2")
plt.grid(True)
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(target, train_preds, alpha=0.6)
plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)  
plt.xlabel("Valeurs réelles (Co2)")
plt.ylabel("Prédictions du modèle")
plt.title("Prédictions vs Réalité - Jeu d'entraînement")
plt.grid(True)
plt.show()
