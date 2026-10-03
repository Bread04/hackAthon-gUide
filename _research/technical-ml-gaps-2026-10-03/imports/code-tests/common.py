import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("/home/user/hackAthon-gUide/08-datathon-handbook/07-worked-example/sample-readmissions.csv")
NUM = ["age","chronic_conditions","length_of_stay","n_medications","prior_admissions_12m","creatinine"]
X = df[NUM].fillna(df[NUM].median()).astype("float64"); y = df["readmitted_30d"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
