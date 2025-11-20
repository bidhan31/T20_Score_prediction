import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("t20i_info.csv")

# (your preprocessing steps here...)

# Final dataset
X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
transformer = ColumnTransformer([
    ('transformer', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team','bowling_team','city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', transformer),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=12, random_state=1))
])

# Train
pipe.fit(X_train, y_train)

# Save
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("✅ Model trained and saved as pipe.pkl")
