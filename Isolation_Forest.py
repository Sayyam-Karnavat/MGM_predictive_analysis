import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle
import os


IF : IsolationForest

DIR = "Model"
model_name = "IF_model.pkl"

df = pd.read_csv("Data/sensor_data_normal+anomaly.csv")
unecessary_columns = ['datetime_stamp' , 'Machine_Id' , 'Plant_Id']
df = df.drop(columns=unecessary_columns)

sc = StandardScaler()
feature_names = df.columns
X = sc.fit_transform(df)


if not os.path.exists(f"{DIR}/{model_name}"):
    os.makedirs("Model")
    IF = IsolationForest(
        n_estimators=500,
        max_samples="auto",
        contamination=0.01,
        n_jobs=-1 # Uses all processor available to train the model
    )

    IF.fit(X=X)
    print("Model fitted ..")

    with open(f"{DIR}/{model_name}" , "wb") as f:
        pickle.dump(obj=IF , file=f)
    print("Model saved ..")


with open(f"{DIR}/{model_name}" , "rb") as f:
    IF = pickle.load(file=f)
print("Model loaded ..")



# The predict function will return 1 or -1 where 1 is normal and -1 is anomaly
anomaly_or_not = IF.predict(X)
df['anomaly_or_not'] = anomaly_or_not

'''
# Anomaly score

Isolation Forest converts average path length into an anomaly score:
    Score close to 1 → very anomalous
    Score close to 0 → very normal
'''
anomaly_scores = IF.score_samples(X)
df['anomaly_scores'] = anomaly_scores


# Save the results

df.to_csv("results.csv", index=False)

print("Results saved to 'results.csv'")














