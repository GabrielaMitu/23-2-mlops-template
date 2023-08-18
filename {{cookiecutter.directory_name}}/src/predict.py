from train import *

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Constrói o caminho absoluto até o arquivo 'bank.csv'
model_pkl = os.path.join(main_directory, "notebooks", "model.pkl")
ohe_pkl = os.path.join(main_directory, "notebooks", "ohe.pkl")

bank_predict = os.path.join(main_directory, "data", "bank_predict.csv")


# Load the model from the pickle file
with open(model_pkl, "rb") as f:
    model = pickle.load(f)

# Load the OneHotEncoder from the pickle file
with open(ohe_pkl, "rb") as f:
    one_hot_enc = pickle.load(f)

# Read the data from the file
df_predict = pd.read_csv(bank_predict)

# Make predictions
y_pred = model.predict(one_hot_enc.transform(df_predict))

# Create a new column with the predictions
df_predict["y_pred"] = y_pred

#dep_mapping = {"yes": 1, "no": 0}
dep_mapping = {1: "yes", 0: "no"}

# Convert the column to category and map the values
df_predict["y_pred"] = df_predict["y_pred"].map(dep_mapping)

# Save the data to a new file
df_predict.to_csv(bank_predict, index=False)
