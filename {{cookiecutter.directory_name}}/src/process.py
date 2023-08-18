from libs import os, pd, pickle, plt, sns, msno, LGBMClassifier, classification_report, confusion_matrix, train_test_split, make_column_transformer, OneHotEncoder

# ---- LOAD DATA ----
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Trocar com o nome do arquivo que você quer processar
data_file = "bank.csv"

# Constrói o caminho absoluto até o arquivo 'bank.csv'
csv_path = os.path.join(main_directory, "data", data_file)

# Lê o arquivo CSV
df = pd.read_csv(csv_path)

dep_mapping = {"yes": 1, "no": 0}

# Convert the column to category and map the values
df["deposit"] = df["deposit"].astype("category").map(dep_mapping)

# Drop de colunas que não serão usadas
df = df.drop(labels = ["default", "contact", "day", "month", "pdays", "previous", "loan", "poutcome", "poutcome"], axis=1)

# ---- Check Missing Values ----
pd.DataFrame(df.isnull().sum()).T

cat_cols = ["job", "marital", "education", "housing"]
num_cols = ["age", "balance", "duration", "campaign"]


