from process import *

# --- Model ----
# split data into features and target 

X = df.drop("deposit", axis=1)
y = df["deposit"]

# and make train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

one_hot_enc = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", drop="first"),
    cat_cols),
    remainder="passthrough")

# Obs.: O OneHot Encoder é um algoritmo que transforma variáveis categóricas em variáveis numéricas.

X_train = one_hot_enc.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=one_hot_enc.get_feature_names_out())

X_test = pd.DataFrame(one_hot_enc.transform(X_test), columns=one_hot_enc.get_feature_names_out())

model = LGBMClassifier() #  cria um modelo de classificação LightGBM
model.fit(X_train, y_train) # treina o modelo

# --- Evaluation ----
y_pred = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="g");

#print(f"Specificity: {specificity}")
#print("Confusion Matrix:")

# --- Save Model ----
# Specify the file path where you want to save the pickle file
file_path = "model.pkl"

# Save the model as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(model, f)

file_path = "ohe.pkl"

# Save the OneHotEncoder as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(one_hot_enc, f)
