# neural_network.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("sos_dataset.csv")
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = to_categorical(y_enc, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Test loss/acc:", scores)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=le.classes_))
model.save("vitaguard_nn.h5")
# Save label encoder mapping
import joblib
joblib.dump(le, "label_encoder.pkl")
print("Saved vitaguard_nn.h5 and label_encoder.pkl")
