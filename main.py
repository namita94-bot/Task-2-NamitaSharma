import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("student.csv")
print(data)

X = data[["Hours", "Attendance"]]
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

new_data = pd.DataFrame([[6, 85]], columns=["Hours", "Attendance"])
result = model.predict(new_data)
print(result)