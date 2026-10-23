from sklearn.tree import DecisionTreeClassifier

# Training examples
X = [[1], [2], [3], [4]]
y = ["A", "A", "B", "B"]

model = DecisionTreeClassifier()
model.fit(X, y)

print(model.predict([[3]]))