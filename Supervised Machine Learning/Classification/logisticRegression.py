from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

hours = float(input("Enter the number of hours studied: "))

result  = model.predict([[hours]])[0]

if result == 1 :
    print(f"Predicted result for {hours} hours of study: PASS")
else:
    print(f"Predicted result for {hours} hours of study: FAIL")

