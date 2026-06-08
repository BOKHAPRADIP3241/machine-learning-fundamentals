from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [40,50,60,70,80]

model = LinearRegression()
model.fit(X, y)

hours = float(input("Enter the number of hours studied: "))

predicted_marks = model.predict([[hours]])

print(f"Predicted marks for {hours} hours of study: {predicted_marks}")
