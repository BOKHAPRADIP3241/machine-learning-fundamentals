from sklearn.tree import DecisionTreeClassifier

X = [[7,2],[8,3],[9,8],[10,9]]
y = [0,0,1,1] #0=Apple, 1=Orange

model = DecisionTreeClassifier()
model.fit(X,y)

size = float(input("Enter the size of the fruit in cm: "))
shade = float(input("Enter the shade of the fruit(1-10): "))

result = model.predict([[size,shade]])[0]

if result == 0:
    print("The fruit is an Apple.")
else:
    print("The fruit is an Orange.")