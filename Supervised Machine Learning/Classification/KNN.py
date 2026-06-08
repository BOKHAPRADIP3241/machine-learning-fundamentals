from sklearn.neighbors import KNeighborsClassifier

x = [
    [180,7],[200,7.5],[250,8],[300,8.5],[330,9],[360,9.5]
]

# 0 for Apple and 1 for Orange260
y = [0,0,0,1,1,1]

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x, y)

weight = float(input("Enter the weight in grams: "))
size = float(input("Enter the size in centimeters: "))

prediction = model.predict([[weight,size]])[0]

if prediction == 1:
    print(f"The predicted class for weight {weight} grams and size {size} cm is: Orange")
else:    print(f"The predicted class for weight {weight} grams and size {size} cm is: Apple")