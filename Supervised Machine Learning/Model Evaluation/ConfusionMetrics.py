from sklearn.metrics import confusion_matrix

y_true = [0, 1, 1, 0, 0, 1, 0, 0]
y_pred = [0, 1, 0, 1, 0, 1, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

"""
[[Tp TN
  FP FN]]
"""