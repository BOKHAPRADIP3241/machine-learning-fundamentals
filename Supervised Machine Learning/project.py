import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#  Load Data 
students_data = pd.read_csv('student-por.csv')
print(students_data.head())

# Encode Categorical Columns 
le = LabelEncoder()
df = students_data.copy()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features & Target 
X = df.drop(columns=['G3'])          # drop final grade (target)
y = df['G3']

# Train / Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics 
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n========== Model Performance ==========")
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")
print("=======================================\n")

# Residuals 
residuals = y_test - y_pred

# PLOTS
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Student Performance Prediction — Linear Regression", fontsize=15, fontweight='bold')

# 1. Histogram of Actual Grades
axes[0, 0].hist(y, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
axes[0, 0].set_title("Distribution of Final Grades (G3)")
axes[0, 0].set_xlabel("Grade")
axes[0, 0].set_ylabel("Count")
axes[0, 0].axvline(y.mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean = {y.mean():.2f}")
axes[0, 0].legend()

# 2. Histogram of Residuals ──────────────────────────────────────────────────
axes[0, 1].hist(residuals, bins=20, color='salmon', edgecolor='white', alpha=0.85)
axes[0, 1].set_title("Residuals Distribution (Actual − Predicted)")
axes[0, 1].set_xlabel("Residual")
axes[0, 1].set_ylabel("Count")
axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=1.5, label="Zero Error")
axes[0, 1].legend()

# 3. Scatter — Actual vs Predicted ──────────────────────────────────────────
axes[1, 0].scatter(y_test, y_pred, color='mediumseagreen', alpha=0.6, edgecolors='white', linewidth=0.5)
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label="Perfect Prediction")
axes[1, 0].set_title("Actual vs Predicted Grades")
axes[1, 0].set_xlabel("Actual Grade (G3)")
axes[1, 0].set_ylabel("Predicted Grade")
axes[1, 0].legend()

# Annotate with metrics
axes[1, 0].text(0.05, 0.92,
                f"MAE={mae:.2f}  MSE={mse:.2f}\nRMSE={rmse:.2f}  R²={r2:.2f}",
                transform=axes[1, 0].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. Scatter — Predicted vs Residuals 
axes[1, 1].scatter(y_pred, residuals, color='mediumpurple', alpha=0.6, edgecolors='white', linewidth=0.5)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1, 1].set_title("Predicted vs Residuals")
axes[1, 1].set_xlabel("Predicted Grade")
axes[1, 1].set_ylabel("Residual")

plt.tight_layout()
plt.savefig('student_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as student_performance_analysis.png")