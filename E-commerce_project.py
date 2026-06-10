# ================================================================
#       E-COMMERCE ML PROJECT — REAL ONLINE RETAIL DATASET
#   Tools : numpy, pandas, matplotlib, sklearn
#   Data  : UCI Online Retail (541K transactions)
#   Steps : Load → EDA → Clean → Feature Engineer →
#           Standardize/Split → Train → Evaluate → K-Means
# ================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              mean_absolute_error, mean_squared_error, r2_score)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')

# ================================================================
# LOAD DATASET
# ================================================================
print("=" * 62)
print("   E-COMMERCE ML PROJECT — REAL ONLINE RETAIL DATASET")
print("=" * 62)

df_raw = pd.read_csv(DATA_PATH, encoding='latin1')
print(f"\n[DATA LOADED] {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print(f"Columns: {df_raw.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df_raw.head(3).to_string()}")


# ================================================================
# STEP 1 — DATA ANALYSIS (EDA)
# ================================================================
print("\n" + "=" * 62)
print("  STEP 1: DATA ANALYSIS (EDA)")
print("=" * 62)

df = df_raw.copy()
print(f"\n  Shape          : {df.shape}")
print(f"  Missing Values :\n{df.isnull().sum()}")
print(f"  Duplicates     : {df.duplicated().sum():,}")
print(f"  Date Range     : {df['InvoiceDate'].min()}  →  {df['InvoiceDate'].max()}")
print(f"  Countries      : {df['Country'].nunique()}")
print(f"  Unique Products: {df['StockCode'].nunique():,}")
print(f"  Unique Customers: {df['CustomerID'].nunique():,}")
print(f"\n  Qty Stats  :\n{df['Quantity'].describe().round(2).to_string()}")
print(f"\n  Price Stats:\n{df['UnitPrice'].describe().round(2).to_string()}")
print(f"\n  Top 5 Countries:\n{df['Country'].value_counts().head(5).to_string()}")

fig = plt.figure(figsize=(18, 14))
fig.suptitle('STEP 1 — Real Online Retail Data: EDA', fontsize=17, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.42)

# Sales by Country (top 8)
ax1 = fig.add_subplot(gs[0, 0])
top_c = df.groupby('Country')['Quantity'].sum().nlargest(8).sort_values()
top_c.plot(kind='barh', ax=ax1, color='#3498db')
ax1.set_title('Top 8 Countries by Qty Sold', fontweight='bold'); ax1.set_xlabel('Total Quantity')

# Transaction count per month
ax2 = fig.add_subplot(gs[0, 1])
df_tmp = df.copy()
df_tmp['InvoiceDate'] = pd.to_datetime(df_tmp['InvoiceDate'])
df_tmp['YearMonth'] = df_tmp['InvoiceDate'].dt.to_period('M').astype(str)
monthly = df_tmp.groupby('YearMonth').size()
monthly.plot(kind='bar', ax=ax2, color='#e67e22', edgecolor='white')
ax2.set_title('Transactions per Month', fontweight='bold')
ax2.set_xlabel('Month'); ax2.tick_params(axis='x', rotation=45, labelsize=7)

# Quantity Distribution (clipped for readability)
ax3 = fig.add_subplot(gs[0, 2])
qty_clip = df['Quantity'].clip(0, 50)
ax3.hist(qty_clip[qty_clip > 0], bins=30, color='#9b59b6', edgecolor='white', alpha=0.85)
ax3.set_title('Quantity Distribution (0–50)', fontweight='bold'); ax3.set_xlabel('Quantity')

# UnitPrice Distribution
ax4 = fig.add_subplot(gs[1, 0])
price_clip = df['UnitPrice'].clip(0, 20)
ax4.hist(price_clip[price_clip > 0], bins=30, color='#1abc9c', edgecolor='white', alpha=0.85)
ax4.set_title('Unit Price Distribution (0–20)', fontweight='bold'); ax4.set_xlabel('Price (£)')

# Missing values bar
ax5 = fig.add_subplot(gs[1, 1])
missing = df.isnull().sum(); missing = missing[missing > 0]
ax5.bar(missing.index, missing.values, color='#e74c3c')
ax5.set_title('Missing Values per Column', fontweight='bold'); ax5.set_ylabel('Count')
for i, v in enumerate(missing.values):
    ax5.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold', fontsize=9)

# Top 10 best-selling products
ax6 = fig.add_subplot(gs[1, 2])
top_p = df.groupby('Description')['Quantity'].sum().nlargest(8).sort_values()
top_p.index = [d[:22] for d in top_p.index]
top_p.plot(kind='barh', ax=ax6, color='#27ae60')
ax6.set_title('Top 8 Best-Selling Products', fontweight='bold'); ax6.set_xlabel('Total Qty')
ax6.tick_params(axis='y', labelsize=7)

# Correlation heatmap
ax7 = fig.add_subplot(gs[2, 0])
num_cols = ['Quantity', 'UnitPrice']
corr = df[num_cols].corr()
im = ax7.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax7.set_xticks([0,1]); ax7.set_yticks([0,1])
ax7.set_xticklabels(num_cols); ax7.set_yticklabels(num_cols)
for i in range(2):
    for j in range(2):
        ax7.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
ax7.set_title('Correlation Heatmap', fontweight='bold')
plt.colorbar(im, ax=ax7, fraction=0.046)

# Revenue by country (top 8, excl returns)
ax8 = fig.add_subplot(gs[2, 1:3])
df_pos = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
df_pos['Revenue'] = df_pos['Quantity'] * df_pos['UnitPrice']
rev_country = df_pos.groupby('Country')['Revenue'].sum().nlargest(8).sort_values()
rev_country.plot(kind='barh', ax=ax8, color='#f39c12')
ax8.set_title('Revenue by Country (Top 8, £)', fontweight='bold'); ax8.set_xlabel('Revenue (£)')
for i, v in enumerate(rev_country.values):
    ax8.text(v + 1000, i, f'£{v:,.0f}', va='center', fontsize=8)

plt.savefig('step1_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  [SAVED] step1_analysis.png")


# ================================================================
# STEP 2 — DATA CLEANING
# ================================================================
print("\n" + "=" * 62)
print("  STEP 2: DATA CLEANING")
print("=" * 62)

df = df_raw.copy()

# 2.1 Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"\n  [2.1] Removed duplicates: {before - len(df):,}  →  {len(df):,} rows")

# 2.2 Drop rows with missing CustomerID (can't build customer features)
before = len(df)
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)
print(f"  [2.2] Dropped missing CustomerID: {before - len(df):,}  →  {len(df):,} rows")

# 2.3 Fill missing Description with 'Unknown'
df['Description'] = df['Description'].fillna('Unknown')
print(f"  [2.3] Filled missing Description with 'Unknown'")

# 2.4 Remove cancelled orders (InvoiceNo starts with 'C') and bad quantities/prices
before = len(df)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
print(f"  [2.4] Removed cancellations & negatives: {before - len(df):,}  →  {len(df):,} rows")

# 2.5 Parse date, create TotalPrice
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice']  = df['Quantity'] * df['UnitPrice']
df['Month']       = df['InvoiceDate'].dt.month
df['DayOfWeek']   = df['InvoiceDate'].dt.dayofweek   # 0=Mon … 6=Sun
df['Hour']        = df['InvoiceDate'].dt.hour
print(f"  [2.5] Parsed dates → Month, DayOfWeek, Hour features added")

# 2.6 Encode Country with LabelEncoder
le_country = LabelEncoder()
df['Country_Enc'] = le_country.fit_transform(df['Country'])
print(f"  [2.6] LabelEncoder: {df['Country'].nunique()} countries encoded")

df.to_csv('cleaned_data.csv', index=False)
print(f"\n  Final clean shape : {df.shape}")
print(f"  Revenue total     : £{df['TotalPrice'].sum():,.0f}")
print("  [SAVED] cleaned_data.csv")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('STEP 2 — Data Cleaning: Before vs After', fontsize=16, fontweight='bold')

ax = axes[0, 0]
missing_b = df_raw.isnull().sum(); missing_b = missing_b[missing_b > 0]
ax.bar(missing_b.index, missing_b.values, color='#e74c3c')
ax.set_title('Missing Values (Before)', fontweight='bold')
for i, v in enumerate(missing_b.values): ax.text(i, v+200, f'{v:,}', ha='center', fontweight='bold', fontsize=9, color='red')

ax = axes[0, 1]
missing_a = df.isnull().sum(); missing_a = missing_a[missing_a > 0]
if len(missing_a) == 0:
    ax.text(0.5, 0.5, '✓ Zero Missing Values!', ha='center', va='center', fontsize=14, color='green', fontweight='bold', transform=ax.transAxes)
ax.set_title('Missing Values (After)', fontweight='bold'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

ax = axes[0, 2]
labels = ['Raw', 'After Dedup', 'After dropna\nCustomerID', 'After Remove\nCancellations']
values = [len(df_raw), len(df_raw.drop_duplicates()),
          len(df_raw.drop_duplicates().dropna(subset=['CustomerID'])), len(df)]
colors = ['#e74c3c','#e67e22','#f1c40f','#27ae60']
ax.bar(labels, values, color=colors)
ax.set_title('Row Count at Each Step', fontweight='bold')
ax.tick_params(axis='x', labelsize=7)
for i, v in enumerate(values): ax.text(i, v+2000, f'{v:,}', ha='center', fontsize=8, fontweight='bold')

ax = axes[1, 0]
ax.hist(df['Quantity'].clip(0, 30), bins=30, color='#27ae60', edgecolor='white', alpha=0.85)
ax.set_title('Quantity (After Cleaning)', fontweight='bold'); ax.set_xlabel('Quantity')

ax = axes[1, 1]
ax.hist(df['UnitPrice'].clip(0, 15), bins=30, color='#3498db', edgecolor='white', alpha=0.85)
ax.set_title('Unit Price (After Cleaning)', fontweight='bold'); ax.set_xlabel('Price (£)')

ax = axes[1, 2]
ax.hist(df['TotalPrice'].clip(0, 100), bins=30, color='#9b59b6', edgecolor='white', alpha=0.85)
ax.set_title('Total Price per Transaction', fontweight='bold'); ax.set_xlabel('£')

plt.tight_layout()
plt.savefig('step2_cleaning.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] step2_cleaning.png")


# ================================================================
# STEP 3 — FEATURE ENGINEERING + STANDARDIZE + SPLIT
# ================================================================
print("\n" + "=" * 62)
print("  STEP 3: FEATURE ENGINEERING + STANDARDIZE + SPLIT")
print("=" * 62)

# Build customer-level features (RFM-style)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency    = ('InvoiceDate',  lambda x: (snapshot_date - x.max()).days),
    Frequency  = ('InvoiceNo',    'nunique'),
    Monetary   = ('TotalPrice',   'sum'),
    AvgPrice   = ('UnitPrice',    'mean'),
    AvgQty     = ('Quantity',     'mean'),
    NumCountry = ('Country',      'nunique'),
).reset_index()

# Target 1: High-value customer? (Monetary > median) → Binary Classification
median_mon = rfm['Monetary'].median()
rfm['HighValue'] = (rfm['Monetary'] > median_mon).astype(int)

# Target 2: Predict Monetary (Regression)
print(f"\n  RFM Table shape: {rfm.shape}")
print(f"  Median Monetary: £{median_mon:.2f}")
print(f"  High-value customers: {rfm['HighValue'].sum():,} / {len(rfm):,}")
print(f"\n  RFM sample:\n{rfm.head(5).to_string()}")

feature_cols = ['Recency', 'Frequency', 'AvgPrice', 'AvgQty', 'NumCountry']
X       = rfm[feature_cols].values
y_class = rfm['HighValue'].values
y_reg   = rfm['Monetary'].values

# sklearn train_test_split
X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
_,       _,      y_train_r, y_test_r  = train_test_split(X, y_reg,   test_size=0.2, random_state=42)

# sklearn StandardScaler
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n  train_test_split(test_size=0.2, random_state=42)")
print(f"  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")
print(f"  StandardScaler mean  : {np.round(scaler.mean_, 2)}")
print(f"  StandardScaler scale : {np.round(scaler.scale_, 2)}")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('STEP 3 — RFM Features: Before vs After StandardScaler', fontsize=15, fontweight='bold')
for i, feat in enumerate(feature_cols):
    ax = axes[0][i] if i < 4 else axes[1][i-4]
    ax.hist(X_train[:, i],   bins=25, alpha=0.6, color='#e74c3c', label='Before')
    ax.hist(X_train_s[:, i], bins=25, alpha=0.6, color='#27ae60', label='After')
    ax.set_title(feat, fontweight='bold', fontsize=9); ax.legend(fontsize=7)

ax = axes[1][3]
ax.pie([X_train.shape[0], X_test.shape[0]],
       labels=[f'Train\n{X_train.shape[0]:,}\n(80%)', f'Test\n{X_test.shape[0]:,}\n(20%)'],
       colors=['#3498db','#e67e22'], autopct='%1.0f%%', startangle=90,
       textprops={'fontsize': 10, 'fontweight': 'bold'})
ax.set_title('Train / Test Split', fontweight='bold')

plt.tight_layout()
plt.savefig('step3_standardize.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] step3_standardize.png")


# ================================================================
# STEP 4 — TRAIN MODELS (sklearn)
# ================================================================
print("\n" + "=" * 62)
print("  STEP 4: TRAIN MODELS  (sklearn)")
print("=" * 62)

# A. Logistic Regression
print("\n  [A] LogisticRegression(max_iter=1000)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_s, y_train_c)
print(f"     Coef: {np.round(lr_model.coef_[0], 3)}")

# B. KNN
print("  [B] KNeighborsClassifier(n_neighbors=5)...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_s, y_train_c)

# C. Decision Tree
print("  [C] DecisionTreeClassifier(max_depth=4)...")
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train_s, y_train_c)
print(f"     Feature importances: {np.round(dt_model.feature_importances_, 3)}")

# D. Linear Regression (predict Monetary value)
print("  [D] LinearRegression()...")
reg_model = LinearRegression()
reg_model.fit(X_train_s, y_train_r)
print(f"     Coef: {np.round(reg_model.coef_, 2)}")
print(f"     Intercept: {reg_model.intercept_:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('STEP 4 — sklearn Model Insights', fontsize=15, fontweight='bold')

ax = axes[0]
cols = ['#e74c3c' if c > 0 else '#3498db' for c in lr_model.coef_[0]]
ax.barh(feature_cols, lr_model.coef_[0], color=cols)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('LogisticRegression\nCoefficients', fontweight='bold')

ax = axes[1]
imp = dt_model.feature_importances_
sorted_idx = np.argsort(imp)
ax.barh(np.array(feature_cols)[sorted_idx], imp[sorted_idx], color='#27ae60')
ax.set_title('DecisionTree\nFeature Importances', fontweight='bold')

ax = axes[2]
cols2 = ['#e74c3c' if c > 0 else '#3498db' for c in reg_model.coef_]
ax.barh(feature_cols, reg_model.coef_, color=cols2)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('LinearRegression\nCoefficients', fontweight='bold')

plt.tight_layout()
plt.savefig('step4_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] step4_training.png")


# ================================================================
# STEP 5 — EVALUATE + PREDICT (sklearn metrics)
# ================================================================
print("\n" + "=" * 62)
print("  STEP 5: EVALUATE & PREDICT  (sklearn metrics)")
print("=" * 62)

y_pred_lr  = lr_model.predict(X_test_s)
y_pred_knn = knn_model.predict(X_test_s)
y_pred_dt  = dt_model.predict(X_test_s)
y_pred_reg = reg_model.predict(X_test_s)

models = {'Logistic Reg': y_pred_lr, 'KNN (K=5)': y_pred_knn, 'Decision Tree': y_pred_dt}
results = {}
print("\n  --- Classification Metrics ---")
for name, preds in models.items():
    acc  = accuracy_score(y_test_c, preds) * 100
    prec = precision_score(y_test_c, preds, zero_division=0) * 100
    rec  = recall_score(y_test_c, preds, zero_division=0) * 100
    f1   = f1_score(y_test_c, preds, zero_division=0) * 100
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    print(f"\n  {name}: Acc={acc:.1f}%  Prec={prec:.1f}%  Recall={rec:.1f}%  F1={f1:.1f}%")

print("\n  Full Report (Decision Tree):")
print(classification_report(y_test_c, y_pred_dt, target_names=['Low Value','High Value'], zero_division=0))

mae_v  = mean_absolute_error(y_test_r, y_pred_reg)
mse_v  = mean_squared_error(y_test_r, y_pred_reg)
rmse_v = np.sqrt(mse_v)
r2_v   = r2_score(y_test_r, y_pred_reg)
print(f"  --- Regression (Predict Monetary) ---")
print(f"  MAE=£{mae_v:.2f}  RMSE=£{rmse_v:.2f}  R²={r2_v:.4f}")

print(f"\n  --- Sample Predictions (10 customers) ---")
print(f"  {'#':<4}{'HighVal?':<10}{'LR':<7}{'KNN':<7}{'DT':<7}{'Act £':<12}{'Pred £'}")
print("  " + "-"*55)
for i in range(10):
    print(f"  {i+1:<4}{y_test_c[i]:<10}{y_pred_lr[i]:<7}{y_pred_knn[i]:<7}"
          f"{y_pred_dt[i]:<7}{y_test_r[i]:<12.2f}{y_pred_reg[i]:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('STEP 5 — sklearn: Model Evaluation (RFM Customer Data)', fontsize=15, fontweight='bold')

ax = axes[0, 0]
mnames = list(results.keys())
accs = [results[m]['Accuracy'] for m in mnames]
bars = ax.bar(mnames, accs, color=['#e74c3c','#3498db','#2ecc71'], width=0.5)
ax.set_title('Model Accuracy', fontweight='bold'); ax.set_ylim(0, 110)
for bar, v in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.tick_params(axis='x', rotation=12)

ax = axes[0, 1]
cm = confusion_matrix(y_test_c, y_pred_dt)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Pred: Low','Pred: High']); ax.set_yticklabels(['Act: Low','Act: High'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=18, fontweight='bold',
                color='white' if cm[i,j]>cm.max()/2 else 'black')
ax.set_title('Confusion Matrix\n(Decision Tree)', fontweight='bold')
plt.colorbar(im, ax=ax)

ax = axes[0, 2]
x = np.arange(len(mnames)); w = 0.25
for i, (metric, col) in enumerate(zip(['Precision','Recall','F1'],['#e74c3c','#f39c12','#9b59b6'])):
    ax.bar(x + i*w, [results[m][metric] for m in mnames], w, label=metric, color=col, alpha=0.85)
ax.set_title('Precision / Recall / F1', fontweight='bold')
ax.set_xticks(x + w); ax.set_xticklabels([m.split()[0] for m in mnames])
ax.set_ylabel('Score (%)'); ax.legend(); ax.set_ylim(0, 110)

ax = axes[1, 0]
cap = np.percentile(y_test_r, 95)
mask = y_test_r < cap
ax.scatter(y_test_r[mask], y_pred_reg[mask], alpha=0.5, color='#3498db', edgecolors='white', s=40)
mn, mx = y_test_r[mask].min(), y_test_r[mask].max()
ax.plot([mn,mx],[mn,mx],'r--', lw=2, label='Perfect')
ax.set_title(f'Actual vs Predicted Monetary\nR²={r2_v:.3f}', fontweight='bold')
ax.set_xlabel('Actual £'); ax.set_ylabel('Predicted £'); ax.legend()

ax = axes[1, 1]
res = y_test_r - y_pred_reg; res_clip = res[np.abs(res) < np.percentile(np.abs(res), 95)]
ax.hist(res_clip, bins=30, color='#e67e22', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', lw=2, linestyle='--')
ax.set_title('Residuals Distribution\n(clipped at 95th pct)', fontweight='bold')

ax = axes[1, 2]; ax.axis('off')
headers = ['#','HighVal','LR','KNN','DT','Actual £','Pred £']
tdata = [[str(i+1),
          '✓' if y_test_c[i]==1 else '✗',
          '✓' if y_pred_lr[i]==1 else '✗',
          '✓' if y_pred_knn[i]==1 else '✗',
          '✓' if y_pred_dt[i]==1 else '✗',
          f'£{y_test_r[i]:.0f}', f'£{y_pred_reg[i]:.0f}'] for i in range(10)]
table = ax.table(cellText=tdata, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.8)
for (r,c), cell in table.get_celld().items():
    if r==0: cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    elif r%2==0: cell.set_facecolor('#ecf0f1')
ax.set_title('Sample Customer Predictions', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('step5_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  [SAVED] step5_evaluation.png")

# Decision Tree plot
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt_model, feature_names=feature_cols, class_names=['Low Value','High Value'],
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title('STEP 5 — Decision Tree (max_depth=4): High-Value Customer Prediction',
             fontsize=13, fontweight='bold')
plt.savefig('step5_decision_tree.png', dpi=130, bbox_inches='tight')
plt.close()
print("  [SAVED] step5_decision_tree.png")


# ================================================================
# STEP 6 — UNSUPERVISED: K-MEANS (sklearn)
# ================================================================
print("\n" + "=" * 62)
print("  STEP 6: K-MEANS CUSTOMER SEGMENTATION  (sklearn)")
print("=" * 62)

Xc  = rfm[['Recency','Frequency','Monetary']].values
sc2 = StandardScaler()
Xcs = sc2.fit_transform(Xc)

print("\n  Elbow Method:")
inertias = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xcs)
    inertias.append(km.inertia_)
    print(f"    K={k}: Inertia={km.inertia_:.2f}")

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
km3.fit(Xcs)
rfm['Segment'] = km3.labels_

seg_names = {}
for c in range(3):
    seg = rfm[rfm['Segment']==c]
    avg_r = seg['Recency'].mean(); avg_f = seg['Frequency'].mean(); avg_m = seg['Monetary'].mean()
    if avg_r < rfm['Recency'].median() and avg_m > rfm['Monetary'].median():
        seg_names[c] = 'Champions'
    elif avg_m < rfm['Monetary'].quantile(0.33):
        seg_names[c] = 'At Risk'
    else:
        seg_names[c] = 'Loyal Customers'
rfm['SegmentName'] = rfm['Segment'].map(seg_names)

print("\n  Customer Segments:")
for c in range(3):
    seg = rfm[rfm['Segment']==c]
    print(f"  Cluster {c} [{seg_names[c]}] ({len(seg):,} customers):  "
          f"Avg Recency={seg['Recency'].mean():.0f}d  "
          f"Avg Frequency={seg['Frequency'].mean():.1f}  "
          f"Avg Monetary=£{seg['Monetary'].mean():.0f}")

colors = ['#e74c3c','#3498db','#2ecc71']
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('STEP 6 — sklearn KMeans: RFM Customer Segmentation', fontsize=16, fontweight='bold')

ax = axes[0, 0]
ax.plot(range(1, 9), inertias, 'o-', color='#9b59b6', lw=2, ms=8)
ax.fill_between(range(1,9), inertias, alpha=0.15, color='#9b59b6')
ax.axvline(3, color='red', linestyle='--', label='K=3 chosen')
ax.set_title('Elbow Method', fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
for c in range(3):
    m = rfm['Segment']==c
    ax.scatter(rfm.loc[m,'Recency'], rfm.loc[m,'Monetary'].clip(upper=5000),
               c=colors[c], label=f"C{c}:{seg_names[c]}", alpha=0.6, s=30)
ax.set_title('Recency vs Monetary (capped £5K)', fontweight='bold')
ax.set_xlabel('Recency (days)'); ax.set_ylabel('Monetary (£)'); ax.legend(fontsize=8)

ax = axes[0, 2]
for c in range(3):
    m = rfm['Segment']==c
    ax.scatter(rfm.loc[m,'Frequency'], rfm.loc[m,'Monetary'].clip(upper=5000),
               c=colors[c], label=f"C{c}:{seg_names[c]}", alpha=0.6, s=30)
ax.set_title('Frequency vs Monetary (capped £5K)', fontweight='bold')
ax.set_xlabel('Frequency'); ax.set_ylabel('Monetary (£)'); ax.legend(fontsize=8)

ax = axes[1, 0]
sizes = [len(rfm[rfm['Segment']==c]) for c in range(3)]
labels_pie = [f"{seg_names[c]}\n({sizes[c]:,})" for c in range(3)]
ax.pie(sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Segment Distribution', fontweight='bold')

ax = axes[1, 1]
avg_m = [rfm[rfm['Segment']==c]['Monetary'].mean() for c in range(3)]
bars = ax.bar([seg_names[c] for c in range(3)], avg_m, color=colors, width=0.5)
ax.set_title('Avg Monetary per Segment', fontweight='bold'); ax.set_ylabel('Avg £')
for bar, v in zip(bars, avg_m):
    ax.text(bar.get_x()+bar.get_width()/2, v+10, f'£{v:.0f}', ha='center', fontweight='bold')
ax.tick_params(axis='x', rotation=10)

ax = axes[1, 2]
centroids_orig = sc2.inverse_transform(km3.cluster_centers_)
rfm_feat_labels = ['Recency', 'Frequency', 'Monetary']
for c in range(3):
    ax.bar(np.arange(3) + c*0.25, centroids_orig[c], 0.25,
           label=seg_names[c], color=colors[c], alpha=0.85)
ax.set_title('Cluster Centroids (Original Scale)', fontweight='bold')
ax.set_xticks(np.arange(3)+0.25); ax.set_xticklabels(rfm_feat_labels); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('step6_kmeans.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] step6_kmeans.png")


# ================================================================
# DONE
# ================================================================
print("\n" + "=" * 62)
print("  ALL 6 STEPS COMPLETE!")
print("  sklearn used:")
print("    train_test_split, StandardScaler, LabelEncoder")
print("    LogisticRegression, KNeighborsClassifier")
print("    DecisionTreeClassifier, LinearRegression, KMeans")
print("    accuracy_score, precision_score, recall_score")
print("    f1_score, confusion_matrix, classification_report")
print("    mean_absolute_error, mean_squared_error, r2_score")
print("=" * 62)