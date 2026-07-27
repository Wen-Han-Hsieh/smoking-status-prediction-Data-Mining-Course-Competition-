import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from tqdm.notebook import tqdm
import time, warnings, json

warnings.filterwarnings('ignore')

# 載入資料
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 特徵工程：視力、聽力欄位處理
def create_extra_features(df):
    # 視力超過 9 的視為異常值，修正為 0.0（極差視力）
    df['eyesight(left)'] = np.where(df['eyesight(left)'] > 9, 0.0, df['eyesight(left)'])
    df['eyesight(right)'] = np.where(df['eyesight(right)'] > 9, 0.0, df['eyesight(right)'])
    # 重編視力欄位：左眼為較佳視力，右眼為較差視力
    best_vision = np.minimum(df['eyesight(left)'], df['eyesight(right)'])
    worst_vision = np.maximum(df['eyesight(left)'], df['eyesight(right)'])
    df['eyesight(left)'] = best_vision
    df['eyesight(right)'] = worst_vision
    # 聽力也是相同邏輯：0表示正常，數字越大越差
    best_hearing = np.minimum(df['hearing(left)'], df['hearing(right)'])
    worst_hearing = np.maximum(df['hearing(left)'], df['hearing(right)'])
    df['hearing(left)'] = best_hearing - 1
    df['hearing(right)'] = worst_hearing - 1

create_extra_features(train)
create_extra_features(test)

# 特徵工程：衍生欄位
def add_derived_features(df):
    df['BMI'] = df['weight(kg)'] / (df['height(cm)'] / 100) ** 2
    df['BP_ratio'] = df['systolic'] / df['relaxation']
    df['liver_ratio'] = df['ALT'] / df['AST']
    df['cholesterol_ratio'] = df['LDL'] / df['HDL']

add_derived_features(train)
add_derived_features(test)

# 畫圖觀察特徵
def plot_feature_distributions(data, exclude_cols=['id', 'smoking'], ncols=5):
    numeric_features = data.select_dtypes(include=['float64', 'int64']).drop(columns=exclude_cols).columns
    num_features = len(numeric_features)
    nrows = (num_features // ncols) + (num_features % ncols > 0)

    # 直方圖
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numeric_features):
        sns.histplot(data[col], bins=30, kde=True, ax=axes[i])
        skewness_value = skew(data[col].dropna())
        axes[i].text(0.95, 0.95, f'Skewness: {skewness_value:.2f}',
                     transform=axes[i].transAxes,
                     fontsize=10,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.suptitle("Feature Distributions with Skewness", fontsize=16, y=1.02)
    plt.show()

    # 箱型圖
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numeric_features):
        sns.boxplot(x=data[col], ax=axes[i])
        axes[i].set_title(f'{col} - Boxplot')
        axes[i].set_xlabel('')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.suptitle("Boxplots of Numeric Features", fontsize=16, y=1.02)
    plt.show()

plot_feature_distributions(train, ['id', 'smoking'], 5)

# 特徵工程：分離類別特徵與數值特徵

# 分離目標與特徵
X_train = train.drop(columns=['smoking', 'id'])
y_train = train['smoking']
X_test = test.drop(columns=['id'])

# One-Hot Encoding 特徵
to_ohe = ['hearing(left)', 'hearing(right)', 'dental caries']
X_train_ohe = X_train[to_ohe]
X_test_ohe = X_test[to_ohe]

# 'Urine protein'也是類別但要做 Label Encoding
X_train_cont = X_train.drop(columns=to_ohe + ['Urine protein']) 
X_test_cont = X_test.drop(columns=to_ohe + ['Urine protein'])

# 對 'age' 做 frequency encoding
def frequency_encode(train_col, test_col):
    freq_map = train_col.value_counts(normalize=True) 
    return train_col.map(freq_map), test_col.map(freq_map).fillna(0)
X_train_cont['age_freq'], X_test_cont['age_freq'] = frequency_encode(X_train_cont['age'], X_test_cont['age'])

# 特徵工程：數值特徵偏態處理 log1p
numeric_cols = X_train_cont.select_dtypes(include=['float64', 'int64']).columns.drop(['age_freq'])

# 找出偏態大於 1 的欄位做 log1p
skewed_features = [col for col in numeric_cols if skew(X_train_cont[col]) > 1]

# 設置每行兩張圖（左為原始，右為 log1p 轉換）
ncols = 2
nrows = math.ceil(len(skewed_features)) 
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 4))
log_transformed_features = []  
# 繪製每個特徵的直方圖
for i, col in enumerate(skewed_features):
    # 原直方圖
    sns.histplot(X_train_cont[col], bins=30, kde=True, ax=axes[i, 0])
    skewness_before = skew(X_train_cont[col].dropna())
    axes[i, 0].set_title(f'{col} - Before log1p\nSkewness: {skewness_before:.2f}')
    axes[i, 0].set_xlabel('')
    axes[i, 0].set_ylabel('')
    # 確認數值非負並執行 log1p 轉換
    if (X_train_cont[col] >= 0).all() and (X_test_cont[col] >= 0).all():
        X_train_cont[col] = np.log1p(X_train_cont[col])
        X_test_cont[col] = np.log1p(X_test_cont[col])
        log_transformed_features.append(col)
    # log1p 直方圖
    sns.histplot(X_train_cont[col], bins=30, kde=True, ax=axes[i, 1])
    skewness_after = skew(X_train_cont[col].dropna())
    axes[i, 1].set_title(f'{col} - After log1p\nSkewness: {skewness_after:.2f}')
    axes[i, 1].set_xlabel('')
    axes[i, 1].set_ylabel('')

plt.tight_layout()
plt.show()
# 打印已轉換的特徵
print("Log-transformed features:", log_transformed_features)

# 特徵工程：數值特徵標準化：RobustScaler 
sc = RobustScaler()
X_train_scaled = pd.DataFrame(sc.fit_transform(X_train_cont.drop(columns=['age_freq'])), columns=X_train_cont.drop(columns=['age_freq']).columns)
X_test_scaled = pd.DataFrame(sc.transform(X_test_cont.drop(columns=['age_freq'])), columns=X_test_cont.drop(columns=['age_freq']).columns)
X_train_scaled['age_freq'] = X_train_cont['age_freq']
X_test_scaled['age_freq'] = X_test_cont['age_freq']

# 特徵工程：類別特徵處理

# 對 'Urine protein' 使用 Label Encoding
le = LabelEncoder()
X_train['Urine protein'] = le.fit_transform(X_train['Urine protein'])
X_test['Urine protein'] = le.transform(X_test['Urine protein'])

# 對其他類別特徵進行 One-Hot Encoding
X_train_ohe = pd.get_dummies(X_train_ohe, columns=to_ohe)
X_test_ohe = pd.get_dummies(X_test_ohe, columns=to_ohe)

# 合併數值欄位與已處理的類別欄位
X_train_final = pd.concat([X_train_scaled, X_train_ohe, X_train[['Urine protein']]], axis=1)
X_test_final = pd.concat([X_test_scaled, X_test_ohe, X_test[['Urine protein']]], axis=1)

# 確保測試集與訓練集的特徵一致
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

# 特徵工程：過濾高相關特徵，找出相關性大於 0.8 的特徵並準備過濾
corr_matrix = X_train_final.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]

# 繪製相關性熱圖
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,  
    fmt=".2f",   
    cmap='coolwarm',
    cbar=True,
    annot_kws={"fontsize": 5, "color": "black"}
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

print("Features dropped due to high correlation (>0.8):", to_drop)
X_train_final = X_train_final.drop(columns=to_drop)
X_test_final = X_test_final.drop(columns=to_drop)

# 訓練進度條
class TqdmCallback:
    def __init__(self, total_iter):
        self.pbar = tqdm(total=total_iter)

    def __call__(self, res):
        self.pbar.update(1)
        if self.pbar.n % 5 == 0:
            print(f"[{self.pbar.n}/{self.pbar.total}] 進度更新")

# 模型評估圖
def plot_model_evaluation(opt, X_train, y_train, cv, title_prefix="Model"):
    # AUC Learning Curve
    auc_scores = opt.cv_results_['mean_test_score']
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(auc_scores)+1), auc_scores, marker='o', label='AUC Score')
    plt.axhline(y=max(auc_scores), color='r', linestyle='--', label=f"Best AUC = {max(auc_scores):.4f}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean AUC (CV)")
    plt.title(f"{title_prefix} AUC Learning Curve")
    plt.legend()
    plt.grid(True)
    # ROC curve
    y_pred_proba_cv = cross_val_predict(opt.best_estimator_, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_pred_proba_cv)
    auc_score = roc_auc_score(y_train, y_pred_proba_cv)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主訓練程式
def main(X_train_final, X_test_final, y_train, test_id):
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    param_space = {
      'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
      'n_estimators': Integer(100, 500),
      'max_depth': Integer(3, 10),
      'subsample': Real(0.5, 1.0),
      'colsample_bytree': Real(0.5, 1.0),
      'min_child_weight': Integer(1, 10),
      'gamma': Real(0, 1.0),
      'reg_alpha': Real(0, 1.0),
      'reg_lambda': Real(0, 1.0),
      'scale_pos_weight': Real(1, 10)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    opt = BayesSearchCV(
        model,
        search_spaces=param_space,
        cv=cv,
        n_iter=70,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    print(f"\n開始 Bayesian Optimization...")
    opt.fit(X_train_final, y_train, callback=TqdmCallback(70))
    print("最佳參數：", opt.best_params_)
    print(f"最佳 AUC：{opt.best_score_:.4f}")

    # 用最佳參數重新訓練
    final_model = XGBClassifier(**opt.best_params_, random_state=42, use_label_encoder=False, eval_metric='logloss')
    final_model.fit(X_train_final, y_train)

    # 預測結果儲存
    pred_test = final_model.predict_proba(X_test_final)[:, 1]
    result = pd.DataFrame({'id': test_id, 'smoking': pred_test})
    result.to_csv(f"final_result_all.csv", index=False)
    print("預測結果已儲存")

    # 計算訓練集預測結果
    y_pred_train = final_model.predict(X_train_final)
    y_train_proba = final_model.predict_proba(X_train_final)[:, 1]

    # final AUC score
    final_auc = roc_auc_score(y_train, y_train_proba)
    print(f"\n最終模型 AUC（Train）：{final_auc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_train, y_pred_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Train - all)")
    plt.tight_layout()
    plt.show()

    # final ROC curve
    fpr, tpr, _ = roc_curve(y_train, y_train_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {final_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve (Train - all)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

main(X_train_final, X_test_final, y_train, test['id'])