import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, \
    roc_curve, precision_recall_curve, average_precision_score
import xgboost as xgb

# 定义使用的特征类型："esm2", "rotate" 或 "both"
use_features = "both"  # 修改为 "esm2", "rotate", 或 "both"

# 定义文件路径模板
positive_train_file_template = 'PositiveSample_Train_Fold{fold}_PBI.csv'
positive_test_file_template = 'PositiveSample_Test_Fold{fold}_PBI.csv'
negative_train_file_template = 'NegativeSample_Train_Fold{fold}_PBI.csv'
negative_test_file_template = 'NegativeSample_Test_Fold{fold}_PBI.csv'

# 读取特征文件
esm2_df = pd.read_csv('reduced_features_64.csv')  # ESM-2特征文件路径
rotate_df = pd.read_csv('MuRE_feature_flattened.csv')  # MuRE特征文件路径

# 假设数字编号在第一列，将其重命名为 'id'
esm2_df.rename(columns={esm2_df.columns[0]: 'id'}, inplace=True)
rotate_df.rename(columns={rotate_df.columns[0]: 'id'}, inplace=True)

# 将 'id' 列转换为字符串类型以便合并
esm2_df['id'] = esm2_df['id'].astype(str)
rotate_df['id'] = rotate_df['id'].astype(str)

# 根据选择使用的特征进行合并或选择
if use_features == "esm2":
    features_df = esm2_df
elif use_features == "rotate":
    features_df = rotate_df
elif use_features == "both":
    # 合并特征文件
    features_df = esm2_df.merge(rotate_df, on='id')  # 基于 'id' 列进行合并
else:
    raise ValueError("Invalid value for use_features. Choose 'esm2', 'rotate', or 'both'.")

# 创建训练、验证、测试数据集的函数
def create_dataset(pos_file, neg_file, features_df):
    pos_df = pd.read_csv(pos_file)
    neg_df = pd.read_csv(neg_file)

    pos_features = []
    neg_features = []
    pos_pairs = []
    neg_pairs = []

    for _, row in pos_df.iterrows():
        feat_1 = features_df.loc[features_df['id'] == str(row.iloc[0]), features_df.columns != 'id'].values.flatten()
        feat_2 = features_df.loc[features_df['id'] == str(row.iloc[1]), features_df.columns != 'id'].values.flatten()
        if len(feat_1) == len(feat_2):  # 检查特征长度是否一致
            pos_features.append(np.concatenate([feat_1, feat_2]))
            pos_pairs.append((row.iloc[0], row.iloc[1]))
        else:
            print(f"Warning: Mismatch in feature length for positive sample: {row.iloc[0]}, {row.iloc[1]}")

    for _, row in neg_df.iterrows():
        feat_1 = features_df.loc[features_df['id'] == str(row.iloc[0]), features_df.columns != 'id'].values.flatten()
        feat_2 = features_df.loc[features_df['id'] == str(row.iloc[1]), features_df.columns != 'id'].values.flatten()
        if len(feat_1) == len(feat_2):  # 检查特征长度是否一致
            neg_features.append(np.concatenate([feat_1, feat_2]))
            neg_pairs.append((row.iloc[0], row.iloc[1]))
        else:
            print(f"Warning: Mismatch in feature length for negative sample: {row.iloc[0]}, {row.iloc[1]}")

    # 创建标签
    pos_labels = np.ones(len(pos_features))
    neg_labels = np.zeros(len(neg_features))

    # 合并数据
    features = np.array(pos_features + neg_features)
    labels = np.concatenate([pos_labels, neg_labels])
    pairs = pos_pairs + neg_pairs

    return features, labels, pairs

# 存储每次实验结果的列表
results = []
all_predictions = []

for fold in range(1, 6):  # 对五折进行循环
    print(f"Running fold {fold}...")

    # 读取每一折的训练和测试数据集
    positive_train_file = positive_train_file_template.format(fold=fold)
    positive_test_file = positive_test_file_template.format(fold=fold)
    negative_train_file = negative_train_file_template.format(fold=fold)
    negative_test_file = negative_test_file_template.format(fold=fold)

    # 生成训练和测试数据集
    X_train, y_train, _ = create_dataset(positive_train_file, negative_train_file, features_df)
    X_test, y_test, test_pairs = create_dataset(positive_test_file, negative_test_file, features_df)

    # 初始化XGBoost分类器
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
    )

    # 训练模型
    model.fit(X_train, y_train, verbose=False)

    # 预测并评估模型
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

    # 保存预测结果和对应的蛋白质对信息
    fold_predictions = pd.DataFrame({
        'Protein_1': [pair[0] for pair in test_pairs],
        'Protein_2': [pair[1] for pair in test_pairs],
        'True Label': y_test,
        'Predicted Label': y_pred,
        'Predicted Probability': y_pred_proba
    })
    fold_predictions.to_csv(f'predictions_fold_{fold}.csv', index=False)
    all_predictions.append(fold_predictions)

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    aupr = average_precision_score(y_test, y_pred_proba)

    # 计算并保存ROC曲线数据
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    roc_data = pd.DataFrame({
        'False Positive Rate': np.round(fpr, 4),
        'True Positive Rate': np.round(tpr, 4),
        'Thresholds': np.round(roc_thresholds, 4)
    })
    roc_data.to_csv(f'roc_curve_fold_{fold}.csv', index=False)

    # 计算并保存PR曲线数据
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_data = pd.DataFrame({
        'Precision': np.round(precision_curve, 4),
        'Recall': np.round(recall_curve, 4),
        'Thresholds': np.round(np.append(pr_thresholds, 1), 4)  # Append 1 to match length of precision/recall arrays
    })
    pr_data.to_csv(f'pr_curve_fold_{fold}.csv', index=False)

    # 保存每次实验的结果
    results.append({
        'Fold': fold,
        'Accuracy': round(accuracy, 4),
        'F1 Score': round(f1, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'MCC': round(mcc, 4),
        'ROC AUC': round(roc_auc, 4),
        'AUPR': round(aupr, 4)
    })

# 将所有折的预测结果合并到一个文件
all_predictions_df = pd.concat(all_predictions)
all_predictions_df.to_csv('all_folds_predictions.csv', index=False)

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 计算平均值和标准差
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'MCC', 'ROC AUC', 'AUPR']
mean_std_data = {metric: [round(results_df[metric].mean(), 4), round(results_df[metric].std(), 4)] for metric in metrics}
mean_std_df = pd.DataFrame(mean_std_data, index=['Mean', 'Std'])

# 将均值和标准差添加到结果中
results_df = pd.concat([results_df, mean_std_df])

# 保存结果为CSV文件
results_df.to_csv('ALL_results.csv', index=True)

print("所有折交叉验证的结果已保存到 'single_murp_results.csv'")
print("所有预测结果已保存到 'all_folds_predictions.csv'")
print("ROC 和 PR 曲线数据已保存到对应的 CSV 文件中")
