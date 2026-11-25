# -*- coding: utf-8 -*-
"""
心力衰竭临床记录数据集 - 训练集/测试集划分（80% / 20%，固定随机种子）
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 固定随机种子，保证划分结果完全可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TARGET_COL = "DEATH_EVENT"


def split_train_test(
    csv_path: Path = Path("heart_failure_clinical_records_dataset.csv"),
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """读取数据并按 8:2 分层划分为训练集和测试集"""
    df = pd.read_csv(csv_path)
    resolved_target = target_col if target_col in df.columns else df.columns[0]

    X = df.drop(columns=[resolved_target])
    y = df[resolved_target]

    # test_size=0.2 表示训练集占 80%、测试集占 20%；stratify=y 保持标签分布一致
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"总样本数: {len(df)} | 特征数: {X.shape[1]} | 标签列: {resolved_target}")
    print(
        f"训练集: {len(X_train)} 条 ({len(X_train) / len(df) * 100:.2f}%) | "
        f"正类(1)占比: {y_train.mean() * 100:.2f}%"
    )
    print(
        f"测试集: {len(X_test)} 条 ({len(X_test) / len(df) * 100:.2f}%) | "
        f"正类(1)占比: {y_test.mean() * 100:.2f}%"
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 若你的文件放在 data/ 目录下，可改为 Path("data/heart_failure_clinical_records_dataset.csv")
    data_path = (
        Path("data/heart_failure_clinical_records_dataset.csv")
        if Path("data/heart_failure_clinical_records_dataset.csv").exists()
        else Path("heart_failure_clinical_records_dataset.csv")
    )
    X_train, X_test, y_train, y_test = split_train_test(csv_path=data_path)