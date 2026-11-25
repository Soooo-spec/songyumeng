# -*- coding: utf-8 -*-
"""
心力衰竭临床记录数据集 - 数据初处理脚本（Data Exploratory & Preprocessing）
================================================================================
运行环境与依赖版本建议（详见文末 requirements.txt）：
- Python >= 3.9
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0

目录约定（全部使用相对路径）：
- 输入数据：data/heart_failure_clinical_records_dataset.csv
- 输出结果：results/ （自动创建，包含统计表与连续特征箱线图）
================================================================================
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# 0. 全局随机种子与路径配置（可复现性保障）
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = DATA_DIR / "heart_failure_clinical_records_dataset.csv"

# 自动创建结果输出目录（若不存在）
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 绘图全局设置：图内标题与标签使用英文，避免中文字体缺失导致的乱码（方块）
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# 字段分组定义
FEATURE_COLS = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "sex",
    "serum_creatinine",
    "serum_sodium",
    "smoking",
    "time",
]
TARGET_COL = "DEATH_EVENT"

# 7 个连续数值特征（用于 IQR 异常值检测与箱线图绘制）
CONTINUOUS_COLS = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]

# 6 个二值/分类字段（5 个二值特征 + 1 个二值目标变量）
BINARY_COLS = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
    "DEATH_EVENT",
]


def load_and_describe(csv_path: Path, out_dir: Path) -> pd.DataFrame:
    """1. 加载数据并输出数值字段描述性统计与分类/布尔字段分布"""
    print("=" * 72)
    print("【步骤 1】加载数据与描述性统计分析")
    print("=" * 72)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {csv_path}，请确认已将 CSV 文件放入 data/ 目录。"
        )

    df = pd.read_csv(csv_path)
    print(f"[信息] 数据加载成功: {csv_path}")
    print(f"[信息] 数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")

    # 1.1 数值字段描述性统计（包含 count, mean, std, min, 25%, 50%, 75%, max）
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc_df = df[num_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc_df = desc_df.rename(columns={"25%": "q25", "50%": "median_q50", "75%": "q75"})

    describe_path = out_dir / "describe.csv"
    desc_df.to_csv(describe_path, index=True, float_format="%.6f")
    print(f"[保存] 数值字段描述性统计已保存至: {describe_path}")
    print("\n--- 数值字段描述性统计摘要 ---")
    print(desc_df.round(4).to_string())

    # 1.2 分类/布尔字段取值分布（0/1 计数与占比）
    cat_records = []
    print("\n--- 分类/布尔字段取值分布（计数与比例） ---")
    for col in BINARY_COLS:
        counts = df[col].value_counts(dropna=False).sort_index()
        props = df[col].value_counts(normalize=True, dropna=False).sort_index()
        print(f"字段 [{col}]:")
        for val, cnt in counts.items():
            prop = props[val]
            print(f"  取值 {val}: 计数 = {cnt:3d}, 比例 = {prop * 100:6.2f}%")
            cat_records.append(
                {
                    "column": col,
                    "value": val,
                    "count": int(cnt),
                    "proportion": round(float(prop), 6),
                    "percentage": round(float(prop * 100), 2),
                }
            )

    cat_dist_df = pd.DataFrame(cat_records)
    cat_dist_path = out_dir / "categorical_distribution.csv"
    cat_dist_df.to_csv(cat_dist_path, index=False)
    print(f"\n[保存] 分类/布尔字段分布统计已保存至: {cat_dist_path}")

    return df


def check_missing_and_outliers(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """2. 检查缺失值，并使用 IQR 识别连续特征异常值、绘制箱线图与给出处理建议"""
    print("\n" + "=" * 72)
    print("【步骤 2】缺失值与异常值检查")
    print("=" * 72)

    # 2.1 缺失值检查
    # 本数据集共 299 行 13 列，经 isnull().sum() 检查所有列缺失数均为 0：
    # “本数据集无缺失值，无需插补”
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())

    print("--- 各列缺失值统计 (isnull().sum()) ---")
    print(missing_series.to_string())

    if total_missing == 0:
        print("\n[结论] 本数据集无缺失值，无需插补。")
    else:
        # 若未来在扩展数据中出现缺失值：连续数值特征建议使用中位数（受极端值影响小），
        # 二值/分类特征建议使用众数（保持合法离散取值）。
        print(f"\n[提示] 检测到共 {total_missing} 个缺失值：数值字段建议用中位数插补，分类字段建议用众数插补。")
        for col in df.columns:
            if missing_series[col] > 0:
                if col in CONTINUOUS_COLS:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    # 2.2 基于 IQR 规则检测连续数值字段异常值并绘制箱线图
    # 规则：下界 = Q1 - 1.5 * IQR，上界 = Q3 + 1.5 * IQR，超出 [下界, 上界] 视为 IQR 异常值
    print("\n--- 连续数值字段 IQR 异常值检测与箱线图生成 ---")
    outlier_summary = []

    for col in CONTINUOUS_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(is_outlier.sum())
        outlier_pct = (outlier_count / len(df)) * 100

        outlier_summary.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
            }
        )

        print(
            f"字段 [{col:24s}] | 异常值个数: {outlier_count:2d} ({outlier_pct:5.2f}%) | "
            f"IQR 围栏: [{lower_bound:.2f}, {upper_bound:.2f}]"
        )

        # 绘制单变量箱线图（英文标题避免中文字体问题）
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[col], ax=ax, color="#4C72B0", width=0.4)
        ax.set_title(f"Boxplot of {col} (Outliers: {outlier_count})", fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        plt.tight_layout()

        box_path = out_dir / f"box_{col}.png"
        plt.savefig(box_path, dpi=150)
        plt.close(fig)

    print(f"[保存] 7 张连续特征箱线图已全部保存至 {out_dir}/box_<字段名>.png")

    # 2.3 异常值临床与工程处理建议
    # 临床视角：心力衰竭患者常伴随重症或合并症，部分偏离 IQR 的高/低值具有真实医学意义，
    # 贸然删除会损失高危患者样本或扭曲临床风险分布：
    # 1) 应优先保留的临床有意义极端值：
    #    - serum_creatinine（血肌酐）：偏高反映肾功能受损/肾衰，是心衰预后的强预测因子，应保留；
    #    - ejection_fraction（射血分数）：极端偏低（如 HFrEF 重度收缩功能不全）或偏高均具临床判别价值，应保留；
    #    - serum_sodium（血钠）：低钠血症（如 <135 mmol/L，甚至 ~113 mmol/L）常见于重度心衰与容量超负荷/利尿剂相关电解质紊乱，应保留；
    #    - platelets（血小板）：极高/极低可能与合并症或血液状态相关，临床上通常保留；
    #    - creatinine_phosphokinase（CPK）：急性心肌损伤/骨骼肌损伤可致显著升高，但右偏极度严重时若用于线性模型，可在交叉验证/训练折内视情况做对数变换或谨慎 Winsorize，而非盲目整行删除。
    # 2) 何时考虑 Winsorize 截断或删除：
    #    - 仅当经临床/数据源核实属于录入错误、单位错误或明显不可能的生理数值时才考虑删除或截断；
    #    - 若使用对极端值敏感的线性模型或神经网络，且希望抑制极重度右偏长尾（如 creatinine_phosphokinase 达 7861、serum_creatinine 达 9.4），
    #      建议仅在训练集内部尝试适度 Winsorize（或在树模型之外配合特征变换）并与“不截断基线”做交叉验证对比，避免直接丢弃真实临床样本。
    advice_text = (
        "\n--- 异常值处理建议（临床与建模视角） ---\n"
        "1. 建议保留的临床极端值：\n"
        "   - serum_creatinine（升高提示肾功能恶化）、serum_sodium（偏低提示低钠血症/病情危重）、\n"
        "     ejection_fraction（心功能分级核心指标）、platelets 等在心衰患者中往往代表真实高危状态，\n"
        "     盲目删除会丢失关键预后信息。\n"
        "2. 谨慎使用 Winsorize 截断或删除的场景：\n"
        "   - 仅在确认存在明显录入/仪器错误，或针对对极值高度敏感的线性/神经网络模型做消融对比时，\n"
        "     才在训练集内谨慎评估 Winsorize 截断；树模型天然对单调缩放与孤立极值具有较强鲁棒性，"
        "通常无需因 IQR 异常值删样本。"
    )
    print(advice_text)

    return pd.DataFrame(outlier_summary)


def prepare_scaled_and_unscaled_splits(df: pd.DataFrame) -> dict:
    """3. 分层划分训练/测试集，并构建未标准化与 StandardScaler 标准化特征矩阵"""
    print("\n" + "=" * 72)
    print("【步骤 3】数据集划分与特征标准化（严格防信息泄露）")
    print("=" * 72)

    # 模型类型与标准化必要性说明：
    # - 逻辑回归（Logistic Regression）、MLP、SVM、KNN 依赖特征内积、距离度量或梯度更新，
    #   各特征量纲差异巨大（例如 platelets 量级在 10^5，而 serum_creatinine 约为 0.5~9.4），
    #   若不标准化，大数值特征会主导距离计算或导致梯度下降震荡/收敛缓慢，因此【需要 StandardScaler】。
    # - 随机森林（Random Forest）、XGBoost 等基于决策树分裂的模型，每次分裂仅比较单个特征与阈值的大小关系，
    #   对特征的单调线性缩放/平移具有不变性，因此【不需要标准化】。
    scaling_explanation = (
        "--- 模型标准化必要性说明 ---\n"
        "• 需要 StandardScaler 的模型（逻辑回归、MLP、SVM、KNN）：\n"
        "  这些模型涉及距离计算、核函数/点积或基于梯度的参数优化，特征量纲差异大会导致权重失衡或收敛困难。\n"
        "• 不需要 StandardScaler 的模型（随机森林、XGBoost）：\n"
        "  树模型通过特征阈值进行节点分裂，对特征尺度变换不敏感，使用原始未标准化特征即可。"
    )
    print(scaling_explanation)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # 先做分层划分（test_size=0.2, random_state=42, stratify=y），保证训练/测试集标签分布一致
    # 且必须在划分之后再 fit StandardScaler，严防测试集分布信息泄露到训练集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 仅在训练集 X_train_raw 上 fit，随后分别 transform 训练集和测试集
    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train_raw)
    X_test_scaled_arr = scaler.transform(X_test_raw)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_arr, index=X_train_raw.index, columns=FEATURE_COLS
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_arr, index=X_test_raw.index, columns=FEATURE_COLS
    )

    print("\n--- 划分与标准化核对结果 ---")
    print(f"训练集样本数: {len(X_train_raw)} | 测试集样本数: {len(X_test_raw)} (test_size=0.2, stratify=y)")
    print(f"训练集 DEATH_EVENT=1 占比: {y_train.mean() * 100:.2f}% | 测试集占比: {y_test.mean() * 100:.2f}%")
    print(
        "标准化后训练集均值（应接近 0）最大绝对值: "
        f"{np.abs(X_train_scaled.mean()).max():.2e}"
    )
    print(
        "标准化后训练集标准差（应接近 1）与 1 的最大偏差: "
        f"{np.abs(X_train_scaled.std(ddof=0) - 1.0).max():.2e}"
    )
    print("已保留两套特征矩阵：")
    print("  1) 未标准化矩阵（供随机森林 / XGBoost 等树模型使用）: X_train_raw, X_test_raw")
    print("  2) 标准化后矩阵（供逻辑回归 / MLP / SVM / KNN 使用）: X_train_scaled, X_test_scaled")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


def main() -> None:
    df = load_and_describe(DATA_FILE, RESULTS_DIR)
    _ = check_missing_and_outliers(df, RESULTS_DIR)
    data_bundle = prepare_scaled_and_unscaled_splits(df)

    print("\n" + "=" * 72)
    print("数据初处理流程全部完成！输出文件位于 results/ 目录。")
    print(f"当前内存中可用矩阵键名: {list(data_bundle.keys())}")
    print("=" * 72)


if __name__ == "__main__":
    main()