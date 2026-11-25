# -*- coding: utf-8 -*-
"""
心力衰竭临床记录数据集探索与特征重要性分析脚本
功能：
1. 计算所有数值字段相关系数矩阵并绘制下三角热力图（保存为 correlation_heatmap.png）
2. 对连续临床指标执行独立样本 t 检验（含 Levene 方差齐性检验）并训练 RandomForestClassifier
   计算特征重要性，合并生成 factor_ranking.csv
3. 输出综合排名前三的指标及医学层面的初步关联说明（仅作统计与文献背景描述，非诊疗建议）
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 固定全局随机种子，保证实验可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 常见二值/分类特征列表（其余数值型临床特征视为连续临床指标）
BINARY_COLS = {"anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"}


def configure_chinese_font() -> None:
    """配置 matplotlib 中文字体，避免中文标题与坐标标签出现方块乱码"""
    candidate_fonts = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Droid Sans Fallback",
    ]
    available_font_names = {f.name for f in fm.fontManager.ttflist}
    chosen_fonts = [name for name in candidate_fonts if name in available_font_names]

    if chosen_fonts:
        plt.rcParams["font.sans-serif"] = chosen_fonts + ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False


def load_dataset(csv_path: Path, target_col: str = "DEATH_EVENT") -> tuple[pd.DataFrame, str]:
    """加载数据集并确认标签列（优先匹配 target_col，若不存在则回退到第一列）"""
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    df = pd.read_csv(csv_path)
    resolved_target = target_col if target_col in df.columns else df.columns[0]
    print(f"[数据加载] 文件: {csv_path} | 维度: {df.shape[0]} 行 x {df.shape[1]} 列 | 标签列: {resolved_target}")
    return df, resolved_target


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Path = Path("correlation_heatmap.png"),
) -> pd.DataFrame:
    """一、计算数值特征（含标签）相关系数矩阵，绘制掩码上三角的 coolwarm 热力图"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    # 构造上三角掩码，仅显示下三角与对角线之外的非冗余区域
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "皮尔逊相关系数"},
        ax=ax,
    )
    ax.set_title("心力衰竭临床记录数值特征相关性热力图", fontsize=14, pad=14)
    ax.set_xlabel("临床特征", fontsize=11, labelpad=10)
    ax.set_ylabel("临床特征", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[相关性热力图] 已保存至: {output_path}")
    return corr_matrix


def run_ttest_and_rf_ranking(
    df: pd.DataFrame,
    target_col: str = "DEATH_EVENT",
    output_csv: Path = Path("factor_ranking.csv"),
) -> pd.DataFrame:
    """二、对连续临床指标执行独立样本 t 检验并结合随机森林重要性生成综合排名表"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除标签列与二值列，提取连续临床指标（如 age, creatinine_phosphokinase, ejection_fraction,
    # platelets, serum_creatinine, serum_sodium, time 等）
    continuous_cols = [
        col for col in numeric_cols if col != target_col and col not in BINARY_COLS
    ]
    feature_cols = [col for col in numeric_cols if col != target_col]

    group_death = df[df[target_col] == 1]
    group_survive = df[df[target_col] == 0]

    ttest_records = []
    for col in continuous_cols:
        s_death = group_death[col].dropna()
        s_survive = group_survive[col].dropna()

        # 1. 先做 Levene 方差齐性检验（显著性水平 alpha = 0.05）
        _, levene_p = stats.levene(s_death, s_survive, center="median")
        is_equal_var = bool(levene_p > 0.05)

        # 2. 根据方差齐性结果选择 Student t 检验或 Welch t 检验
        t_stat, p_val = stats.ttest_ind(s_death, s_survive, equal_var=is_equal_var)

        ttest_records.append(
            {
                "指标名": col,
                "死亡组均值(1)": round(float(s_death.mean()), 4),
                "存活组均值(0)": round(float(s_survive.mean()), 4),
                "Levene_p值": round(float(levene_p), 6),
                "方差齐性": "是(Student)" if is_equal_var else "否(Welch)",
                "t值": round(float(t_stat), 4),
                "p值": float(p_val),
            }
        )

    ttest_df = pd.DataFrame(ttest_records)

    # 3. 使用随机森林分类器训练标签，获取特征重要性得分
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_df = pd.DataFrame(
        {
            "指标名": feature_cols,
            "重要性得分": rf_model.feature_importances_,
        }
    )

    # 4. 合并连续临床指标的 t 检验与随机森林重要性结果，并计算综合排名
    ranking_df = pd.merge(ttest_df, rf_df, on="指标名", how="left")
    ranking_df["p值排名"] = ranking_df["p值"].rank(ascending=True, method="min").astype(int)
    ranking_df["重要性排名"] = (
        ranking_df["重要性得分"].rank(ascending=False, method="min").astype(int)
    )
    ranking_df["平均名次"] = (ranking_df["p值排名"] + ranking_df["重要性排名"]) / 2.0

    ranking_df = ranking_df.sort_values(
        by=["平均名次", "p值", "重要性得分"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    ranking_df["排名"] = ranking_df.index + 1

    # 整理输出列顺序与格式
    ordered_cols = [
        "排名",
        "指标名",
        "死亡组均值(1)",
        "存活组均值(0)",
        "方差齐性",
        "t值",
        "p值",
        "重要性得分",
        "p值排名",
        "重要性排名",
        "平均名次",
    ]
    ranking_out = ranking_df[ordered_cols].copy()
    ranking_out["重要性得分"] = ranking_out["重要性得分"].round(6)
    ranking_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n--- 连续临床指标 T 检验与随机森林重要性综合排名 ---")
    print(ranking_out.to_string(index=False))
    print(f"\n[排名结果] 已保存至: {output_csv}")

    return ranking_df


def print_top3_conclusions(ranking_df: pd.DataFrame) -> None:
    """三、按综合排名输出前三名指标，并给出医学背景下的初步关联解释（非诊疗建议）"""
    medical_notes = {
        "time": (
            "在该队列中，随访观察时长（time）与死亡事件呈显著负相关，死亡组平均随访时间明显短于存活组，"
            "提示随访期内的事件发生与观察时间窗口密切相关。在统计与模型分析中，随访时长反映的是观察周期长度而非基线生理病理指标，"
            "因此在解释风险因素时需将其与临床基线指标区分看待。"
        ),
        "ejection_fraction": (
            "左心室射血分数（ejection_fraction）降低反映心脏每次收缩泵出的血液比例减少，通常与心室收缩功能受损有关。"
            "在心力衰竭相关研究中，较低的射血分数往往与较高的泵衰竭及不良结局风险相关，死亡组均值低于存活组。"
            "该指标是评估心功能状态的重要客观参数之一，但具体临床解读仍需结合患者整体病情。"
        ),
        "serum_creatinine": (
            "血清肌酐（serum_creatinine）升高常提示肾小球滤过功能下降或存在肾功能不全，在心力衰竭人群中与心肾相互作用密切相关。"
            "较高水平的血清肌酐通常与病情严重程度增加及较高的死亡风险相关，死亡组均值高于存活组。"
            "在临床研究中，肾功能指标常被用作评估心衰患者预后分层的重要参考维度。"
        ),
        "age": (
            "年龄（age）增长通常伴随心血管系统退行性改变及合并症负担增加，高龄心衰患者的生理储备相对较低。"
            "在队列统计中，较高年龄往往与较高的死亡风险相关，死亡组平均年龄高于存活组。"
            "年龄作为基础人口学特征，常与其他临床指标共同影响预后风险评估。"
        ),
    }

    print("\n" + "=" * 72)
    print("三、综合排名前三指标及医学初步解释（仅作统计关联说明，非诊疗建议）")
    print("=" * 72)

    top3 = ranking_df.head(3)
    for _, row in top3.iterrows():
        feat = row["指标名"]
        rank_num = int(row["排名"])
        p_val = row["p值"]
        imp_val = row["重要性得分"]
        explanation = medical_notes.get(
            feat,
            f"指标 {feat} 在死亡组与存活组之间存在差异（p={p_val:.4e}，随机森林重要性={imp_val:.4f}），"
            f"提示其在该数据集中与死亡风险相关，具体临床意义需结合医学背景综合判断。",
        )
        print(
            f"\n第 {rank_num} 名：{feat} "
            f"(死亡组均值={row['死亡组均值(1)']}, 存活组均值={row['存活组均值(0)']}, "
            f"t={row['t值']}, p={p_val:.2e}, RF重要性={imp_val:.4f})"
        )
        print(explanation)


def main() -> None:
    configure_chinese_font()
    csv_file = Path("heart_failure_clinical_records_dataset.csv")
    df, target_col = load_dataset(csv_file, target_col="DEATH_EVENT")
    _ = plot_correlation_heatmap(df, output_path=Path("correlation_heatmap.png"))
    ranking_df = run_ttest_and_rf_ranking(
        df,
        target_col=target_col,
        output_csv=Path("factor_ranking.csv"),
    )
    print_top3_conclusions(ranking_df)


if __name__ == "__main__":
    main()