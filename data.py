from datasets import load_dataset
import pandas as pd

def extract_ccbench_subset(
    save_path: str = "ccbench_data_analysis_deepseek_claude.csv",
    category: str = "data_analysis",
    models_of_interest = ["DeepSeek-V3.1-Terminus", "Claude-Sonnet-4"]
):

    print("🚀 正在加载 CC-Bench-trajectories 数据集...")
    ds = load_dataset("zai-org/CC-Bench-trajectories", split="train", verification_mode="no_checks")

    print(f"📊 数据总量：{len(ds)} 条")
    print("可用字段：", list(ds.features.keys()))

    # Step 1: 筛选任务类别
    subset_category = ds.filter(lambda x: x["task_category"] == category)
    print(f"✅ 已筛选任务类别 '{category}'，剩余 {len(subset_category)} 条")

    # Step 2: 筛选指定模型
    subset_models = subset_category.filter(lambda x: x["model_name"] in models_of_interest)
    print(f"✅ 已筛选模型 {models_of_interest}，剩余 {len(subset_models)} 条")

    # Step 3: 转换为 pandas DataFrame
    df = pd.DataFrame(subset_models)

    # Step 4: 导出为 CSV
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"💾 已保存到 {save_path}")

    # Step 5: 返回数据集信息摘要
    summary = df.groupby("model_name")["task_id"].count()
    print("\n模型样本分布:")
    print(summary)
    return df


if __name__ == "__main__":
    df = extract_ccbench_subset()
