
from pathlib import Path
import pandas as pd


def load_reviews(file_path):
    """读取 Amazon 评论 CSV 数据。"""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    return pd.read_csv(file_path)


def save_reviews(df, file_path):
    """保存处理后的评论数据。"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
