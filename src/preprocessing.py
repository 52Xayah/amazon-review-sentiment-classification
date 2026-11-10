
def clean_reviews(df):
    """
    清洗 Amazon 评论数据。

    清洗步骤：
    1. 删除 Text、Score 缺失的数据
    2. 删除空评论
    3. 删除重复评论
    4. 将 Score 转换为整数

    Returns:
        pandas.DataFrame: 清洗后的评论数据
    """
    df = df.copy()

    # 删除核心字段缺失的数据
    df = df.dropna(subset=["Text", "Score"])

    # 删除空评论
    df["Text"] = df["Text"].astype(str).str.strip()
    df = df[df["Text"] != ""]

    # 删除重复评论
    df = df.drop_duplicates(
        subset=["UserId", "ProductId", "Text"]
    )

    # 评分转整数
    df["Score"] = df["Score"].astype(int)

    return df
