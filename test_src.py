
from pathlib import Path

from src.data_loader import load_reviews, save_reviews
from src.preprocessing import clean_reviews
from src.feature_extraction import create_tfidf_vectorizer
from src.models.logistic_regression import create_logistic_regression
from src.models.lstm import create_lstm_model
from src.evaluation import evaluate_model

print("1. 模块导入成功")

# 检查原始数据读取和清洗
root = Path(__file__).resolve().parent
df = load_reviews(root / "data/raw/Reviews.csv")
cleaned = clean_reviews(df)

print("2. 数据清洗完成：", cleaned.shape)
assert cleaned.shape == (567145, 10)

# 检查模型和特征提取器能否创建
vectorizer = create_tfidf_vectorizer()
lr_model = create_logistic_regression()
lstm_model = create_lstm_model()

print("3. TF-IDF 创建成功")
print("4. Logistic Regression 创建成功")
print("5. LSTM 创建成功")
print("6. Evaluation 模块导入成功")

print("\n所有 src 模块验证通过！")
