
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer():
    """创建项目使用的 TF-IDF 特征提取器。"""
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )


def extract_tfidf_features(X_train, X_test):
    """训练 TF-IDF 并转换训练集、测试集。"""
    vectorizer = create_tfidf_vectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer
