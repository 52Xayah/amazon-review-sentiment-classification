
from sklearn.linear_model import LogisticRegression


def create_logistic_regression():
    """创建项目使用的逻辑回归分类器。"""
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )


def train_logistic_regression(X_train, y_train):
    """训练逻辑回归模型。"""
    model = create_logistic_regression()
    model.fit(X_train, y_train)
    return model
