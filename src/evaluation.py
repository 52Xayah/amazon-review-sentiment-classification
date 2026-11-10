
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """计算分类模型的主要评价指标。"""
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro"
        ),
        "Weighted F1": f1_score(
            y_true, y_pred, average="weighted"
        )
    }

    print(f"\n{model_name} Evaluation")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return results


def save_comparison(results, file_path):
    """保存模型对比结果。"""
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    return df
