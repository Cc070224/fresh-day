from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = Path("dataset/dataset/avocado_shelf_life.csv")
MODEL_PATH = Path("models/fruit_model.joblib")
FEATURES = ["storage_group", "ripening_stage"]
TARGET = "days_to_overripe"


def main():
    # 데이터 파일을 읽고 실제 구조를 확인합니다.
    data = pd.read_csv(DATA_PATH)
    print("실제 열 이름:")
    print(data.columns.tolist())
    print("\n데이터 형식:")
    print(data.dtypes)
    print("\n열별 결측값 개수:")
    print(data.isna().sum())
    print(f"\n중복 행 개수: {data.duplicated().sum()}")

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {missing_columns}")

    if data[required_columns].isna().any().any():
        raise ValueError("입력 특성 또는 예측 목표에 결측값이 있습니다.")

    # fruit_id와 같은 식별자는 예측 특성에서 제외합니다.
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 저장 조건은 원-핫 인코딩하고 숙성 단계는 숫자형으로 유지합니다.
    preprocessing = ColumnTransformer(
        transformers=[
            ("storage", OneHotEncoder(handle_unknown="ignore"), ["storage_group"]),
            ("stage", "passthrough", ["ripening_stage"]),
        ]
    )

    model_grids = {
        "LinearRegression": (
            LinearRegression(),
            {"regressor__fit_intercept": [True, False]},
        ),
        "DecisionTreeRegressor": (
            DecisionTreeRegressor(random_state=42),
            {
                "regressor__max_depth": [None, 3, 5, 7],
                "regressor__min_samples_leaf": [1, 3, 5],
            },
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=42),
            {
                "regressor__n_estimators": [50, 100],
                "regressor__max_depth": [None, 3, 5],
                "regressor__min_samples_leaf": [1, 3, 5],
            },
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {
                "regressor__n_estimators": [50, 100],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [2, 3],
                "regressor__min_samples_leaf": [1, 3],
            },
        ),
    }

    # 학습 데이터에서 5겹 교차 검증으로 각 모델의 하이퍼파라미터를 비교합니다.
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)
    search_results = {}
    for model_name, (regressor, parameter_grid) in model_grids.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("regressor", regressor),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="r2",
            cv=cross_validation,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        search_results[model_name] = search
        print(
            f"\n{model_name} 5겹 교차 검증 R²: "
            f"{search.best_score_:.4f}"
        )
        print(f"최적 하이퍼파라미터: {search.best_params_}")

    best_name, best_search = max(
        search_results.items(), key=lambda item: item[1].best_score_
    )
    final_model = best_search.best_estimator_
    predictions = final_model.predict(X_test)
    test_r2 = r2_score(y_test, predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\n선택된 최종 모델: {best_name}")
    print(f"테스트 세트 R²: {test_r2:.4f}")
    print(f"테스트 세트 RMSE: {test_rmse:.4f}일")

    # 최종 모델에는 전처리 과정도 포함되어 있어 새 데이터에 그대로 사용할 수 있습니다.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"모델 저장 위치: {MODEL_PATH}")


if __name__ == "__main__":
    main()
