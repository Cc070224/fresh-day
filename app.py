from pathlib import Path

import joblib
import pandas as pd


DATA_PATH = Path("dataset/dataset/avocado_shelf_life.csv")
MODEL_PATH = Path("models/fruit_model.joblib")


def choose_option(options, labels):
    for number, value in enumerate(options, start=1):
        print(f"{number}. {labels.get(value, value)}")

    while True:
        choice = input("번호를 선택하세요: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("목록에 있는 번호를 입력해 주세요.")


def main():
    # 데이터에서 입력 특성의 실제 값과 자료형을 확인합니다.
    data = pd.read_csv(DATA_PATH)
    storage_options = sorted(data["storage_group"].dropna().unique())
    stage_options = sorted(data["ripening_stage"].dropna().unique())
    print("저장 환경 실제 값:", storage_options)
    print("숙성 단계 실제 값:", stage_options)
    print("입력 특성 자료형:")
    print(data[["storage_group", "ripening_stage"]].dtypes)

    # 학습된 모델만 불러오며, 앱에서 모델을 다시 학습하지 않습니다.
    if not MODEL_PATH.exists():
        raise FileNotFoundError("모델 파일이 없습니다. 먼저 train.py를 실행해 주세요.")
    model = joblib.load(MODEL_PATH)

    storage_labels = {
        "T10": "10°C 보관 (상대습도 85%)",
        "T20": "20°C 보관 (상대습도 85%)",
        "Tam": "연구 당시 주변 환경 보관",
    }
    stage_labels = {stage: f"숙성 단계 {stage}" for stage in stage_options}

    print("\n저장 환경을 선택하세요:")
    storage_group = choose_option(storage_options, storage_labels)
    print("\n현재 숙성 정도를 선택하세요:")
    ripening_stage = choose_option(stage_options, stage_labels)

    # 학습 때 사용한 열 이름과 순서로 입력 데이터를 구성합니다.
    input_data = pd.DataFrame(
        [
            {
                "storage_group": storage_group,
                "ripening_stage": ripening_stage,
            }
        ]
    )
    remaining_days = float(model.predict(input_data)[0])

    print(f"\n과숙까지 예상 남은 기간: 약 {remaining_days:.1f}일")
    print("이 결과는 품질 관리 참고용이며, 식품 안전 기한을 의미하지 않습니다.")


if __name__ == "__main__":
    main()
