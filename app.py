from datetime import date
from pathlib import Path

import joblib
import pandas as pd


AVOCADO_DATA_PATH = Path("dataset/dataset/avocado_shelf_life.csv")
REFERENCE_DATA_PATH = Path("dataset/dataset/fruit_storage_reference.csv")
MODEL_PATH = Path("models/fruit_model.joblib")


def choose_option(options, labels):
    if not options:
        raise ValueError("선택할 항목이 없습니다.")

    for number, value in enumerate(options, start=1):
        print(f"{number}. {labels.get(value, value)}")

    while True:
        choice = input("번호를 선택하세요: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("목록에 있는 번호를 입력해 주세요.")


def read_integer(prompt, minimum):
    while True:
        value = input(prompt).strip()
        if value.isdigit() and int(value) >= minimum:
            return int(value)
        print(f"{minimum} 이상의 정수를 입력해 주세요.")


def read_purchase_date():
    while True:
        value = input("구매 날짜를 입력하세요 (YYYY-MM-DD): ").strip()
        try:
            purchased_on = date.fromisoformat(value)
        except ValueError:
            print("날짜를 YYYY-MM-DD 형식으로 입력해 주세요.")
            continue
        if purchased_on <= date.today():
            return purchased_on
        print("구매 날짜는 오늘 이후일 수 없습니다.")


def feature_one():
    # 데이터에서 입력 특성의 실제 값과 자료형을 확인합니다.
    data = pd.read_csv(AVOCADO_DATA_PATH)
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


def show_storage_advice(row, storage_method):
    storage_labels = {"room": "실온", "refrigerator": "냉장"}
    print(f"\n선택한 보관 방법: {storage_labels[storage_method]} 보관")

    note = row["condition_note"]
    note_translations = {
        "room 50–70 F; refrigerator <=40 F":
            "실온 50~70°F, 냉장고 40°F 이하를 기준으로 한 안내입니다.",
        "room until ripe; refrigerator guidance":
            "실온에서 익힌 뒤 냉장 보관하는 참고 안내입니다.",
        "room not recommended": "실온 보관은 권장되지 않습니다.",
        "group includes oranges, grapefruit, clementines":
            "이 그룹에는 오렌지, 자몽, 클레멘타인이 포함됩니다.",
        "room duration only AFTER ripe; group guidance":
            "실온 보관 기간은 익은 뒤에만 적용되며, 과일 그룹 참고 기준입니다.",
        "room duration only AFTER ripe":
            "실온 보관 기간은 익은 뒤를 기준으로 합니다.",
        "room until ripe; refrigerated whole uncut":
            "익을 때까지 실온에 둔 뒤, 자르지 않은 통과 상태로 냉장 보관하세요.",
    }
    if pd.notna(note) and str(note).strip():
        print("보관 참고:", note_translations.get(str(note), "CSV에 별도 보관 안내가 있습니다."))
    else:
        print("CSV에 별도의 보관 조건 안내가 없습니다.")


def show_daily_plan(remaining_quantity, remaining_days):
    if remaining_quantity == 0:
        print("남은 과일이 없어 식용 계획은 없습니다.")
    elif remaining_days <= 0:
        print("참고 기간이 지났거나 남은 기간이 없습니다. 과일 상태를 확인해 주세요.")
        print("섭취를 강제 권장하지 않으며, 상태가 의심스러우면 먹지 마세요.")
    else:
        daily_amount = remaining_quantity / remaining_days
        if daily_amount < 1:
            interval = remaining_days / remaining_quantity
            lower_days = int(interval)
            upper_days = lower_days if interval == lower_days else lower_days + 1
            if lower_days == upper_days:
                print(f"낭비를 줄이기 위한 참고 계획: 약 {lower_days}일마다 1개")
            else:
                print(f"낭비를 줄이기 위한 참고 계획: 약 {lower_days}~{upper_days}일마다 1개")
        elif daily_amount == int(daily_amount):
            print(f"낭비를 줄이기 위한 참고 계획: 하루 약 {int(daily_amount)}개")
        else:
            lower_amount = int(daily_amount)
            print(
                f"낭비를 줄이기 위한 참고 계획: 하루 약 "
                f"{lower_amount}~{lower_amount + 1}개"
            )
        print("개인 상황과 과일 상태에 맞춰 조정하세요. 반드시 먹어야 하는 양은 아닙니다.")


def feature_two():
    # 저장 참고 자료의 실제 열, 자료형, 과일 종류와 기간 단위를 확인합니다.
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    required_columns = [
        "fruit_or_group",
        "room_min_days",
        "room_max_days",
        "refrigerator_min_days",
        "refrigerator_max_days",
        "condition_note",
        "source_url",
    ]
    missing_columns = [column for column in required_columns if column not in reference.columns]
    if missing_columns:
        raise ValueError(f"저장 참고 자료에 필요한 열이 없습니다: {missing_columns}")

    fruit_labels = {
        "apple": "사과",
        "avocado": "아보카도",
        "banana": "바나나",
        "blueberry": "블루베리",
        "citrus_group": "감귤류 그룹",
        "grape": "포도",
        "peach_nectarine_plum_pear_group": "복숭아·천도복숭아·자두·배 그룹",
        "strawberry": "딸기",
        "pineapple": "파인애플",
        "melon_group": "멜론 그룹",
    }
    print("\n저장 참고 자료의 실제 열:")
    print(reference.columns.tolist())
    print("자료형:")
    print(reference.dtypes)
    fruit_options = reference["fruit_or_group"].dropna().tolist()
    print("과일 종류 및 그룹:", [fruit_labels.get(value, value) for value in fruit_options])
    print("저장 기간 열의 단위는 일입니다. 빈 값은 해당 방식의 참고 기간이 없다는 뜻입니다.")
    print("\n과일 종류를 선택하세요:")
    fruit = choose_option(fruit_options, fruit_labels)

    purchased_on = read_purchase_date()
    purchased_quantity = read_integer("구매한 수량을 입력하세요 (개): ", 1)
    while True:
        remaining_quantity = read_integer("현재 남은 수량을 입력하세요 (개): ", 0)
        if remaining_quantity <= purchased_quantity:
            break
        print("남은 수량은 구매 수량보다 많을 수 없습니다.")

    storage_method = choose_option(
        ["room", "refrigerator"],
        {"room": "실온 보관", "refrigerator": "냉장 보관"},
    )
    appearance = choose_option(
        ["good", "minor_damage", "rotten"],
        {
            "good": "외관이 양호함",
            "minor_damage": "가벼운 긁힘 또는 멍이 있음",
            "rotten": "눈에 띄게 썩었거나 곰팡이가 있음",
        },
    )

    matched_rows = reference[reference["fruit_or_group"] == fruit]
    if matched_rows.empty:
        raise ValueError("선택한 과일의 저장 참고 자료를 찾을 수 없습니다.")
    row = matched_rows.iloc[0]
    show_storage_advice(row, storage_method)
    print("참고 자료 출처:", row["source_url"])

    if appearance == "minor_damage":
        print("가벼운 손상이 있어 먼저 상태를 확인하고, 먹을 수 있는 상태라면 우선 이용을 고려하세요.")
    elif appearance == "rotten":
        print("눈에 띄는 부패나 곰팡이가 있으면 먹지 마세요. 섭취 계획은 안내하지 않습니다.")
        return

    elapsed_days = (date.today() - purchased_on).days
    print(f"구매 후 경과일: {elapsed_days}일")

    if fruit == "avocado":
        avocado_data = pd.read_csv(AVOCADO_DATA_PATH)
        storage_options = sorted(avocado_data["storage_group"].dropna().unique())
        stage_options = sorted(avocado_data["ripening_stage"].dropna().unique())
        storage_labels = {
            "T10": "10°C·상대습도 85% (연구 실험 조건)",
            "T20": "20°C·상대습도 85% (연구 실험 조건)",
            "Tam": "연구 당시 주변 환경 (실험 조건)",
        }
        stage_labels = {stage: f"숙성 단계 {stage}" for stage in stage_options}

        if not MODEL_PATH.exists():
            raise FileNotFoundError("모델 파일이 없습니다. 먼저 train.py를 실행해 주세요.")
        model = joblib.load(MODEL_PATH)

        print("\n아보카도 모델의 실험 저장 조건을 선택하세요:")
        model_storage = choose_option(storage_options, storage_labels)
        print("\n현재 숙성 정도를 선택하세요:")
        ripening_stage = choose_option(stage_options, stage_labels)

        input_data = pd.DataFrame(
            [{"storage_group": model_storage, "ripening_stage": ripening_stage}]
        )
        remaining_days = float(model.predict(input_data)[0])
        print(f"\n머신러닝 예측값: 과숙까지 약 {remaining_days:.1f}일")
        print("예측은 현재 숙성 단계 기준이며, 구매 후 경과일을 다시 빼지 않았습니다.")
        print("연구 실험 조건은 가정의 실온·냉장 조건과 같다고 볼 수 없습니다.")
        print("예측 결과는 품질 관리 참고용이며 식품 안전 기한이 아닙니다.")
        show_daily_plan(remaining_quantity, remaining_days)
        return

    storage_prefix = "room" if storage_method == "room" else "refrigerator"
    minimum_days = row[f"{storage_prefix}_min_days"]
    maximum_days = row[f"{storage_prefix}_max_days"]
    if pd.isna(minimum_days):
        if pd.isna(maximum_days):
            print("선택한 보관 방식의 참고 기간이 자료에 없습니다. 일일 계획을 계산할 수 없습니다.")
        else:
            print(f"공개 자료의 최대 참고 기간: {float(maximum_days):g}일")
            print("최소 기간이 자료에 없어 보수적인 일일 계획은 계산하지 않습니다.")
        return

    minimum_days = float(minimum_days)
    if pd.notna(maximum_days):
        maximum_days = float(maximum_days)
        print(f"\n공개 자료의 참고 기간: {minimum_days:g}~{maximum_days:g}일")
    else:
        print(f"\n공개 자료의 최소 참고 기간: {minimum_days:g}일")

    condition_note = row["condition_note"]
    if (
        storage_method == "room"
        and pd.notna(condition_note)
        and "AFTER ripe" in str(condition_note)
    ):
        print("이 실온 참고 기간은 과일이 익은 뒤부터 적용됩니다.")
        print("익은 날짜를 입력받지 않았으므로 구매일 기준 남은 기간과 일일 계획은 계산하지 않습니다.")
        return

    remaining_days = minimum_days - elapsed_days
    print(f"보수적으로 계산한 남은 참고 기간: 약 {remaining_days:.1f}일")
    show_daily_plan(remaining_quantity, remaining_days)


def read_positive_number(prompt):
    while True:
        try:
            value = float(input(prompt).strip())
        except ValueError:
            print("0보다 큰 숫자를 입력해 주세요.")
            continue
        if pd.notna(value) and value > 0 and value != float("inf"):
            return value
        print("0보다 큰 숫자를 입력해 주세요.")


def feature_three():
    nutrition = pd.read_csv("dataset/dataset/fruit_nutrition.csv")
    required_columns = ["fruit", "name_ko", "edible_serving_g"]
    missing_columns = [column for column in required_columns if column not in nutrition.columns]
    if missing_columns:
        raise ValueError(f"영양 자료에 필요한 열이 없습니다: {missing_columns}")

    fruit_options = nutrition["fruit"].dropna().tolist()
    fruit_labels = dict(zip(nutrition["fruit"], nutrition["name_ko"]))
    nutrient_labels = {
        "kcal_per_serving": ("열량", "kcal"),
        "sugar_g_per_serving": ("당류", "g"),
        "vitamin_c_pct_DV_per_serving_as_published": ("비타민 C (하루 기준치)", "%"),
        "fiber_g_per_serving": ("식이섬유", "g"),
        "potassium_mg_per_serving": ("칼륨", "mg"),
        "protein_g_per_serving": ("단백질", "g"),
    }
    totals = {column: 0.0 for column in nutrient_labels}
    total_counts = {column: 0 for column in nutrient_labels}
    fruit_count = 0

    print("\n과일 영양 분석은 CSV의 1회 제공량 기준으로 계산합니다.")
    print("실제 과일 한 개의 무게 자료가 없어, 수량은 기준 제공량 횟수로 입력합니다.")
    print("무게는 껍질과 씨 등을 제외한 식용 부분의 무게로 입력해 주세요.")
    print("비타민 C는 자료에 공개된 하루 기준치(%)이며 mg 단위가 아닙니다.")

    while True:
        print("\n먹은 과일을 선택하세요:")
        fruit = choose_option(fruit_options, fruit_labels)
        row = nutrition[nutrition["fruit"] == fruit].iloc[0]
        fruit_name = fruit_labels.get(fruit, fruit)
        if pd.isna(fruit_name):
            fruit_name = fruit
        if pd.isna(row["edible_serving_g"]) or float(row["edible_serving_g"]) <= 0:
            raise ValueError(f"{fruit_name}의 1회 제공량 자료가 없어 계산할 수 없습니다.")
        serving_grams = float(row["edible_serving_g"])

        amount_mode = choose_option(
            ["servings", "grams"],
            {
                "servings": f"기준 제공량 횟수 (1회 = {serving_grams:g}g)",
                "grams": "식용 부분의 무게 (g)",
            },
        )
        if amount_mode == "servings":
            amount = read_positive_number("먹은 기준 제공량 횟수를 입력하세요: ")
            scale = amount
            eaten_grams = amount * serving_grams
        else:
            eaten_grams = read_positive_number("먹은 식용 부분 무게를 입력하세요 (g): ")
            scale = eaten_grams / serving_grams

        print(f"\n{fruit_name} 입력량: 약 {eaten_grams:.1f}g")
        print("입력량 기준 주요 영양 특징:")
        fruit_count += 1
        for column, (label, unit) in nutrient_labels.items():
            if column in nutrition.columns and pd.notna(row[column]):
                value = float(row[column]) * scale
                totals[column] += value
                total_counts[column] += 1
                print(f"- {label}: {value:.1f} {unit}")

        if "source_url" in nutrition.columns and pd.notna(row.get("source_url")):
            print("영양 정보 출처:", row["source_url"])

        add_another = choose_option(
            [True, False],
            {True: "예, 다른 과일도 입력", False: "아니요, 합계 보기"},
        )
        if not add_another:
            break

    print("\n=== 이번 섭취의 영양 합계 ===")
    displayed_total = False
    for column, (label, unit) in nutrient_labels.items():
        if total_counts[column] == fruit_count:
            print(f"- {label}: {totals[column]:.1f} {unit}")
            displayed_total = True
    if not displayed_total:
        print("모든 입력 과일에 공통으로 제공된 영양 합계가 없습니다.")
    print("자료에 없는 영양 항목은 추정하지 않았습니다.")


def main():
    menu_options = [1, 2, 3, 4]
    menu_labels = {
        1: "아보카도 남은 보관 기간 예측",
        2: "과일 섭취 계획 및 보관 관리",
        3: "과일 영양 분석",
        4: "종료",
    }
    while True:
        print("\n=== Fresh Day ===")
        choice = choose_option(menu_options, menu_labels)
        if choice == 1:
            feature_one()
        elif choice == 2:
            feature_two()
        elif choice == 3:
            feature_three()
        else:
            print("프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    main()
