import os
import sqlite3
import pandas as pd

from unknown_unknown import Estimator
from unknown_unknown.data_item import DataItem
from unknown_unknown.estimator import BoundType, CountEstimator, ValueEstimator

true_db = "../data/unknown/estat_tps00001.db"
folder = "../data/unknown/split_n_10_p_1.0"
table_name = "estat_tps00001"
keep_amount = 10

true_df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite3.connect(true_db))
last_digit_column = (
    true_df[[col for col in true_df.columns if col.isdigit()]]
    .ffill(axis="columns")
    .iloc[:, -1]
)
true_df = true_df[["geographical"]].copy()
true_df["population_latest"] = last_digit_column

count_value = true_df["geographical"].nunique()
sum_value = true_df["population_latest"].sum()
avg_value = true_df["population_latest"].mean()

ground_truth_values = (count_value, sum_value, avg_value)

print(
    f"Ground Truth                              : {count_value:>10.2f} COUNT, {sum_value:>10.2f} SUM, {avg_value:>10.2f} AVG"
)

db_files = [f for f in os.listdir(folder) if f.endswith(".db")][:keep_amount]

source_samples = []
for db_file in db_files:
    connection = sqlite3.connect(os.path.join(folder, db_file))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)

    last_digit_column = (
        df[[col for col in df.columns if col.isdigit()]]
        .ffill(axis="columns")
        .iloc[:, -1]
    )
    df = df[["geographical"]].copy()
    df["population_latest"] = last_digit_column

    data_items = [
        DataItem(name=row["geographical"], value=row["population_latest"], source_id=db_file)
        for _, row in df.iterrows()
    ]
    source_samples.append(data_items)

estimator = Estimator(source_samples)

print(
    f"Sample                                    : {estimator.sample_count():>10.2f} COUNT, {estimator.sample_sum():>10.2f} SUM, {estimator.sample_avg():>10.2f} AVG"
)

# bounds for count, sum, avg
lower_count_bound_good_turing, upper_count_bound_good_turing = estimator.estimated_count_bounds(BoundType.GOOD_TURING)
lower_count_bound_chao84, upper_count_bound_chao84 = estimator.estimated_count_bounds(BoundType.CHAO84)
lower_sum_bound_good_turing, upper_sum_bound_good_turing = estimator.estimated_sum_bounds(BoundType.GOOD_TURING)
lower_sum_bound_chao84, upper_sum_bound_chao84 = estimator.estimated_sum_bounds(BoundType.CHAO84)
lower_avg_bound, upper_avg_bound = estimator.estimated_avg_bounds()

print(
    f"Count bounds (Good-Turing)                : [{lower_count_bound_good_turing:>15.2f}, {upper_count_bound_good_turing:>15.2f}] COUNT"
)
print(
    f"Count bounds (Chao84)                     : [{lower_count_bound_chao84:>15.2f}, {upper_count_bound_chao84:>15.2f}] COUNT"
)
print(
    f"Sum bounds (Good-Turing)                  : [{lower_sum_bound_good_turing:>15.2f}, {upper_sum_bound_good_turing:>15.2f}] SUM"
)
print(
    f"Sum bounds (Chao84)                       : [{lower_sum_bound_chao84:>15.2f}, {upper_sum_bound_chao84:>15.2f}] SUM"
)
print(
    f"Avg bounds                                : [{lower_avg_bound:>15.2f}, {upper_avg_bound:>15.2f}] AVG"
)

estimates = []

for use_bucket, use_monte_carlo, suffix in [
    (False, False, ""),
    (True,  False, " (bucket)"),
    (False, True,  " (mc)"),
]:
    for count_estimator in CountEstimator:
        for value_estimator in ValueEstimator:
            if count_estimator in [CountEstimator.JACKKNIFE1, CountEstimator.JACKKNIFE2]:
                continue

            if value_estimator in [ValueEstimator.GOOD_TURING, ValueEstimator.FREQUENCY_F12]:
                continue

            estimates.append((
                count_estimator.name.capitalize() + suffix,
                value_estimator.name.capitalize(),
                estimator.estimated_count(count_estimator, use_bucket=use_bucket, use_monte_carlo=use_monte_carlo),
                estimator.estimated_sum(value_estimator, count_estimator, use_bucket=use_bucket, use_monte_carlo=use_monte_carlo),
                estimator.estimated_avg(use_bucket=use_bucket, bucket_count_estimator=count_estimator, bucket_value_estimator=value_estimator, bucket_use_monte_carlo=use_monte_carlo),
            ))

estimates.sort(
    key=lambda x: (
        abs(x[3] - ground_truth_values[1]),
        abs(x[2] - ground_truth_values[0]),
    )
)

for estimate in estimates:
    print(
        f"{estimate[0] + ',':<20s} {estimate[1]:<13s} : {estimate[2]:>10.2f} COUNT, {estimate[3]:>10.2f} SUM, {estimate[4]:>10.2f} AVG"
    )
