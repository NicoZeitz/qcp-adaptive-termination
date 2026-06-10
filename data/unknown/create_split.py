import os
import random
import numpy as np
import sqlite3
import argparse
import pandas as pd
import shutil

pd.set_option('display.float_format', lambda x: '%.3f' % x)

seed = 42
data_sources: list[int] = [15] * 10
ρ = 0.0
db_name = 'estat_tps00001'
table_name = 'estat_tps00001'

# allow setting the values via command line or take the default values
parser = argparse.ArgumentParser(description='Create split databases from the main database.')
parser.add_argument('--seed', '-s', type=int, default=seed, help='Random seed for reproducibility')
parser.add_argument('--data_sources', type=int, nargs='+', default=data_sources, help='List of number of data items to sample for each data source')
parser.add_argument('--correlation', '-c', type=float, default=ρ, help='Correlation between publicity and value likelihood')
parser.add_argument('--db_name', type=str, default=db_name, help='Name of the main database (without .db extension)')
parser.add_argument('--table_name', type=str, default=table_name, help='Name of the main table in the database')
args = parser.parse_args()

seed = args.seed
data_sources = args.data_sources
ρ = args.correlation
db_name = args.db_name
table_name = args.table_name

random.seed(seed)
np.random.seed(seed)


def _weighted_sample_without_replacement_indices(weights: pd.Series, n: int, rng: np.random.Generator) -> list[int]:
    """Sample indices exactly proportional to weights without replacement.

    At each draw, probabilities are re-normalized over the remaining items.
    This preserves the original bias and avoids pandas' stricter feasibility check.
    """
    if n <= 0:
        return []

    w = pd.Series(weights, dtype=float).fillna(0.0).to_numpy()
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")

    n_items = len(w)
    if n > n_items:
        raise ValueError(f"Cannot sample n={n} items from only {n_items} rows.")

    positive_count = int(np.count_nonzero(w > 0))
    if n > positive_count:
        raise ValueError(
            f"Cannot sample n={n} without replacement from only {positive_count} rows with positive weight. "
            "Increase the number of non-zero weights, reduce n, or allow replacement."
        )

    remaining_idx = np.arange(n_items)
    remaining_w = w.copy()
    chosen: list[int] = []

    for _ in range(n):
        probs = remaining_w / remaining_w.sum()
        draw_pos = int(rng.choice(len(remaining_idx), p=probs))
        chosen.append(int(remaining_idx[draw_pos]))
        remaining_idx = np.delete(remaining_idx, draw_pos)
        remaining_w = np.delete(remaining_w, draw_pos)

    return chosen

# load estat_tps00001.tsv
# horizontally split the database into `l` data sources, each sampling n_j = |s_j| data items from the ground truth D without replacement (each data source has a item only once)
# every data item d_i has a publicity likelihood p_i of being sampled (TODO: will come to this) following a distribution X
# attribute values have a certain likelihood to appear in the ground truth (value likelihood) following a distribution Y
# possible correlated publicity-value correlation ρ >= 0
# value is only once in D, but multiple times in S depending on publicity

connection = sqlite3.connect(f'{db_name}.db')
df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)

folder = f'split_n_{len(data_sources)}_p_{ρ:.1f}'
shutil.rmtree(folder, ignore_errors=True)
os.makedirs(folder, exist_ok=True)

# remove the rows EU27_2020, EA21, EA20, EA19, EFTA from the main db (these are aggregates that we do not want to be sampled in the data sources)
df = df[~df['geographical'].isin(['EU27_2020', 'EA21', 'EA20', 'EA19', 'EFTA'])]

# calculate the publicity likelihood p_i for each data item d_i based on the value likelihood and the correlation ρ
# use the columns 2025, 2024, 2023, ..., 2014 (older years as backup for null values)
value_likelihoods = df[[col for col in df.columns if col.isdigit()]].mean(axis=1, skipna=True)
# force value_likelihoods into 0 to 1 range
value_likelihoods = (value_likelihoods - value_likelihoods.min()) / (value_likelihoods.max() - value_likelihoods.min())
# TODO: smooth value likelihoods to be (n * max(weights) / sum(weights))<=1 with softmax
# value_likelihoods = np.exp(value_likelihoods) / np.sum(np.exp(value_likelihoods)) * len(df)

# for each data source s_j, sample n_j data items from the ground truth D without replacement based on the publicity likelihoods
for j, n_j in enumerate(data_sources):
    j = j+1
    publicity_likelihoods = ρ * value_likelihoods + (1 - ρ) * np.random.rand(len(df))
    df_copy = df.copy()
    # sample = df_copy.sample(n=n_j, weights=publicity_likelihoods, random_state=random.randint(0, 10000))

    if n_j > len(df_copy):
        raise ValueError(f"Cannot sample n={n_j} rows from only {len(df_copy)} available rows.")

    rng = np.random.default_rng(random.randint(0, 10000))
    sampled_indices = _weighted_sample_without_replacement_indices(publicity_likelihoods, n_j, rng)
    sample = df_copy.iloc[sampled_indices].copy()

    # build the new sqlite_database for the data source s_j
    sample.to_sql(f'{table_name}', f'sqlite:///{folder}/{db_name}_source_{j:02d}.db', if_exists='replace', index=False)

    # copy over the other tables
    # country mapping table (only keep the countries that are in the sample)
    country_mapping_df = pd.read_sql_query("SELECT * FROM country_mapping", connection)
    sample_countries = sample['geographical'].unique()
    country_mapping_df = country_mapping_df[country_mapping_df['code'].isin(sample_countries)]
    country_mapping_df.to_sql('country_mapping', f'sqlite:///{folder}/{db_name}_source_{j:02d}.db', if_exists='replace', index=False)

    # flags mapping table (only keep the flags that are in the sample)
    flags_mapping_df = pd.read_sql_query("SELECT * FROM flags_mapping", connection)
    sample_flags = []
    for col in sample.columns:
        if col.endswith('_flags'):
            sample_flags.extend(sample[col].dropna().unique())
    flags_mapping_df = flags_mapping_df[flags_mapping_df['code'].isin(sample_flags)]
    flags_mapping_df.to_sql('flags_mapping', f'sqlite:///{folder}/{db_name}_source_{j:02d}.db', if_exists='replace', index=False)

    # frequency mapping table (full table)
    frequency_mapping_df = pd.read_sql_query("SELECT * FROM frequency_mapping", connection)
    frequency_mapping_df.to_sql('frequency_mapping', f'sqlite:///{folder}/{db_name}_source_{j:02d}.db', if_exists='replace', index=False)


main_df = df

# load all dbs in the folder
db_files = [f for f in os.listdir(folder) if f.endswith('.db')]
db_dfs = {}
for db_file in db_files:
    connection = sqlite3.connect(os.path.join(folder, db_file))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
    db_dfs[db_file] = df

# check out how many of the values in the main db are present in the other dbs (all merged) and with what multiplicity (use geographical column as key), also print empty values. print population average as well
main_geographical = main_df['geographical']
main_population_latest = main_df[[col for col in main_df.columns if col.isdigit()]].ffill(axis="columns").iloc[:, -1]

merged_db_df = pd.concat(db_dfs.values(), ignore_index=True)
merged_geographical = merged_db_df['geographical']
presence = main_geographical.isin(merged_geographical)
multiplicity = merged_geographical.value_counts().reindex(main_geographical).fillna(0).astype(int)
result_df = pd.DataFrame({'geographical': main_geographical, 'population_latest': main_population_latest})
result_df = pd.merge(multiplicity.reset_index(drop=False), result_df, on="geographical", how="right").fillna(0)
result_df.rename(columns={'count': 'multiplicity'}, inplace=True)
result_df.sort_values(by=['population_latest', 'multiplicity'], ascending=[False, False], inplace=True)

# also save to log file
# create a log.txt file to log parameters
with open(f'{folder}/log.txt', 'w') as f:
    f.write(f"Seed: {seed}\n"   )
    f.write(f"Data sources: {data_sources}\n")
    f.write(f"Correlation ρ: {ρ}\n")
    f.write(f"Database name: {db_name}\n")
    f.write(f"Table name: {table_name}\n")
    f.write(f"Presence in merged dbs: {presence.sum()} out of {len(main_geographical)}\n")
    f.write(str(result_df)+ "\n")
    # write individual df's
    for df_file in sorted(db_dfs):
        f.write(f"\n{df_file}\n")
        f.write(str(db_dfs[df_file]) + "\n")