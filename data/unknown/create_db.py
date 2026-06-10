import pandas as pd

excluded_country_mapping = """
EU27_2020;European Union - 27 countries (from 2020)
EA21;Euro area – 21 countries (from 2026)
EA20;Euro area – 20 countries (2023-2025)
EA19;Euro area - 19 countries  (2015-2022)
EFTA;European Free Trade Association
AM;Armenia
AZ;Azerbaijan
GE;Georgia
RU;Russia
TR;Turkey
""".strip().split('\n')
excluded_country_mapping_df = pd.DataFrame([line.split(';') for line in excluded_country_mapping], columns=['code', 'country'])

country_mapping = """
AD;Andorra
AL;Albania
BA;Bosnia and Herzegovina
BY;Belarus
MC;Monaco
MD;Moldova
ME;Montenegro
MK;North Macedonia
RS;Serbia
SM;San Marino
UA;Ukraine
UK;United Kingdom
XK;Kosovo
BE;Belgium
BG;Bulgaria
CZ;Czechia
DK;Denmark
DE;Germany
EE;Estonia
IE;Ireland
EL;Greece
ES;Spain
FR;France
HR;Croatia
IT;Italy
CY;Cyprus
LV;Latvia
LT;Lithuania
LU;Luxembourg
HU;Hungary
MT;Malta
NL;Netherlands
AT;Austria
PL;Poland
PT;Portugal
RO;Romania
SI;Slovenia
SK;Slovakia
FI;Finland
SE;Sweden
IS;Iceland
LI;Liechtenstein
NO;Norway
CH;Switzerland
""".strip().split('\n')
country_mapping_df = pd.DataFrame([line.split(';') for line in country_mapping], columns=['code', 'country'])

frequency_mapping = """
A;Annual
S;Semi-annual
Q;Quarterly
M;Monthly
D;Daily
""".strip().split('\n')
frequency_mapping_df = pd.DataFrame([line.split(';') for line in frequency_mapping], columns=['code', 'frequency'])

flags_mapping = """
b;break in time series
be;break in time series, estimated
bp;break in time series, provisional
e;estimated
ep;estimated, provisional
p;provisional
""".strip().split('\n')
flags_mapping_df = pd.DataFrame([line.split(';') for line in flags_mapping], columns=['code', 'flag'])


df = pd.read_csv('./estat_tps00001.tsv', sep='\t')

# rename all column names by removing whitespace
df.columns = df.columns.str.strip()

# remove leading and trailing whitespace from all string values in the dataframe
df = df.apply(lambda x: x.str.strip())

# replace all ':' values with None
df.replace(':', None, inplace=True)

# add extra flag column for all year (column name: \d{4})
for col in df.columns:
    if col.isdigit():
        value = df[col]
        # split 1820631 e
        value = value.str.split(' ')
        # write the first part to the original column and the second part to a new column with suffix '_flags'
        # if no flags exists write empty string
        df[col] = value.str[0]
        df[col + '_flags'] = value.str[1] # .fillna('')

        # convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

# split the first column into multiple freq,indic_de,geo\TIME_PERIOD, remove original column
df[['frequency', 'demographic_indicator', 'geographical']] = df.iloc[:, 0].str.split(',', expand=True)
df.drop(df.columns[0], axis=1, inplace=True)

# reorder columns to have frequency, demographic_indicator, geographical at the start, then year + year_flags columns
cols = df.columns.tolist()
cols = ['frequency', 'demographic_indicator', 'geographical'] + [col for col in sorted(cols) if col not in ['frequency', 'demographic_indicator', 'geographical']]
df = df[cols]

# drop all values in the excluded_country_mapping_df
df = df[~df['geographical'].isin(excluded_country_mapping_df['code'])]

# sort by geographical
df.sort_values(by='geographical', inplace=True)

# save as sqlite file
df.to_sql('estat_tps00001', 'sqlite:///estat_tps00001.db', if_exists='replace', index=False)
print(df.head())
country_mapping_df.to_sql('country_mapping', 'sqlite:///estat_tps00001.db', if_exists='replace', index=False)
print(country_mapping_df.head())
frequency_mapping_df.to_sql('frequency_mapping', 'sqlite:///estat_tps00001.db', if_exists='replace', index=False)
print(frequency_mapping_df.head())
flags_mapping_df.to_sql('flags_mapping', 'sqlite:///estat_tps00001.db', if_exists='replace', index=False)
print(flags_mapping_df.head())