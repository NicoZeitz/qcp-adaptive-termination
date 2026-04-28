from os import listdir
from os.path import isfile, join
import json
from pathlib import Path

path = "../data/test"

files = [Path(join(path, f)) for f in listdir(path) if isfile(join(path, f))]

file = files[0]
with open(file) as f:
    nl_query = file.stem
    print(nl_query)
    content = json.load(f)

    sql_query = list(content.keys())[0]
    result = content[sql_query]

    print(sql_query)
    print(result)
