import os
from sqlalchemy import create_engine, inspect
from mschema.schema_engine import SchemaEngine

db_path = "./data/imdb.db"
abs_path = os.path.abspath(db_path)
db_engine = create_engine(f'sqlite:///{abs_path}')

# inspector = inspect(db_engine)
# inspector.get_table_comment('titles')

# set comment for table 'main.titles'
schema_engine = SchemaEngine(engine=db_engine, db_name="imdb")

# Add a placeholder comment to every column so it appears in M-Schema output.
for table_info in schema_engine.mschema.tables.values():
	for field_info in table_info["fields"].values():
		field_info["comment"] = "TODO:"

mschema = schema_engine.mschema
mschema_str = mschema.to_mschema()
print(mschema_str)


dialect = db_engine.dialect.name
question = ''
evidence = ''
prompt = """You are now a {dialect} data analyst, and you are given a database schema as follows:

【Schema】
{db_schema}

【Question】
{question}

【Evidence】
{evidence}

Please read and understand the database schema carefully, and generate an executable SQL based on the user's question and evidence. The generated SQL is protected by ```sql and ```.
""".format(dialect=dialect, question=question, db_schema=mschema_str, evidence=evidence)

print(prompt)

# Replace the function call_llm() with your own function or method to interact with a LLM API.
# response = call_llm(prompt)