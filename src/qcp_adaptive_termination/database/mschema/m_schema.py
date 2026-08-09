import json
from typing import Any

from qcp_adaptive_termination.database.mschema.utils import examples_to_str


class MSchema:
    def __init__(self, db_id: str = "Anonymous", schema: str | None = None):
        self.db_id = db_id
        self.schema = schema
        self.tables = {}
        self.foreign_keys = []
        self.database_comment: str = ""

    def add_table(self, name, fields=None, comment=None):
        self.tables[name] = {
            "fields": (fields or {}).copy(),
            "examples": [],
            "comment": comment,
        }

    def add_field(
        self,
        table_name: str,
        field_name: str,
        field_type: str = "",
        primary_key: bool = False,
        nullable: bool = True,
        default: Any = None,
        autoincrement: bool = False,
        comment: str = "",
        examples: list | None = None,
        **kwargs,
    ):
        if examples is None:
            examples = []
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default if default is None else f"{default}",
            "autoincrement": autoincrement,
            "comment": comment,
            "examples": (examples or []).copy(),
            **kwargs,
        }

    def add_foreign_key(self, table_name, field_name, ref_schema, ref_table_name, ref_field_name):
        self.foreign_keys.append([table_name, field_name, ref_schema, ref_table_name, ref_field_name])

    def get_field_type(self, field_type, simple_mode=True) -> str:
        if not simple_mode:
            return field_type
        else:
            return field_type.split("(")[0]

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def has_column(self, table_name: str, field_name: str) -> bool:
        if self.has_table(table_name):
            return field_name in self.tables[table_name]["fields"]
        else:
            return False

    def get_field_info(self, table_name: str, field_name: str) -> dict:
        try:
            return self.tables[table_name]["fields"][field_name]
        except Exception:
            return {}

    def single_field_mschema(
        self,
        table_name: str,
        field_name: str,
        example_num: int = 3,
        show_type_detail: bool = False,
    ) -> str:
        field_info = self.get_field_info(table_name, field_name)
        if not field_info:
            return ""
        raw_type = self.get_field_type(field_info["type"], not show_type_detail)
        line = f"({field_name}:{raw_type.upper()}"
        if field_info.get("comment"):
            line += f", {field_info['comment'].strip()}"
        # 打上主键标识
        if field_info.get("primary_key"):
            line += ", Primary Key"
        # 如果有示例，添加上
        examples = [s for s in field_info.get("examples", []) if s is not None]
        examples = examples_to_str(examples)
        if examples and example_num > 0:
            if len(examples) > example_num:
                examples = examples[:example_num]
            if raw_type in ["DATE", "TIME", "DATETIME", "TIMESTAMP"]:
                examples = [examples[0]]
            elif max((len(str(s)) for s in examples), default=0) > 50:
                examples = []
            elif max((len(str(s)) for s in examples), default=0) > 20:
                examples = [examples[0]]
            if examples:
                line += f", Examples: [{', '.join(str(e) for e in examples)}]"
        line += ")"
        return line

    def field_mschema(
        self,
        table_name: str,
        example_num: int = 3,
        show_type_detail: bool = False,
    ) -> dict[str, str]:
        return {
            field_name: self.single_field_mschema(table_name, field_name, example_num, show_type_detail)
            for field_name in self.tables.get(table_name, {}).get("fields", {})
        }

    def single_table_mschema(
        self,
        table_name: str,
        selected_columns: list | None = None,
        example_num=3,
        show_type_detail=False,
    ) -> str:
        table_info = self.tables.get(table_name, {})
        output = []
        table_comment = table_info.get("comment", "")
        if table_comment is not None and table_comment != "None" and len(table_comment) > 0:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}, {table_comment}")
            else:
                output.append(f"# Table: {table_name}, {table_comment}")
        else:
            if self.schema is not None and len(self.schema) > 0:
                output.append(f"# Table: {self.schema}.{table_name}")
            else:
                output.append(f"# Table: {table_name}")

        field_lines = []
        # 处理表中的每一个字段
        for field_name in table_info["fields"]:
            if selected_columns is not None and field_name.lower() not in selected_columns:
                continue
            line = self.single_field_mschema(table_name, field_name, example_num, show_type_detail)
            if line:
                field_lines.append(line)
        output.append("[")
        output.append(",\n".join(field_lines))
        output.append("]")

        return "\n".join(output)

    def table_mschema(
        self,
        example_num: int = 3,
        show_type_detail: bool = False,
    ) -> dict[str, str]:
        return {
            table_name: self.single_table_mschema(table_name, None, example_num, show_type_detail)
            for table_name in self.tables
        }

    def to_mschema(
        self,
        selected_tables: list | None = None,
        selected_columns: list | None = None,
        example_num=3,
        show_type_detail=False,
        include_db_id: bool = True,
    ) -> str:
        """
        convert to a MSchema string.
        selected_tables: 默认为None，表示选择所有的表
        selected_columns: 默认为None，表示所有列全选，格式['table_name.column_name']
        """
        if selected_columns is not None:
            selected_columns_lower = [s.lower() for s in selected_columns]
            selected_tables_lower = [s.split(".")[0].lower() for s in selected_columns]
        else:
            selected_columns_lower = None
            selected_tables_lower = [s.lower() for s in selected_tables] if selected_tables is not None else None

        output = ([f"【DB_ID】 {self.db_id}"] if include_db_id else []) + ["【Schema】"]
        for table_name in self.tables:
            if selected_tables_lower is not None and table_name.lower() not in selected_tables_lower:
                continue
            cur_cols = (
                [
                    c.lower()
                    for c in self.tables[table_name]["fields"]
                    if f"{table_name}.{c}".lower() in selected_columns_lower
                ]
                if selected_columns_lower is not None
                else None
            )
            output.append(self.single_table_mschema(table_name, cur_cols, example_num, show_type_detail))

        # 添加外键信息，选择table_type为view时不展示外键
        if self.foreign_keys:
            normalized_tables = selected_tables_lower
            fk_lines = []
            for fk in self.foreign_keys:
                ref_schema = fk[2]
                table1, column1, _, table2, column2 = fk
                if (
                    normalized_tables is None
                    or (table1.lower() in normalized_tables and table2.lower() in normalized_tables)
                ) and ref_schema == self.schema:
                    fk_lines.append(f"{fk[0]}.{fk[1]}={fk[3]}.{fk[4]}")
            if fk_lines:
                output.append("【Foreign keys】")
                output.extend(fk_lines)

        return "\n".join(output)

    def dump(self):
        schema_dict = {
            "db_id": self.db_id,
            "schema": self.schema,
            "tables": self.tables,
            "foreign_keys": self.foreign_keys,
        }
        return schema_dict

    def save(self, file_path: str):
        schema_dict = self.dump()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, ensure_ascii=False, indent=2)

    def load(self, file_path: str):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        self.db_id = data.get("db_id", "Anonymous")
        self.schema = data.get("schema", None)
        self.tables = data.get("tables", {})
        self.foreign_keys = data.get("foreign_keys", [])
