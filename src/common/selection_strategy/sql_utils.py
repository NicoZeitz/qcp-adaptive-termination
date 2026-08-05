def quote_table(name: str) -> str:
    return ".".join(f'"{p}"' for p in name.split(".", 1))


def quote_col(name: str) -> str:
    return f'"{name}"'
