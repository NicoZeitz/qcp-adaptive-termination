import datetime
import decimal
import re


def is_email(string):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    match = re.match(pattern, string)
    if match:
        return True
    else:
        return False


def examples_to_str(examples: list) -> list[str]:
    """
    from examples to a list of str
    """
    values = examples
    for i in range(len(values)):
        if isinstance(values[i], datetime.date):
            values = [values[i]]
            break
        elif isinstance(values[i], datetime.datetime):
            values = [values[i]]
            break
        elif isinstance(values[i], decimal.Decimal):
            values[i] = str(float(values[i]))
        elif is_email(str(values[i])):
            values = []
            break
        elif "http://" in str(values[i]) or "https://" in str(values[i]):
            values = []
            break
        elif values[i] is not None and not isinstance(values[i], str):
            pass
        elif values[i] is not None and ".com" in values[i]:
            pass

    return [str(v) for v in values if v is not None and len(str(v)) > 0]
