from app.utils.print_controls import *

def parse_range_or_like(sql_col: str, value: str) -> str:
    value = value.strip()
    if value == ",":
        return ""

    if "," in value:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) == 2:
            part1, part2 = parts
            is_num1 = part1.replace('.', '', 1).replace('-', '', 1).isdigit()
            is_num2 = part2.replace('.', '', 1).replace('-', '', 1).isdigit()

            try:
                if is_num1 and is_num2:
                    return f"TRY_CAST({sql_col} AS FLOAT) BETWEEN {float(part1)} AND {float(part2)}"
                elif is_num1 and not part2:
                    return f"TRY_CAST({sql_col} AS FLOAT) >= {float(part1)}"
                elif not part1 and is_num2:
                    return f"TRY_CAST({sql_col} AS FLOAT) <= {float(part2)}"
            except ValueError:
                print_error(f"Invalid numeric range: {value}")
                return ""

    return f"{sql_col} LIKE '%{value}%'"

def build_filter_clause(params: dict, plant: str, column_map: dict) -> str:
    filters = [f"PLANT='{plant.capitalize()}'"]
    i = 0

    while True:
        field_key = f"filter_field_{i}"
        operator_key = f"filter_operator_{i}"
        value_key = f"filter_value_{i}"

        if field_key not in params:
            break

        field = params.get(field_key)
        operator = params.get(operator_key)
        value = params.get(value_key)
        i += 1

        if not field or not operator:
            continue

        sql_col = column_map.get(field)
        if not sql_col:
            print_error(f"Unknown field: {field}")
            continue

        if operator == "contains":
            clause = parse_range_or_like(sql_col, value)
            if clause:
                filters.append(clause)
                print_info(f"Added filter: [cyan]{clause}[/cyan]")

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    print_success(f"Built WHERE clause: [green]{where_clause}[/green]")
    return where_clause

def get_month_case_statement() -> str:
    return """
    CASE [Month]
        WHEN 'January' THEN 1
        WHEN 'February' THEN 2
        WHEN 'March' THEN 3
        WHEN 'April' THEN 4
        WHEN 'May' THEN 5
        WHEN 'June' THEN 6
        WHEN 'July' THEN 7
        WHEN 'August' THEN 8
        WHEN 'September' THEN 9
        WHEN 'October' THEN 10
        WHEN 'November' THEN 11
        WHEN 'December' THEN 12
        ELSE 13
    END
    """
