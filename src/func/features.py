def categorize_area(x):
    range_start = (x // 50) * 50
    range_end = range_start + 49
    return f"{range_start}~{range_end}"


def categorize_date(x):
    if 1 <= x <= 10:
        return 10
    elif 11 <= x <= 20:
        return 20
    else:
        return 30


def categorize_price(x):
    scale = 10000
    range_start = (x // scale) * scale
    range_end = range_start + scale - 1
    return f"{range_start}~{range_end}"
