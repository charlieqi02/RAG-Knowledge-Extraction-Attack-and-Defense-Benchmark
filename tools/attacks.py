import re 



def extract_indexes(text):
    index_pattern_single = r"'index':\s*(\d+)"
    index_pattern_double = r'"index":\s*(\d+)'
    matches_single = re.findall(index_pattern_single, text)
    matches_double = re.findall(index_pattern_double, text)
    matches = matches_single + matches_double
    indexes = list(map(int, matches))
    return indexes



