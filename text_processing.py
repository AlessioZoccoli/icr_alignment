import re
from json import load


def estimate_word_width_simple(tsc_word: str, default_char_width: int = 20) -> int:

    with open("average_charsizes_extended.json", "r") as f:
        AVG_CHARSIZES = load(f)

    word = re.sub(r"\(.*?\)|\'.*?\'", "", tsc_word.lower()
                  .replace('(et)', '7')
                  .replace('(rum)', '2')) \
        .replace('-', '') \
        .replace(',', '') \
        .replace('v', 'u')

    est_w = sum([
        AVG_CHARSIZES[c][0] if c in AVG_CHARSIZES.keys() else default_char_width
        for c in list(word)
    ])
    return int(est_w)


def estimate_word_width(tsc_word: str, default_char_width: int = 20) -> int:
    with open("average_charsizes_extended.json", "r") as f:
        AVG_CHARSIZES = load(f)

    word = re.sub(r"\(.*?\)|\'.*?\'", "", tsc_word)\
        .replace('-', '') \
        .replace(',', '') \
        .replace('v', 'u')

    est_w = sum([
        AVG_CHARSIZES[c][0] if c in AVG_CHARSIZES.keys() else default_char_width
        for c in list(word)
    ])
    return int(est_w)
