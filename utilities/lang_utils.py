unicode_numbers = {
    'devanagari': [chr(alpha) for alpha in range(2406, 2416)]
}

unicode_map = {
    'devanagari': [chr(alpha) for alpha in range(2304, 2432)]
}

lang2script = {
    'gom': 'devanagari',
    'hi': 'devanagari'
}

lang2code = {
    'konkani': 'gom', #Goa-Konkani (Wiki-Standard)
    'hindi': 'hi'
}

def get_lang_chars(lang, allow_numbers=False):
    script = lang2script[lang2code[lang]]
    chars = unicode_map[script]
    if not allow_numbers:
        numbers = unicode_numbers[script]
        chars = [c for c in chars if c not in numbers]
    return chars