from collections import Counter, defaultdict
from functools import lru_cache

BABELCODE2LANG = {
    "101": "Cantonese",
    "102": "Assamese",
    "103": "Bengali",
    "104": "Pashto",
    "105": "Turkish",
    "106": "Tagalog",
    "107": "Vietnamese",
    "201": "Haitian",
    "202": "Swahili",
    "203": "Lao",
    "204": "Tamil",
    "205": "Kurmanji",
    "206": "Zulu",
    "207": "Tok-Pisin",
    "301": "Cebuano",
    "302": "Kazakh",
    "303": "Telugu",
    "304": "Lithuanian",
    "305": "Guarani",
    "306": "Igbo",
    "307": "Amharic",
    "401": "Mongolian",
    "402": "Javanese",
    "403": "Dholuo",
    "404": "Georgian",
}

IGNORED_SYMBOLS = [
    '\n', '<space>', 'space>', '<silence>', '<noise>', '>', '-', '<',
    '与', '业', '东', '中', '为', '事', '于', '亚', '人', '亿', '件', '伊', '会', '住', '作', '使', '供',
    '值', '停', '元', '克', '公', '其', '军', '决', '况', '准', '出', '击', '利', '区', '半', '华', '南',
    '卫', '厅', '发', '取', '口', '司', '合', '同', '向', '国', '在', '城', '备', '好', '定', '实', '宣',
    '家', '对', '导', '射', '将', '尤', '就', '居', '展', '岚', '工', '已', '市', '布', '年', '并', '开',
    '弹', '强', '得', '态', '急', '惕', '我', '技', '投', '拉', '拓', '提', '政', '故', '敢', '数', '新',
    '方', '日', '时', '是', '未', '本', '李', '来', '栗', '歉', '止', '此', '民', '油', '治', '济', '清',
    '火', '爱', '状', '现', '用', '百', '的', '益', '石', '社', '种', '科', '穷', '突', '紧', '经', '美',
    '者', '要', '见', '警', '认', '论', '贫', '资', '迅', '这', '进', '远', '迫', '速', '道', '里', '防',
    '青', '非', '革', '领', '驱', '鼎'
]

# manner class
manner_of_articulation_to_phones = {
    'Plosive': ['p', 'b', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'g', 'ɡ', 'q', 'ʔ', 'č', 'ď'],
    'Nasal': ['m', 'ɱ', 'n', 'ɲ', 'ŋ'],
    'Trill': ['r', 'R'],
    'Flap': ['ɾ', 'ɽ'],
    'Fricative': ['β', 'f', 'v', 'ð', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'x', 'ɣ', 'ʁ', 'ħ', 'h', 'ɦ', 'ɕ'],
    'LatFric': ['ɬ', 'ɮ'],
    'Approximant': ['ɹ', 'ɻ', 'j', 'ɯ', 'ɥ'],
    'LatApprox': ['l', 'ʎ'],
    'Vowels': ['i', 'ɨ', 'y', 'u', 'w', 'ɪ', 'e', 'ę', 'ë', 'ø', 'ə', 'ɤ', 'o', 'ʊ', 'ɛ', 'œ', 'ɜ', 'ɔ', 'æ', 'ɐ', 'a',
               'ɑ', 'á'],
    'Clicks': ['ǀ', 'ǃ', 'ǁ', ],
    'VoicedImplosives': ['ɗ', 'ɓ', 'ɠ', 'ɓ', 'ɗ'],
}
phone_to_manner = {
    phone: manner
    for manner, phones in manner_of_articulation_to_phones.items()
    for phone in phones
}

# articulation point
place_of_articulation_to_phones = {
    'Bilabial': ['p', 'b', 'm', 'β', 'ɓ'],
    'Labiodental': ['ɱ', 'f', 'v'],
    'Dental': ['ð', 'ǀ'],
    'Alveolar': ['t', 'd', 'n', 'r', 'ɾ', 's', 'z', 'ɬ', 'ɮ', 'ɹ', 'l', 'ď'],
    'Postalveolar': ['ʃ', 'ʒ', 'ǃ'],
    'Retroflex': ['ʈ', 'ɖ', 'ɽ', 'ʂ', 'ɻ'],
    'Palatal': ['c', 'ɟ', 'ɲ', 'j', 'ʎ', 'č'],
    'Velar': ['k', 'g', 'ɡ', 'ŋ', 'x', 'ɣ', 'ɯ', 'ɠ'],
    'Uvular': ['q', 'ʁ', 'R'],
    'Pharyngeal': ['ħ'],
    'Glottal': ['ʔ', 'h', 'ɦ'],
    'Vow-Cl': ['i', 'ɨ', 'y', 'u', 'w', 'ɪ'],
    'Vow-Clm': ['e', 'ë', 'ø', 'ə', 'ɤ', 'o', 'ʊ', 'ę', 'ɚ'],
    'Vow-Openm': ['ɛ', 'œ', 'ɜ', 'ɔ', 'æ', 'ɐ'],
    'Vow-Open': ['a', 'ɑ', 'á'],
    'Alveolo-Palatal': ['ɕ'],
    'Dental-Alveolar': ['ɗ'],
    'Voiced-Labial-Palatal': ['ɥ'],
}
phone_to_place = {
    phone: place
    for place, phones in place_of_articulation_to_phones.items()
    for phone in phones
}


def determine_place(phone: str, phone_to_place=phone_to_place):
    # Empty token - insertion or deletion
    if phone == '*':
        return phone
    # Remove stress symbol
    if phone.startswith('ˈ'):
        phone = phone[1:]
    for phone_tok in phone_to_place:
        if phone.startswith(phone_tok):
            return phone_to_place[phone_tok]
    return '?'


def determine_manner(phone: str, phone_to_manner=phone_to_manner):
    # Empty token - insertion or deletion
    if phone == '*':
        return phone
    # Remove stress symbol
    if phone.startswith('ˈ'):
        phone = phone[1:]
    for phone_tok in phone_to_manner:
        if phone.startswith(phone_tok):
            return phone_to_manner[phone_tok]
    return '?'


PHONE_BASES = frozenset(phone_to_place).union(phone_to_manner)


def determine_base(phone: str, bases_=PHONE_BASES):
    # Empty token - insertion or deletion
    if phone == '*':
        return phone
    # Remove stress symbol
    if phone.startswith('ˈ'):
        phone = phone[1:]
    for base in bases_:
        if phone.startswith(base):
            return base
    return '?'


def get_lang_to_phones(ground_truth_paths):
    lang_to_phones = defaultdict(Counter)
    for p in ground_truth_paths:
        lang = BABELCODE2LANG.get(p.stem, p.stem)
        for line in p.open():
            phones = line.strip().split()[1:]
            for p in phones:
                if not any(p.startswith(sym) for sym in '<<'):
                    lang_to_phones[lang][p] += 1
    return lang_to_phones


def get_special_symbols(ground_truth_paths):
    syms = set()
    for p in ground_truth_paths:
        for line in p.open():
            phones = line.strip().split()[1:]
            for p in phones:
                # The two characters below are actually different
                if any(p.startswith(sym) for sym in '<<'):
                    syms.add(p)
    return syms


def get_lang_to_places(ground_truth_paths, unique=False):
    lang_to_phones = get_lang_to_phones(ground_truth_paths)
    lang_to_places = defaultdict(Counter)
    for lang, phone_counts in lang_to_phones.items():
        for phone, count in phone_counts.items():
            place = determine_place(phone)
            lang_to_places[lang][place or '?'] += 1 if unique else count
    return lang_to_places


def get_lang_to_manners(ground_truth_paths, unique=False):
    lang_to_phones = get_lang_to_phones(ground_truth_paths)
    lang_to_manners = defaultdict(Counter)
    for lang, phone_counts in lang_to_phones.items():
        for phone, count in phone_counts.items():
            place = determine_manner(phone)
            lang_to_manners[lang][place or '?'] += 1 if unique else count
    return lang_to_manners
