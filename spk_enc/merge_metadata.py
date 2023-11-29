import json
import dataclasses
from dataclasses import dataclass 
from typing import List

LANGUAGE_CODE: {
    "arb": 0,
    "ben": 1,
    "cat": 2,
    "ces": 3,
    "cmn": 4,
    "cym": 5,
    "dan": 6,
    "deu": 7,
    "eng": 8,
    "est": 9,
    "fin": 10,
    "fra": 11,
    "hin": 12,
    "ind": 13,
    "ita": 14,
    "jpn": 15,
    "kor": 16,
    "mlt": 17,
    "nld": 18,
    "pes": 19,
    "pol": 20,
    "por": 21,
    "ron": 22,
    "rus": 23,
    "slk": 24,
    "spa": 25,
    "swe": 26,
    "swh": 27,
    "tel": 28,
    "tgl": 29,
    "tha": 30,
    "tur": 31,
    "ukr": 32,
    "urd": 33,
    "uzn": 34,
    "vie": 35
}
        

LANGUAGES_WHISPER2VOCODER = {
    "en": "eng",
    "zh": "cmn",
    "de": "deu",
    "es": "spa",
    "ru": "rus",
    "ko": "kor",
    "fr": "fra",
    "ja": "jpn",
    "pt": "por",
    "tr": "tur",
    "pl": "pol",
    "ca": "cat",
    "nl": "nld",
    "ar": "arb",
    "sv": "swe",
    "it": "ita",
    "id": "ind",
    "hi": "hin",
    "fi": "fin",
    "vi": "vie",
    "he": "heb",
    "uk": "ukr",
    "el": "ell",
    "ms": "zsm",
    "cs": "ces",
    "ro": "ron",
    "da": "dan",
    "hu": "hun",
    "ta": "tam",
    "no": "nob",
    "th": "tha",
    "ur": "urd",
    "hr": "hrv",
    "bg": "bul",
    "lt": "lit",
    "la": "lat",
    "mi": "mri",
    "ml": "mal",
    "cy": "cym",
    "sk": "slk",
    "te": "tel",
    "fa": "pes",
    "lv": "lvs",
    "bn": "ben",
    "sr": "srp",
    "az": "azj",
    "sl": "slv",
    "kn": "kan",
    "et": "est",
    "mk": "mkd",
    "br": "bre",
    "eu": "eus",
    "is": "isl",
    "hy": "hye",
    "ne": "npi",
    "mn": "khk",
    "bs": "bos",
    "kk": "kaz",
    "sq": "sqi",
    "sw": "swh",
    "gl": "glg",
    "mr": "mar",
    "pa": "pan",
    "si": "sin",
    "km": "khm",
    "sn": "sna",
    "yo": "yor",
    "so": "som",
    "af": "afr",
    "oc": "oci",
    "ka": "kat",
    "be": "bel",
    "tg": "tgk",
    "sd": "snd",
    "gu": "guj",
    "am": "amh",
    "yi": "yid",
    "lo": "lao",
    "uz": "uzn",
    "fo": "fao",
    "ht": "hat",
    "ps": "pbt",
    "tk": "tuk",
    "nn": "nno",
    "mt": "mlt",
    "sa": "san",
    "lb": "ltz",
    "my": "mya",
    "bo": "bod",
    "tl": "tgl",
    "mg": "mlg",
    "as": "asm",
    "tt": "tat",
    "haw": "haw",
    "ln": "lin",
    "ha": "hau",
    "ba": "bak",
    "jw": "jav",
    "su": "sun",
    "yue": "yue",
}

@dataclass
class Data:
    audio: str
    unit: List[int]
    speaker: str
    nationality: str
    embeded: str
    language: str
    
def whisper2vocoder(lang_whisper: str):
    lang_vocoder = LANGUAGES_WHISPER2VOCODER.get(lang_whisper, None)
    if lang_vocoder is None or lang_vocoder not in LANGUAGE_CODE.keys():
        return None
    return lang_vocoder
    
    
def merge(manifest_path, audio_embed_path, audio2lang_path):
    with open(manifest_path, "r") as f:
        manifest = f.readlines()
    with open(audio_embed_path, "r") as f:
        audio_embed = f.readlines()    
    with open(audio2lang_path, "r") as f:
        audio2lang = f.readlines()
        
    assert len(manifest) != len(audio_embed), f"Length of {manifest_path} ({len(manifest)}) != length of {audio_embed_path} ({len(audio_embed)})"
    assert len(manifest) != len(audio2lang), f"Length of {manifest_path} ({len(manifest)}) != length of {audio2lang_path} ({len(audio2lang)})"
        
    countNone = 0
    for i in range(len(manifest)):
        print(i)
        
        manifest_line = manifest[i]
        manifest_json = json.loads(manifest_line)
        
        audio_embed_line = audio_embed[i]
        audio_embed_json = json.loads(audio_embed_line)
        
        audio2lang_line = audio2lang[i]
        audio2lang_json = json.loads(audio2lang_line)
        
        assert manifest_json["audio"] != audio_embed["audio"], f"Audio in manifest != audio_embed at line {i+1}"
        assert manifest_json["audio"] != audio2lang["audio"], f"Audio in manifest != audio2lang at line {i+1}"
        
        if whisper2vocoder(audio2lang_json["language"]) is None:
            countNone += 1
            continue
        dictionary = Data(manifest_json["audio"], manifest_json["unit"], manifest_json["speaker"], 
                          manifest_json["nation"], audio_embed_json["embed"], whisper2vocoder(audio2lang_json["language"]))
        with open("/content/drive/MyDrive/vox1_metadata_all.json", "a") as outfile:
            outfile.write(json.dumps(dataclasses.asdict(dictionary)) + "\n")
            
    print("Xong!")
        
        
        