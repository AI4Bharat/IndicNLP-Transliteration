# AI4Bharat Transliteration Application

A deep transliteration engine for major languages of the Indian sub-continent.

This package provides support for:
1. Python Library for transliteration from Roman to Native text (using NN-based models)
2. HTTP API exposing for interation with web applications

## Languages Supported

|ISO 639 code|Language|
|---|---------------------|
|bn |Bengali - বাংলা        |
|gom|Gujarati - ગુજરાતી      |
|gu |Hindi - हिंदी           |
|hi |Kannada - ಕನ್ನಡ        |
|kn |Konkani Goan - कोंकणी  |
|mai|Maithili - मैथिली       |
|ml |Malayalam - മലയാളം    |
|mr |Marathi - मराठी        |
|pa |Panjabi - ਪੰਜਾਬੀ       |
|sd |Sindhi - سنڌي‎        |
|si |Sinhala - සිංහල       |
|te |Telugu - తెలుగు        |
|ta |Tamil - தமிழ்         |
|ur |Urdu - اُردُو          |

## Usage

### Python Library

Import the transliteration engine by:
```
from ai4bharat.transliteration import XlitEngine
```

**Example 1** : Using word Transliteration

```py

e = XlitEngine("hi")
out = e.translit_word("computer", topk=5, beam_width=10)
print(out)
# output:{'hi': ['कम्प्यूटर', 'कंप्यूटर', 'कम्पूटर', 'कम्पुटर', 'कम्प्युटर']}

```

Note:
- `beam_width` increases beam search size, resulting in improved accuracy but increases time/compute.
- `topk` returns only specified number of top results.

**Example 2** : Using Sentence Transliteration

```py

e = XlitEngine("ta")
out = e.translit_sentence("vanakkam ulagam !", beam_width=10)
print(out)
# output: {'ta': 'வணக்கம் உலகம் !'}

```

Note:
- Only single top most prediction is returned for each word in sentence.

**Example 3** : Using Multiple language Transliteration

```py

e = XlitEngine(["ta", "ml"])
# leave empty or use "all" to load all available languages
# e = XlitEngine("all)

out = e.translit_word("amma", topk=5, beam_width=10)
print(out)
# {'ta': ['அம்மா', 'அம்ம', 'அம்மை', 'ஆம்மா', 'ம்மா'], 'ml': ['അമ്മ', 'എമ്മ', 'അമ', 'എഎമ്മ', 'അഎമ്മ']}

out = e.translit_sentence("hello world", beam_width=10)
print(out)
# output: {'ta': 'ஹலோ வார்ல்ட்', 'ml': 'ഹലോ വേൾഡ്'}

## Specify language name to get only specific language result
out = e.translit_word("amma", lang_code = "ml", topk=5, beam_width=10)
print(out)
# output: ['അമ്മ', 'എമ്മ', 'അമ', 'എഎമ്മ', 'അഎമ്മ']

```

**Example 4** : Transliteration for all available languages
```py

e = XlitEngine()
out = e.translit_sentence("Hello World", beam_width=10)
print(out)
# {'bn': 'হেল ওয়ার্ল্ড', 'gu': 'હેલો વર્લ્ડ', 'hi': 'हेलो वर्ल्ड', 'kn': 'ಹೆಲ್ಲೊ ವರ್ಲ್ಡ್', 'gom': 'हॅलो वर्ल्ड', 'mai': 'हेल्लो वर्ल्ड', 'ml': 'ഹലോ വേൾഡ്', 'mr': 'हेलो वर्ल्ड', 'pa': 'ਹੇਲੋ ਵਰਲਡ', 'sd': 'هيلو ورلد', 'si': 'හිලෝ වර්ල්ඩ්', 'ta': 'ஹலோ வார்ல்ட்', 'te': 'హల్లో వరల్డ్', 'ur': 'ہیلو وارڈ'}

```


### Web API Server

Running a flask server in 3 lines:
```py
from ai4bharat.transliteration import xlit_server
app, engine = xlit_server.get_app()
app.run(debug=True, host='0.0.0.0', port=8000)
```

You can also check the extended [sample script](https://github.com/AI4Bharat/IndianNLP-Transliteration/blob/master/apps/api_expose.py) as shown below:

1. Make required modification in SSL paths in `api_expose.py`. By default set to local host and both http & https are enabled.

2. Run the API expose code:
`$ sudo env PATH=$PATH python3 api_expose.py`
(Export `GOOGLE_APPLICATION_CREDENTIALS` if needed, by default functions realted to Google cloud is disabled.)

3. In browser (or) curl, use link as http://{IP-address}:{port}/tl/{lang-id}/{word in eng script}
If debug mode enabled port will be 8000, else port will be 80.

Example:
http://localhost:80/tl/ta/amma
http://localhost:80/languages

---

## Release Notes

This package contains applications built around the Transliteration engine. The contents of this package can also be downloaded from [latest GitHub release](https://github.com/AI4Bharat/IndianNLP-Transliteration/releases/latest) is sufficient for inference usage.

All the NN models (along with metadata) of Xlit - Transliteration are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].



[cc-by-sa]: http://creativecommons.org/licenses/by/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
