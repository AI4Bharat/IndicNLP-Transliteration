# Applications

Contains Applications built around the Transliteration engine. The contents of this folder (to be downloaded from [latest release](https://github.com/AI4Bharat/IndianNLP-Transliteration/releases/latest)) is sufficient for usage.

1. Python Library for running transliteration from Roman to Native text
2. HTTPS API exposing for interation with web applications
3. NN based models and related metadata

# Usage

## Python Library

### From Github Releases
1. Download the release object xlit_apps_vx.x.x.zip <br>
`$ wget https://github.com/AI4Bharat/IndianNLP-Transliteration/releases/latest/download/xlit_apps_v0.4.1.zip`

2. Set pythonpath environment variable with absolute path to repository<br>
`export PYTHONPATH=/realpath/to/xlit_apps:$PYTHONPATH`

3. Import and start using !

**Example 1** : Using word Transliteration

Note: <br>
`beam_width` increases beam search size, resulting in improved accuracy but increases time/compute. <br>
`topk` returns only specified number of top results.


```
from xlit_src import XlitEngine

e = XlitEngine("hi")
out = e.translit_word("aam", topk=5, beam_width=10)
print(out)
# output:{'hi': ['कम्प्यूटर', 'कंप्यूटर', 'कम्पूटर', 'कम्पुटर', 'कम्प्युटर']}

```
**Example 2** : Using Sentence Transliteration

Note: <br>
`beam_width` increases beam search size, resulting in improved accuracy but increases time/compute. <br>
Only single top most prediction is returned for sentences.


```
from xlit_src import XlitEngine

e = XlitEngine("ta")
out = e.translit_sentence("vanakkam ulagam !", beam_width=10)
print(out)
# output: {'ta': 'வணக்கம் உலகம் !'}

```

**Example 3** : Using Multiple language Transliteration

```
from xlit_src import XlitEngine

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

---

## Web API

1. Make required modification in SSL paths in `api_expose.py`. By default set to local host and both http & https are enabled <br>

2. Run the API expose code <br>
`$ sudo env PATH=$PATH python3 api_expose.py` <br>
Export `GOOGLE_APPLICATION_CREDENTIALS` if needed, by default functions realted to Google cloud is disabled.

3. In browser (or) curl, use link as https://{IP-address}:{port}/tl/{lang-id}/{word in eng script} <br>
If debug mode enabled port will be 8000, else port will be 80 <br>
example: <br>
https://localhost:80/tl/ta/amma  <br>
https://localhost:80/languages  <br>

---

## Language Codes:
```
* bn  - Bengali
* gom - Konkani Goan
* gu  - Gujarati
* hi  - Hindi
* kn  - Kannada
* mai - Maithili
* ml  - Malayalam
* mr  - Marathi
* pa  - Punjabi Eastern
* sd  - Sindhi
* si  - Sinhala
* ta  - Tamil
* te  - Telugu
* ur  - Urdu
```

## Dependencies:
* torch
* numpy

Web api, also depends
* flask
* flask_cors
* gevent