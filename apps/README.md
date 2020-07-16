# Applications

Contains Applications buit around the Transliteration engine.

1. HTTPS API exposing for interation with web applications

2. NN based models and related codes for transliteration engine


Dependency Libraries:
* flask
* flask_cors
* torch
* numpy
* gevent

### Usage:

1. Export the python path variable <br>
`$ export PYTHONPATH=/path-to-apps/models/:$PYTHONPATH` <br>
2. Make required modification in SSL paths in `api_expose.py`. By default set to local host and both http & https are enabled <br>
Run the API expose code <br>
`$ sudo env PATH=$PATH python3 api_expose.py` <br>
3. In browser (or) curl, use link as https://{IP-address}:8000/tl/{lang-id}/{word in eng script} <br>
example: <br>
https://localhost:8000/tl/gom/ama  <br>

Language Codes:
* Goan Konkani - gom
* Maithili - mai

