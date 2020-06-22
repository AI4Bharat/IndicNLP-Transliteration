# Applications

Contains Applications buit around the Transliteration engine.

1. HTTPS API exposing for interation with web applications

:warning: Contents of `models` folder will not be commited.

Dependency Libraries:
* flask
* flask_cors
* torch
* numpy
* gevent

Usage:

1.HTTPS api <br>
`$ export PYTHONPATH=/path-to-apps/models/:$PYTHONPATH` <br>
`$ sudo env PATH=$PATH python3 api_expose.py`
