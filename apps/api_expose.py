import os, sys

DEBUG = True
## Set in order to host in specific domain
SSL_FILES = None
# SSL_FILES = ('/etc/letsencrypt/live/xlit-api.ai4bharat.org/fullchain.pem',
#              '/etc/letsencrypt/live/xlit-api.ai4bharat.org/privkey.pem')

BASEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASEPATH)
import xlit_server

app, engine = xlit_server.get_app()

def host_https():
    https_server = WSGIServer(('0.0.0.0', 443), app,
                                     certfile=SSL_FILES[0], keyfile=SSL_FILES[1])
    print('Starting HTTPS Server...')
    https_server.serve_forever()
    return

if __name__ == '__main__':
    
    if not DEBUG: # Production Server
        from flask_cors import CORS, cross_origin
        cors = CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
        # app.run(host='0.0.0.0', port=443, ssl_context=SSL_FILES)

        from gevent.pywsgi import WSGIServer
        if SSL_FILES:
            from multiprocessing import Process
            Process(target=host_https).start()

        http_server = WSGIServer(('0.0.0.0', 80), app)
        print('Starting HTTP Server...')
        http_server.serve_forever()
    else: # Development Server
        app.run(debug=True, host='0.0.0.0', port=8000)
