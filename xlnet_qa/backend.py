#!/usr/bin/env python
from flask import Flask, request, json, Response
from .qa_server import QAServer


class Backend:
    def __new__(cls, backend_type="shell", **args):
        if backend_type in ["restful", "restapi"]:
            return Restful(**args)
        else:
            return Shell(**args)


class Shell:
    def __init__(self, **args):
        self.session = QAServer(**args)
    
    def run(self):
        while True:
            query = input(":: ")
            response, score = self.session(query)
            print(response)
    

class Restful:
    def __init__(self, port=5000, **args):
        self.url_rule = "/api/query"
        self.methods = ["POST"]
        self.port = port
        self.app = Flask(__name__)
        self.app.add_url_rule(self.url_rule, methods=self.methods, view_func=self.get_response)
        self.session = QAServer(**args)

    def get_response(self):
        query = request.form.get('text').strip()
        session_id = request.form.get('sessionId', "123456")

        response, score = self.session(query, session_id=session_id)
        if response is None:
            return Response(json.dumps({"code":0, "message":"200 OK", 'sessionId':session_id, "data":{"response": ":)"}}), mimetype='application/json')
        return Response(json.dumps({"code":0, "message":"200 OK", 'sessionId':session_id, "data":{"response": response, "score":score}}), mimetype='application/json')

    def run(self):
        self.app.run(host='0.0.0.0', port=self.port, threaded=True)
