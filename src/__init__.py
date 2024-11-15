# src/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)
    from src.api import api
    app.register_blueprint(api)
    
    return app
