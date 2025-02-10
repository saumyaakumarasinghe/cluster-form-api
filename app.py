from flask import Flask


def create_app(script_info=None):
    app = Flask(__name__)
    app.config.from_object("config.DevelopmentConfig")

    from routes import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    return app
