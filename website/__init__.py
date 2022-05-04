#initializes Flask application and SQLalchemy database
from flask import Flask
from os import path

def create_app():
	app = Flask(__name__)
	app.config['SECRET_KEY'] = 'secret'

	#Imports blueprints from within package
	from .views import views
	#Connects blueprints to flask app
	app.register_blueprint(views, url_prefix='/')

	return app

