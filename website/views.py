#Blueprint file defines all routes (URLs) for site 
from flask import Blueprint, render_template, request, flash, jsonify
import json

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])

def index():
	#if request.method == 'POST':
		#note = request.form.get('note')

	return render_template("index.html")


