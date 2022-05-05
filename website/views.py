#Blueprint file defines all routes (URLs) for site 
from flask import Blueprint, render_template, request, flash, jsonify
from website.regression import ridge_pred
import numpy as np 
import json

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])

def index():
	prediction = 0
	if request.method == 'POST':
		date = request.form.get('date')
		age = request.form.get('age')
		mrt_distance = request.form.get('mrt')
		num_stores = request.form.get('conv')
		lat = request.form.get('lat')
		lon = request.form.get('lon')

		usr_input = [[date, age, mrt_distance, num_stores, lat, lon]]
		prediction = ridge_pred(usr_input)
		prediction = round(prediction[0], 2)

	return render_template("index.html", prediction = prediction)


