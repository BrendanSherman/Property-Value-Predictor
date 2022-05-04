#main executable program
from website import create_app

app = create_app()

if __name__ == '__main__': #only run web server if main.py is directly executed
	app.run(debug=True)