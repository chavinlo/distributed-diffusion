from flask import Flask, render_template
app = Flask(__name__, static_folder='static', static_url_path='')

@app.route('/')
def root():
    return render_template('serverlist.html')

if __name__ == '__main__':
    app.run(debug=True)
