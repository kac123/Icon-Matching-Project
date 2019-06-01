import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import cv2

import zernike
import util

UPLOAD_FOLDER = 'uploads'
RESULTS = 'results'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS'] = RESULTS

maybe = {}
zernike_database = util.load_obj('zernike_database_icon_10')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fid = str(uuid.uuid4())
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            maybe[fid] = zernike.create_query(img)
            return redirect(url_for('uploaded_file',
                                    filename=fid))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename): #cv2.imread
    return str(zernike.test_query(maybe[filename], zernike_database))
    return send_from_directory(app.config['RESULTS'],
                               "heysexy.txt")