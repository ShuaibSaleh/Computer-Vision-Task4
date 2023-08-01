from flask import Flask, render_template, request
from flask_cors import CORS
import os
import base64  # convert from string to  bits
import json
import cv2
import numpy as np
import time
import calendar
import image as img1
import json
import functions as fn
import regiongrowing as reggrow
from regiongrowing import Point
import agglomerative as agglo
from agglomerative import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

CORS(app)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("processing.html")


@app.route("/processing", methods=["GET", "POST"])
def processing():
    path = ""

    if request.method == "POST":

        imageA_data = base64.b64decode(
            request.form["imageA_data"].split(',')[1])

        img_path = img1.saveImage(imageA_data, "input_image")

        operation = request.form["operation"]
        subOperation = request.form["subOperation"]

        if operation == 'spt':
            path = fn.spectral_thresholding(img_path)
        elif operation == 'rtl':
            path = fn.RGB_LUV(img_path)
        elif operation == 'mss_bandwidth':
            path = fn.mean_shift_segmentation(img_path, int(subOperation))
        elif operation == 'optimal_threshold':
            path = fn.optimal_threshold(img_path)
        elif operation == 'otsu_threshold':
            path = fn.otsu_threshold(img_path)
        elif operation == 'k_means':
            path = fn.kmeans_segmentation(img_path, 30, 160)

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        return json.dumps({1: f'<img src="{path}?t={time_stamp}" id="imageC" alt="" >'})

    else:
        return render_template("processing.html")


@app.route("/regiongrowing", methods=["GET", "POST"])
def regiongrowing():
    if request.method == "POST":
        image_data = base64.b64decode(
            request.form["image_data"].split(',')[1])

        img_path = img1.saveImage(image_data, "regiongrowing_img")

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        seed_points = []
        for i in range(3):
            x = np.random.randint(0, img.shape[0])
            y = np.random.randint(0, img.shape[1])
            seed_points.append(Point(x, y))

        binaryImg = reggrow.Region_growing(img_gray, seed_points, 10)

        final_img = './static/images/output/output.jpg'

        # plt.savefig(final_img)
        plt.imsave('./static/images/output/output.jpg', binaryImg, cmap='gray')

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        output_path = './static/images/output/output.jpg'

        # return json.dumps({1: f'test'})
        return json.dumps({1: f'<img src="{output_path}?t={time_stamp}" id="ApplyEdges" alt="" >'})

    else:
        return render_template("regiongrowing.html")


@app.route("/agglomerative", methods=["GET", "POST"])
def agglomerative():
    if request.method == "POST":
        image_data = base64.b64decode(
            request.form["image_data"].split(',')[1])

        img_path = img1.saveImage(image_data, "regiongrowing_img")

        img0 = cv2.imread(img_path)
        img = cv2.cvtColor(img0, cv2.COLOR_RGB2Luv)

        pixels = img.reshape(img.shape[0]*img.shape[1], 3)
        print("image pixels: ", pixels.shape)
        n_clusters = 5
        agglo = AgglomerativeClustering(k=n_clusters, initial_k=25)
        agglo.fit(pixels)

        new_img = [[agglo.predict_center(list(pixel))
                    for pixel in row] for row in img]
        new_img = np.array(new_img, np.uint8)
        new_img = cv2.cvtColor(new_img.astype('float32'), cv2.COLOR_Luv2RGB)
        mpimg.imsave("./static/images/output/output.jpg", new_img)

        final_img = './static/images/output/output.jpg'

        # plt.savefig(final_img)
        # plt.imsave('./static/images/output/output.jpg', binaryImg, cmap='gray')

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        output_path = './static/images/output/output.jpg'

        # return json.dumps({1: f'test'})
        return json.dumps({1: f'<img src="{output_path}?t={time_stamp}" id="ApplyEdges" alt="" >'})

    else:
        return render_template("agglomerative.html")


if __name__ == "__main__":
    app.run(debug=True, port=5017)
