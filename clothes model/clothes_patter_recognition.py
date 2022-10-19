from __future__ import print_function
from flask import Flask, request ,jsonify
import werkzeug
import numpy as np
import pandas as pd
# from sklearn.cluster import KMeans
# import os
from sklearn.svm import SVC
from test_radon import *
import cv2
from skimage.measure.entropy import shannon_entropy
import pickle

app = Flask(__name__)

@app.route('/upload', methods=["POST"])
def upload():
    if (request.method == "POST"):
        imagefile = request.files['image']
        # im = Image.open('./color_detection/Red_Color.jpg')
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("/home/maryem/images/"+ filename)
        # im = Image.open("/home/maryem/images/"+ filename)
        def patternlessOrNot(imagepath):
            img = cv2.imread(imagepath,0)
            img =cv2.resize(img,(135,135))
            img = cv2.GaussianBlur(img,(5,5),0)
            filtered = cv2.Canny(img,150,100)
            value = shannon_entropy(filtered[30:105,30:105])
            if (value > 0.0):
                result = test(imagepath)
                return result
            else:
                return "patternless"


        def extract_siftFeature(img_path):
            path = img_path
            img = cv2.imread(path, 0)
            img =cv2.resize(img,(150,150))
            sift = cv2.xfeatures2d.SIFT_create(2000)
            kp, des = sift.detectAndCompute(img, None)
            return des

        def vstackDescriptors(descriptor_list):
            descriptors = np.array(descriptor_list[0])
            for descriptor in descriptor_list[1:]:
                descriptors = np.vstack((descriptors, descriptor))
            return descriptors


        def extractFeatures(des):
            kmeans = pickle.load(open("/home/maryem/mysite/kmeans.pkl", "rb"))
            im_features = np.array(np.zeros(100))
            for j in range(len(des)):
                feature = des[j]
                feature = feature.reshape(1, 128)
                idx = kmeans.predict(feature)
                im_features[idx] += 1
            return im_features

        def test_Model(img_path):
            x= []
            des = extract_siftFeature(img_path)
            test_features = extractFeatures(des)
            x.append(test_features)
            test_features = np.array(x)
            scale = pickle.load(open("/home/maryem/mysite/scale.pkl", "rb"))
            test_features = scale.transform(test_features)
            print("collection of sift test features is done")
            return test_features


        # def classifyTrainImages():
        #     features = pd.read_csv('training_pattern.csv')
        #     X = features.iloc[:, :260]
        #     Y = features.iloc[:, 260]
        #     svm = SVC()
        #     svm.fit(X, Y)
        #     pickle.dump(svm, open("svm.pkl", "wb"))


        def prediction(x):
            data =[]
            data.append(x)
            data = np.array(data)
            name_dict = {
                "1": "plaid",
                "2": "irregular",
                "3": "striped",
                "4": "patternless" }
            svm = pickle.load(open("/home/maryem/mysite/svm.pkl", "rb"))
            result = svm.predict(data)
            predictions = name_dict[str(int(result))]
            return predictions


        def test(image_path):
            x = []
            kernel_test = test_Model(image_path)
            kernel_test = kernel_test.flatten()
            test_features = radon_fun(image_path)

            for i in kernel_test:
                x.append(i)
            for j in test_features:
                x.append(j)
            result = prediction(x)
            return result

        path = "/home/maryem/images/"+ filename
        type = patternlessOrNot(path)

        return jsonify({
            "message": type
        })