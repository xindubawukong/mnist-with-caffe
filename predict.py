


import csv
import numpy as np
import caffe
import pandas as pd

test_csv_path = 'test.csv'
model_path = 'lenet.prototxt'
pretrained_path = 'lenet_iter_10000.caffemodel'

clf = caffe.Classifier(model_path, pretrained_path, image_dims=(28, 28))

a=[]
with open(test_csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    X = np.array([np.reshape([float(v) / 255 for v in row], (28, 28, 1)) for row in reader])
    for i, y in enumerate(clf.predict(X, oversample=False)):
        a.append(np.argmax(y))

n=len(a)
ImageId=[i+1 for i in range(n)]
submit_pd=pd.DataFrame({'ImageId':ImageId,'Label':a})
submit_pd.to_csv('ans.csv',index=False)