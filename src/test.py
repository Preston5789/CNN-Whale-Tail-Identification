
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from skimage import io
x_label = pd.read_csv('train.csv')
y_train = x_label.Id.values[x_label.Id.values != 'new_whale']


# In[ ]:


import fnmatch, os
test_path = '../test/'
test_img_list = fnmatch.filter(os.listdir(test_path),'*.jpg')


# In[ ]:


from keras.models import load_model
model = load_model('../model.h5')
w = 100
h = 100
rgb = 1
with open('ans.csv','w') as f:
    f.write('Image,Id\n')
    for img in test_img_list:
        f.write(img+",")
        test_img = io.imread(test_path + img)
        test_img = test_img.reshape(1,w,h,rgb)/255.
        r = model.predict(test_img,verbose=0)
        value = np.sort(r)[0][-4:]
        output = np.unique(y_train)[r[0].argsort()[-4:]]
        put = False
        for i in range(3,0,-1):
            if value[i]<0.97 and not put:
                f.write('new_whale ')
                put = True
            f.write(output[i] + " ")
        f.write(output[0] + "\n")
    f.close()

