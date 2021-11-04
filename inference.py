import cv2
import numpy as np
from keras.models import load_model

# Load all kinds of labels
print("Loading labels")
index2class = []
with open("2021VRDL_HW1_datasets/classes.txt", 'r') as file:
    while True:
        line = file.readline()
        if not line:  # EOF encountered
            break
        index2class.append(line.split('\n')[0])

# Assign index to each of the labels
class2index = dict()
for i, c in enumerate(index2class):
    class2index[c] = i

# Keep the order of reading testing data
x_test_order = []
with open("2021VRDL_HW1_datasets/testing_img_order.txt", 'r') as file:
    while True:
        line = file.readline()
        if not line:  # EOF encountered
            break
        x_test_order.append(line.split('\n')[0])


# Load images of testing data in the specified order
def load_images(img_dir, load_order):
    dataset = []
    for file in load_order:
        img_path = img_dir + '/' + file
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, (224, 224))
        dataset.append(img_resize)
    return np.array(dataset)

print("Loading images")
x_test = load_images("2021VRDL_HW1_datasets/testing_images", x_test_order)

# Load model and do prediction
print("Loading model")
model = load_model("model.h5")
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

# Generate submission file
print("Producing submission file")
submission = []
for i in range(len(x_test_order)):
    img = x_test_order[i]
    pred_class = index2class[y_pred[i]]
    submission.append([img, pred_class])

np.savetxt('answer.txt', submission, fmt='%s')
