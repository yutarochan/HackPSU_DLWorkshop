'''
Emotion Recognition Training
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
import cv2
import numpy as np
import seaborn as sns
from skimage import io
from os import listdir
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Input
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image
from keras.utils import np_utils

emo_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised']
DATA_DIR = 'data/proc/'

# Load Dataset
print 'Loading Dataset'
load_img = lambda filepath: [cv2.resize(cv2.cvtColor(io.imread(filepath+'/'+f), \
                cv2.COLOR_BGR2RGB), (64, 64)) for f in listdir(filepath)]

X = []
y = []
for i, emo in enumerate(emo_labels):
    images = load_img(DATA_DIR+emo)
    X += images
    y += [i]*len(images)

data_counts = [y.count(i) for i in range(len(emo_labels))]
y = np_utils.to_categorical(y, len(emo_labels))

print 'TOTAL IMAGES: ' + str(len(X))

# Split Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9892)

# Define and Build Model
def build_model(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=len(data_counts)):
    # Determine Input Size
    input_shape = _obtain_input_shape(input_shape, default_size=64, min_size=48, data_format=K.image_data_format(),
                        include_top=include_top)

    #  Setup Input Tensors
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (2, 2), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (2, 2), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dropout(0.3)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model considers `input_tensor`
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Generate Model
    return Model(inputs, x, name='vgg19')

# Build and Compile Model
model = build_model(include_top=True, input_shape=np.array(X_train[0]).shape, classes=len(emo_labels))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.summary()

# Train Model
print 'Training the neural network...'
nb_epoch = 20
history = model.fit(np.array(X_train), np.array(y_train), batch_size=128, epochs=nb_epoch, verbose=1,
    validation_data=(np.array(X_test), np.array(y_test)))

# Save Model Weights
print 'Saving Model'
model.save('emotion_model.h5')
json_model = model.to_json()

print 'Output Model Topology'
model_out = open('emotion_model.json', 'wb')
model_out.write(json_model)
model_out.close()

# Plot Model Accuracy and Loss
print 'Plot accuracy and loss...'
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
