import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import *
import cv2
import glob
from tqdm import *
#################### LSB ######################

coverImg = cv2.imread("image/datasets_200511_442144_data_forest_bost101.jpg",cv2.IMREAD_COLOR)
coverImg.astype("uint8")
secretImg = cv2.imread("image/datasets_200511_442144_data_forest_cdmc280.jpg",cv2.IMREAD_COLOR)
secretImg.astype("uint8")
encodedImg = coverImg

width,height,channel = coverImg.shape


name1 = "CoverImg -> SecretImg"
cv2.namedWindow(name1)
cv2.imshow(name1,cv2.hconcat((coverImg,secretImg)))

encodedImg = ((coverImg >> 4) << 4) + (secretImg >> 4)


name2 = "CoverImg ->  EncodedImg ->  SubtractImg"

cv2.namedWindow(name2)
cv2.imshow(name2,cv2.hconcat((coverImg,encodedImg,coverImg-encodedImg)))

decode_coverImg = (encodedImg >> 4) << 4
decode_secretImg = (encodedImg << 4)

name3 = "Decode_CoverImg -> Decode_SecretImg"

cv2.namedWindow(name3)
cv2.imshow(name3,cv2.hconcat((decode_coverImg,decode_secretImg)))

cv2.waitKey(0)
cv2.destroyAllWindows()


#################### NN ######################
def revLoss(s_true, s_pred):
    return beta * K.sum(K.square(s_true - s_pred))

def fullLoss(y_true, y_pred):
    s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
    s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]
    s_loss = revLoss(s_true, s_pred)
    c_loss = K.sum(K.square(c_true - c_pred))
    return s_loss + c_loss


def xMakeImageEncoder(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_3x3')(input_S)
    x2 = Conv2D(25, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_4x4')(input_S)
    x3 = Conv2D(15, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_5x5')(input_S)
    x4 = Conv2D(10, (6, 6), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_6x6')(input_S)
    x5 = Conv2D(5, (7, 7), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_7x7')(input_S)
    x = concatenate([x1,x2,x3, x4, x5])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_prep1_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_prep1_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_prep1_5x5')(x)
    x = concatenate([x1, x2, x3])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_prep2_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_prep2_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_prep2_5x5')(x)
    x = concatenate([x1, x2, x3])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_prep3_3x3')(input_C)
    x2 = Conv2D(25, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_prep3_4x4')(input_C)
    x3 = Conv2D(15, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_prep3_5x5')(input_C)
    x4 = Conv2D(10, (6, 6), strides=(1, 1), padding='same', activation='relu', name='conv_prep3_6x6')(input_C)
    x5 = Conv2D(5, (7, 7), strides=(1, 1), padding='same', activation='relu', name='conv_prep3_7x7')(input_C)

    x = concatenate([x,x1,x2,x3,x4,x5])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_3x3')(x)
    x2 = Conv2D(25, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_4x4')(x)
    x3 = Conv2D(15, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_5x5')(x)
    x4 = Conv2D(10, (6, 6), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_6x6')(x)
    x5 = Conv2D(5, (7, 7), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_7x7')(x)
    x = concatenate([x1,x2,x3, x4, x5])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid1_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid1_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid1_5x5')(x)
    x = concatenate([x1, x2, x3])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_3x3')(x)
    x2 = Conv2D(25, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_4x4')(x)
    x3 = Conv2D(15, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_5x5')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_6x6')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_7x7')(x)
    x = concatenate([x1,x2, x3, x4, x5])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid3_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid3_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid3_5x5')(x)
    x = concatenate([x1, x2, x3])
    output_Cprime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', name='output_C')(x)

    return Model(inputs=[input_S, input_C],outputs=output_Cprime,name = 'ImageEncoder')


def xMakeImageDecoder(input_size, fixed=False):
    reveal_input = Input(shape=(input_size))
    input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_3x3')(input_with_noise)
    x2 = Conv2D(25, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_4x4')(input_with_noise)
    x3 = Conv2D(15, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_5x5')(input_with_noise)
    x4 = Conv2D(10, (6, 6), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_6x6')(input_with_noise)
    x5 = Conv2D(5, (7, 7), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_7x7')(input_with_noise)
    x = concatenate([x1,x2,x3, x4, x5])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev1_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev1_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev1_5x5')(x)
    x = concatenate([x1, x2, x3])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev2_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev2_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev2_5x5')(x)
    x = concatenate([x1, x2, x3])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev3_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev3_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev3_5x5')(x)
    x = concatenate([x1, x2, x3])

    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev4_3x3')(x)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev4_4x4')(x)
    x3 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev5_5x5')(x)
    x = concatenate([x1, x2, x3])

    output_Sprime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', name='output_S')(x)

    if not fixed:
        return Model(inputs=reveal_input,outputs=output_Sprime, name='ImageDecoder')
    else:
        return Container(inputs=reveal_input,outputs=output_Sprime,name='ImageDecoderFixed')

def xMakeModel(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))

    encoder = xMakeImageEncoder(input_size)
    decoder = xMakeImageDecoder(input_size)

    decoder.compile(optimizer='adam', loss=revLoss)
    decoder.trainable = False

    output_Cprime = encoder([input_S, input_C])
    output_Sprime = decoder(output_Cprime)

    autoencoder = Model(inputs=[input_S, input_C],outputs=concatenate([output_Sprime, output_Cprime]))
    autoencoder.compile(optimizer='adam', loss=fullLoss)
    return encoder, decoder, autoencoder


path = glob.glob(r'.\tiny-imagenet-200\train\n01443537\images\*.JPEG')
beta = 1.0
cv_img = np.zeros(shape = (500,64,64,3))
count = 0
for img in path:
    cv_img[count,:,:,:] = cv2.imread(img)
    count = count + 1


secretImg = cv_img[:250,:,:,:]
coverImg = cv_img[250:,:,:,:]
oriSecretImg = secretImg.copy()
oriCoverImg = coverImg.copy()

secretImg = secretImg/255
coverImg = coverImg/255


for i in range(0,250):
    for j in range(0,3):
        secretImg[i,:,:,j] = cv2.dct(np.float32(secretImg[i,:,:,j]))
        coverImg[i,:,:,j] = cv2.dct(np.float32(coverImg[i,:,:,j]))
        for row in range(0,len(secretImg[i,:,0,j])):
            for col in range(0,len(secretImg[i,0,:,j])):
                if row >= len(secretImg[i,:,0,j]) and col >= len(secretImg[i,0,:,j]):
                    secretImg[i, row, col, j] = 0
                    coverImg[i, row, col, j] = 0
        secretImg[i, :, :, j] = cv2.idct(secretImg[i,:,:,j])
        coverImg[i, :, :, j] = cv2.idct(coverImg[i, :, :, j])

encoder_model, reveal_model, autoencoder_model = xMakeModel(secretImg.shape[1:])

autoencoder_model.load_weights('HidingImage50loop.hdf5')
"""
BATCH_SIZE = 32
m = secretImg.shape[0]
loss_history = []
for epoch in range(0,50):
    np.random.shuffle(secretImg)
    np.random.shuffle(coverImg)

    t = tqdm(range(0, secretImg.shape[0], BATCH_SIZE), mininterval=0)
    ae_loss = []
    revLoss = []
    for idx in t:
        sbatch = secretImg[idx:min(idx + BATCH_SIZE, m)]
        cbatch = coverImg[idx:min(idx + BATCH_SIZE, m)]

        C_prime = encoder_model.predict([sbatch, cbatch])

        ae_loss.append(autoencoder_model.train_on_batch(x=[sbatch, cbatch], y=np.concatenate((sbatch, cbatch), axis=3)))
        revLoss.append(reveal_model.train_on_batch(x=C_prime,y=sbatch))
        K.set_value(autoencoder_model.optimizer.lr, 0.001)
        K.set_value(reveal_model.optimizer.lr, 0.001)

        t.set_description('Epoch {} | Batch: {:3} of {}. Loss AE {:10.2f} | Loss Rev {:10.2f}'.format(epoch + 1, idx, m,np.mean(ae_loss), np.mean(revLoss)))
    loss_history.append(np.mean(ae_loss))

#autoencoder_model.save_weights('HidingImage.hdf5')

"""


decoded = autoencoder_model.predict([secretImg[50:60], coverImg[50:60]])
decoded_S, decoded_C = decoded[...,0:3], decoded[...,3:6]

plt.subplot(221)
plt.imshow(secretImg[59])
plt.subplot(222)
plt.imshow(coverImg[59])
plt.subplot(223)
plt.imshow(decoded_S[9])
plt.subplot(224)
plt.imshow(decoded_C[9])
plt.show()


#################### Optional Task  - LSB ######################


coverImg = cv2.imread("image/datasets_200511_442144_data_forest_bost101.jpg",cv2.IMREAD_COLOR)
coverImg.astype("uint8")
secretImg1 = cv2.imread("image/datasets_200511_442144_data_forest_cdmc280.jpg",cv2.IMREAD_COLOR)
secretImg1.astype("uint8")
secretImg2 = cv2.imread("image/datasets_200511_442144_data_Opencountry_art582.jpg",cv2.IMREAD_COLOR)
secretImg2.astype("uint8")
encodedImg = coverImg.copy()

name1 = "CoverImg -> SecretImg1 -> SecretImg2"
cv2.namedWindow(name1)
cv2.imshow(name1,cv2.hconcat((coverImg,secretImg1,secretImg2)))

width,height,channel = coverImg.shape

secretImg1_Bayer = np.zeros([width,height])
secretImg1_Bayer.astype("uint8")
secretImg2_Bayer = np.zeros([width,height])
secretImg2_Bayer.astype("uint8")
secretImg1_deBayer = np.zeros([width,height,channel])
secretImg1_deBayer.astype("uint8")
secretImg2_deBayer = np.zeros([width,height,channel])
secretImg2_deBayer.astype("uint8")

for row in range(0,height):
    count = 1
    for col in range(0,width):
        if row%2 == 0: ## RGRGRGRG
            if count == 1:
                secretImg1_Bayer[row, col] = secretImg1[row, col, 0]
                secretImg2_Bayer[row, col] = secretImg2[row, col, 0]
                count = count + 1
            elif count == 2:
                secretImg1_Bayer[row, col] = secretImg1[row, col, 1]
                secretImg2_Bayer[row, col] = secretImg2[row, col, 1]
                count = 1
        elif row%2 == 1: ## GBGBGBGBGB
            if count == 1:
                secretImg1_Bayer[row, col] = secretImg1[row, col, 1]
                secretImg2_Bayer[row, col] = secretImg2[row, col, 1]
                count = count + 1
            elif count == 2:
                secretImg1_Bayer[row, col] = secretImg1[row, col, 2]
                secretImg2_Bayer[row, col] = secretImg2[row, col, 2]
                count = 1

coverImg[:,:,0] = ((np.uint8(coverImg[:,:,0]) >> 4) << 4) + (np.uint8(secretImg1_Bayer) >> 4)
coverImg[:,:,1] = ((np.uint8(coverImg[:,:,1]) >> 4) << 4) + (np.uint8(secretImg2_Bayer) >> 4)


name2 = "CoverImg ->  EncodedImg ->  SubtractImg"
cv2.namedWindow(name2)
cv2.imshow(name2,cv2.hconcat((encodedImg,coverImg,coverImg-encodedImg)))


secretImg1_Bayer = np.uint8(coverImg[:,:,0]) << 4
secretImg2_Bayer = np.uint8(coverImg[:,:,1]) << 4
for row in range(0,height-1):
    count = 1
    for col in range(0,width-1):
        if row%2 == 0: ## RGRGRGRG
            if count == 1:
                secretImg1_deBayer[row, col, 0] = secretImg1_Bayer[row    , col    ]
                secretImg1_deBayer[row, col, 1] = secretImg1_Bayer[row    , col + 1]
                secretImg1_deBayer[row, col, 2] = secretImg1_Bayer[row + 1, col + 1]
                secretImg2_deBayer[row, col, 0] = secretImg2_Bayer[row    , col    ]
                secretImg2_deBayer[row, col, 1] = secretImg2_Bayer[row    , col + 1]
                secretImg2_deBayer[row, col, 2] = secretImg2_Bayer[row + 1, col + 1]
                count = count + 1
            elif count == 2:
                secretImg1_deBayer[row, col, 0] = secretImg1_Bayer[row    , col - 1]
                secretImg1_deBayer[row, col, 1] = secretImg1_Bayer[row    , col    ]
                secretImg1_deBayer[row, col, 2] = secretImg1_Bayer[row + 1, col    ]
                secretImg2_deBayer[row, col, 0] = secretImg2_Bayer[row    , col - 1]
                secretImg2_deBayer[row, col, 1] = secretImg2_Bayer[row    , col    ]
                secretImg2_deBayer[row, col, 2] = secretImg2_Bayer[row + 1, col    ]
                count = 1
        elif row%2 == 1: ## GBGBGBGBGB
            if count == 1:
                secretImg1_deBayer[row, col, 0] = secretImg1_Bayer[row - 1 ,col    ]
                secretImg1_deBayer[row, col, 1] = secretImg1_Bayer[row    , col    ]
                secretImg1_deBayer[row, col, 2] = secretImg1_Bayer[row    , col + 1]
                secretImg2_deBayer[row, col, 0] = secretImg2_Bayer[row - 1, col    ]
                secretImg2_deBayer[row, col, 1] = secretImg2_Bayer[row    , col    ]
                secretImg2_deBayer[row, col, 2] = secretImg2_Bayer[row    , col + 1]
                count = count + 1
            elif count == 2:
                secretImg1_deBayer[row, col, 0] = secretImg1_Bayer[row - 1 ,col - 1]
                secretImg1_deBayer[row, col, 1] = secretImg1_Bayer[row    , col - 1]
                secretImg1_deBayer[row, col, 2] = secretImg1_Bayer[row    , col    ]
                secretImg2_deBayer[row, col, 0] = secretImg2_Bayer[row - 1, col - 1]
                secretImg2_deBayer[row, col, 1] = secretImg2_Bayer[row    , col - 1]
                secretImg2_deBayer[row, col, 2] = secretImg2_Bayer[row    , col    ]
                count = 1
name3 = "Decode_CoverImg -> Decode_SecretImg1 -> Decode_SecretImg2"

decode_coverImg = coverImg.copy()
decode_coverImg[:,:,0] = np.uint8(decode_coverImg[:,:,0] >> 4) << 4
decode_coverImg[:,:,1] = np.uint8(decode_coverImg[:,:,1] >> 4) << 4

cv2.namedWindow(name3)
cv2.imshow(name3,cv2.hconcat((np.uint8(decode_coverImg),np.uint8(secretImg1_deBayer),np.uint8(secretImg2_deBayer))))
cv2.waitKey(0)