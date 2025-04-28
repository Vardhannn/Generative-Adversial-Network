
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.image import psnr as tf_psnr, ssim as tf_ssim
from keras.applications import VGG19
from keras.models import load_model
from numpy.random import randint

from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tqdm import tqdm

#%%

print(len(tf.config.list_physical_devices("GPU")))
# %%


def res_block(ip):
    
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    
    return add([ip,res_model])

def upscale_block(ip):
    
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2 )(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model


def create_gen(gen_ip, num_res_block):
    
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)

    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9,9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)


def discriminator_block(ip, filters, strides=1, bn=True):
    
    disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)
    
    if bn:
        disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    
    return disc_model


def create_disc(disc_ip):

    df = 64
    
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)
# %%

from keras.applications import ResNet50
from keras.models import Model

def build_resnet(hr_shape):
    resnet = ResNet50(weights="imagenet", include_top=False, input_shape=hr_shape)
    return Model(inputs=resnet.inputs, outputs=resnet.layers[-1].output)

#%%
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)

    psnr_value = tf_psnr(hr_ip, gen_img, max_val=1.0)
    ssim_value = tf_ssim(hr_ip, gen_img, max_val=1.0)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features, psnr_value, ssim_value])
# %%

n=100
lr_list = os.listdir("G:/Damn/Project/data/lr_images/")[:n]

lr_images = []
for img in tqdm(lr_list):
    img_lr = cv2.imread("G:/Damn/Project/data/lr_images/" + img)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)   


hr_list = os.listdir("G:/Damn/Project/data/hr_images/")[:n]
   
hr_images = []
for img in tqdm(hr_list):
    img_hr = cv2.imread("G:/Damn/Project/data/hr_images/" + img)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)   

lr_images = np.array(lr_images)
hr_images = np.array(hr_images)
# %%

image_number = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(lr_images[image_number], (32, 32, 3)))
plt.subplot(122)
plt.imshow(np.reshape(hr_images[image_number], (128, 128, 3)))
plt.show()
# %%

lr_images = lr_images / 255.
hr_images = hr_images / 255.
# %%

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, 
                                                      test_size=0.33, random_state=42)


# %%

hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)
# %%

generator = create_gen(lr_ip, num_res_block = 16)
generator.summary()
# %%

discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
discriminator.summary()

# %%

hr_shape = (128, 128, 3)
resnet = build_resnet(hr_shape)
resnet.summary()
resnet.trainable = False
# %%

gan_model = create_comb(generator, discriminator, resnet, lr_ip, hr_ip)
gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")


#%%

def psnr_metric(y_true, y_pred):
    return tf_psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf_ssim(y_true, y_pred, max_val=1.0)


# %%

gan_model.compile(
    loss=["binary_crossentropy", "mse", "mae", "mae"],  
    loss_weights=[1e-3, 1, 0, 0],  
    optimizer="adam"
)
gan_model.summary()

#%%

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def visualize_feature_maps(model, layer_names, input_image, num_feature_maps=16):
    outputs = [model.get_layer(name).output for name in layer_names]
    submodel = Model(inputs=model.input, outputs=outputs)
    feature_maps = submodel.predict(input_image)
    for layer_name, fmap in zip(layer_names, feature_maps):
        print(f"Feature maps from layer: {layer_name}")
        plt.figure(figsize=(15, 15))
        for i in range(min(num_feature_maps, fmap.shape[-1])):
            plt.subplot(4, 4, i + 1)
            plt.imshow(fmap[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f"Feature maps from layer: {layer_name}")
        plt.show()

if __name__ == "__main__":
    image_path = "G:/Damn/Project/data/image_32.png"
    input_image = load_and_preprocess_image(image_path, target_size=(128, 128))
    layer_names = [
        'conv1_conv',
        'conv2_block1_1_conv',
        'conv2_block1_2_conv',
    ]
    visualize_feature_maps(resnet, layer_names, input_image, num_feature_maps=16)
# %%

batch_size = 64
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])
    
# %%
  
epochs = 2 
save_interval = 10  

g_losses = []
d_losses = []
psnr_values = []
ssim_values = []

for e in range(epochs):
    fake_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size, 1))

    epoch_g_losses = []
    epoch_d_losses = []
    epoch_psnr_values = []
    epoch_ssim_values = []

    for b in tqdm(range(len(train_hr_batches)), desc=f"Epoch {e + 1}/{epochs}"):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]

        fake_imgs = generator.predict(lr_imgs)
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

        image_features = resnet.predict(hr_imgs)
        g_loss, _, _, psnr_value, ssim_value = gan_model.train_on_batch(
            [lr_imgs, hr_imgs], [real_label, image_features, real_label, real_label]
        )

        epoch_g_losses.append(g_loss)
        epoch_d_losses.append(d_loss[0]) 
        epoch_psnr_values.append(np.mean(psnr_value))
        epoch_ssim_values.append(np.mean(ssim_value))

    avg_g_loss = np.mean(epoch_g_losses)
    avg_d_loss = np.mean(epoch_d_losses)
    avg_psnr = np.mean(epoch_psnr_values)
    avg_ssim = np.mean(epoch_ssim_values)

    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    psnr_values.append(avg_psnr)
    ssim_values.append(avg_ssim)

    print(f"Epoch {e + 1}/{epochs}, G Loss: {avg_g_loss}, D Loss: {avg_d_loss}, PSNR: {avg_psnr}, SSIM: {avg_ssim}")

    if (e + 1) % save_interval == 0:
        generator.save(f"G:/Damn/Project/code/generator_epoch_{e + 1}.h5")
        print(f"Model saved at epoch {e + 1}")
        
#%%

def plot_loss(loss_values, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, label=title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

plot_loss(g_losses, "Generator Loss", "Loss")

plot_loss(d_losses, "Discriminator Loss", "Loss")

plot_loss(psnr_values, "PSNR Over Epochs", "PSNR")

plot_loss(ssim_values, "SSIM Over Epochs", "SSIM")
# %%


generator = load_model('G:/Damn/Project/code/generator_epoch_100.h5', compile=False)


[X1, X2] = [lr_test, hr_test]

ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]


gen_image = generator.predict(src_image)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(src_image[0,:,:,:])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(gen_image[0,:,:,:])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(tar_image[0,:,:,:])

plt.show()


