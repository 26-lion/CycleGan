from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, add, concatenate, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras import activations
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import cv2

def resnet(filter, input):
  y = Conv2D(filter, (3,3), padding="same")(input)
  y = BatchNormalization(axis=-1, momentum=0.8)(y)
  y = activations.relu(y)
  y = Conv2D(filter, (3,3), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = concatenate([y, input])
  y = activations.relu(y)
  return y

def generator_model(ImageShape):
  input = Input(ImageShape)
  y = Conv2D(32, (5,5), padding="same")(input)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = activations.elu(y)
  y = Conv2D(64, (3,3), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = activations.elu(y)
  y = Conv2D(128, (3,3), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = activations.elu(y)
  for i in range(6):
    y = resnet(128,y)
  y = Conv2DTranspose(64, (3,3), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = activations.elu(y)
  y = Conv2DTranspose(32, (3,3), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = activations.elu(y)
  y = Conv2D(3, (5,5), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  output = activations.tanh(y)
  model = Model(inputs=input, outputs=output)
  return model

def discriminator_model(ImageShape):
  input = Input(ImageShape)
  y = Conv2D(64, (4,4), padding="same")(input)
  y = LeakyReLU()(y)
  y = Conv2D(128, (4,4), padding="same")(y)
  y = BatchNormalization(axis=-1, momentum=0.8)(y)
  y = LeakyReLU()(y)
  y = Conv2D(256, (4,4), padding="same")(y)
  y = BatchNormalization(axis=-1, momentum=0.8)(y)
  y = LeakyReLU()(y)
  y = Conv2D(256, (4,4), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = LeakyReLU()(y)
  y = Conv2D(512, (4,4), padding="same")(y)
  y = BatchNormalization(axis=-1,momentum=0.8)(y)
  y = LeakyReLU()(y)
  y = Flatten()(y)
  y = Dense(64, activation="relu")(y)
  output = Dense(1, activation="sigmoid")(y)
  model = Model(inputs = input, outputs=output)
  model.compile(loss="mse", optimizer="Adam")
  return model

def gan_model(generator1, generator2, discriminator, ImageShape):
  generator1.trainable=True
  generator2.trainable=False
  discriminator.trainable=False
  #Adverserial loss
  input = Input(ImageShape)
  generated_fake = generator1(input)
  output_discriminated = discriminator(generated_fake)
  #Identity loss
  input_identity = Input(ImageShape)
  output_identity = generator1(input_identity)
  #Forward cycle loss
  output_reconstructed = generator2(generated_fake)
  #Backward cycle loss
  output_2 = generator2(input_identity)
  output_reconstructed2 = generator1(output_2)
  model = Model(inputs=[input, input_identity], outputs=[output_discriminated, output_identity, output_reconstructed, output_reconstructed2])
  model.compile(loss=["mse", "mae", "mae", "mae"], loss_weights=[1, 5, 10, 10], optimizer="Adam")
  return model

def real_images(dataset, samples):
  n = np.random.randint(0, 2500, samples)
  X = []
  for i in n:
    X.append(dataset[i])
  X = np.reshape(X, (samples, 64, 64, 3))
  y = np.ones((samples))
  return X,y
def fake_images(generator, dataset, samples):
  n = np.random.randint(0, 2500, samples)
  X = []
  for i in n:
    X.append(dataset[i])
  X = np.reshape(X, (samples, 64, 64, 3))
  X = generator.predict(X)
  y = np.zeros((len(X)))
  return X,y
def train(d_F, d_A, g_F, g_A, combined_FtoA, combined_AtoF, faces, anime):
  epochs, batch_size = 50, 10
  bat_per_epoch = int(len(faces)/batch_size)
  steps = bat_per_epoch*epochs
  for i in range(steps):
    print("Epoch"+str(i)+"th")
    X_faces, y_faces = real_images(faces, batch_size)
    X_anime, y_anime = real_images(anime, batch_size)
    X_faces_fake, y_faces_fake = fake_images(g_F, anime, batch_size)
    X_anime_fake, y_anime_fake = fake_images(g_A, faces, batch_size)
    #Gan faces to anime
    combined_FtoA.train_on_batch([X_faces, X_anime], [y_faces, X_faces, X_anime, X_faces])
    d_A.train_on_batch(X_anime, y_anime)
    d_A.train_on_batch(X_anime_fake, y_anime_fake)
    #Gan anime to faces
    combined_AtoF.train_on_batch([X_anime, X_faces], [y_anime, X_anime, X_faces, X_anime])
    d_F.train_on_batch(X_faces, y_faces)
    d_F.train_on_batch(X_faces_fake, y_faces_fake)
    if i%100==0:
      g_A.save("FacestoAnime"+str(i)+".h5")
  g_A.save("FacestoAnime.h5")

dataset = np.load("/content/drive/My Drive/gan.npz")
data_Faces, data_anime = dataset["arr_0"], dataset["arr_1"]
data_Faces, data_anime = data_Faces/255, data_anime/255
imageshape = data_Faces[0].shape[0:]
print(imageshape)

F_to_A = generator_model(imageshape)
A_to_F = generator_model(imageshape)
discriminator_of_anime = discriminator_model(imageshape)
discriminator_of_faces = discriminator_model(imageshape)
gan_F_to_A = gan_model(F_to_A, A_to_F, discriminator_of_anime, imageshape)
gan_A_to_F = gan_model(A_to_F, F_to_A, discriminator_of_faces, imageshape)
train(discriminator_of_faces, discriminator_of_anime , A_to_F, F_to_A,gan_F_to_A, gan_A_to_F, data_Faces, data_anime)
