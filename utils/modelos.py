
def skip_conection_pruebas(input_img):
    from tensorflow.keras import layers
    from tensorflow.keras.layers import BatchNormalization
    import numpy as np
    
    #Encoder
    y = layers.Conv2D(16, kernel_size=(3,3), padding='same')(input_img)
    y = layers.MaxPooling2D(pool_size=(2, 2))(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(32, kernel_size=(3,3), padding='same')(y)
    y = layers.MaxPooling2D(pool_size=(2, 2))(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, kernel_size=(3,3), padding='same')(y)
    y = layers.MaxPooling2D(pool_size=(2, 2))(y)
    y = layers.LeakyReLU()(y)
    y1 = layers.Conv2D(128, kernel_size=(3,3), padding='same')(y) #1
    y = layers.MaxPooling2D(pool_size=(2, 2))(y1)
    y = layers.LeakyReLU()(y)
    y2 = layers.Conv2D(256, kernel_size=(3,3), padding='same')(y) #2
    y = layers.MaxPooling2D(pool_size=(2, 2))(y2)
    y = layers.LeakyReLU()(y2)
    #y = layers.Conv2D(512, (3, 3), padding='same')(y) 
    #y = layers.LeakyReLU()(y)

    #Flattening for the bottleneck
    vol = y.shape
    x = layers.Flatten()(y)
    latent = layers.Dense(1, activation='relu')(x)     
 
    #Decoder
    y = layers.Dense(np.prod(vol[1:]), activation='relu')(latent)
    y = layers.Reshape((vol[1], vol[2], vol[3]))(y)

    #y = layers.Conv2DTranspose(512, (3, 3), padding='same')(y)
    #y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(256, kernel_size=(3,3), padding='same')(y)
    y = layers.UpSampling2D((2,2))(y)
    y = layers.LeakyReLU()(y)
    #y = layers.Add()([y2, y]) #2
    #y = layers.LeakyReLU()(y)
    #y = BatchNormalization()(y)
    y = layers.Conv2DTranspose(128, kernel_size=(3,3), padding='same')(y)
    #y = layers.UpSampling2D((2,2))(y)
    #y = layers.UpSampling2D((2,2))(y)
    y = layers.LeakyReLU()(y)
    y = layers.Add()([y1, y]) #1
    y = layers.LeakyReLU()(y)
    y = BatchNormalization()(y)
    y = layers.Conv2DTranspose(64, kernel_size=(3,3), padding='same')(y)
    y = layers.UpSampling2D((2,2))(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(32, kernel_size=(3,3), padding='same')(y)
    y = layers.UpSampling2D((2,2))(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(16, kernel_size=(3,3), padding='same')(y)
    y = layers.UpSampling2D((2,2))(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2DTranspose(1, kernel_size=(3,3), activation='sigmoid', padding='same')(y)    
    decoded = y

    return decoded


def dn_cnn(input_img):
    from tensorflow.keras import layers, activations
    from tensorflow.keras.layers import BatchNormalization
    import numpy as np
    
    #Encoder
    #y = layers.Conv2D(1024, (3, 3), padding='same')(input_img)
    #y = BatchNormalization()(y)
    #y = layers.LeakyReLU()(y)
    #y = layers.Conv2D(1024, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    #y = layers.LeakyReLU()(y)
    #y = layers.Conv2D(512, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    #y = layers.LeakyReLU()(y)
    
    y = layers.Conv2D(64, (3, 3), padding='same')(input_img)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (5, 5), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (5, 5), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (5, 5), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (5, 5), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (5, 5), padding='same')(y)
    #y = BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(1, (5, 5), padding='same')(y) #activation='sigmoid'
    y = layers.LeakyReLU()(y)
    #y = layers.Subtract()([input_img, y])
    y = layers.Add()([input_img, y]) #skip-1
    y = layers.Activation('sigmoid')(y)
    decoded = y
    return decoded


def vae_definitive(input_img):
    import tensorflow
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model
    from keras import backend as K
    from tensorflow.python.keras import backend as K
    from keras import metrics
    from keras.datasets import mnist
    from keras import layers
    from keras.callbacks import ModelCheckpoint

    original_dim = 25600
    # set the dimensionality of the latent space to a plane for visualization later
    latent_dim = 1
    epsilon_std = 1.0

    #Encoder
    y = layers.Conv2D(32, (3, 3), padding='same')(input_img) #80
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y) #40
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(128, (3, 3), padding='same')(y) # skip-1 #20
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(256, (3, 3), padding='same')(y) #10
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(512, (3, 3), padding='same')(y)# skip-2 #5
    y = layers.LeakyReLU()(y)
    y = layers.Conv2D(1024, (3, 3), padding='same')(y)
    y = layers.LeakyReLU()(y)

    #Flattening for the bottleneck
    vol = y.shape
    x = layers.Flatten()(y)

    #latent
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    def sampling(args):
        z_mean, z_log_var = args
        batch = tensorflow.shape(z_mean)[0]
        epsilon = tensorflow.keras.backend.random_normal(shape=(batch, latent_dim), mean=0.,
                                stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    x = layers.Dense(160*160*1024, activation="relu")(z)
    x = layers.Reshape((160,160,1024))(x)

    #Decoder
    y = layers.Conv2DTranspose(1024, (3, 3), padding='same')(y) #10
    y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(512, (3, 3), padding='same')(y)# skip-2 #5
    y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(256, (3, 3), padding='same')(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(128, (3,3), padding='same')(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(64, (3,3), padding='same')(y)
    y = layers.LeakyReLU()(y)
    y = layers.Conv2DTranspose(32, (3,3), padding='same')(y)
    y = layers.LeakyReLU()(y)

    x_decoded_mean = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(x)    

    return x_decoded_mean


def resnet(input_img):
    from tensorflow.keras import layers
    from tensorflow.keras.layers import BatchNormalization
    import numpy as np
    
    y = layers.Conv2D(64, (3, 3), padding='same')(input_img)
    y = layers.LeakyReLU()(y)
    
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(128, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(128, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(128, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(128, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(128, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(128, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(256, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(256, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(256, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(256, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(256, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(256, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(512, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(512, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(512, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(512, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    #RESNET block#####################
    # First convolutional layer
    y1 = layers.Conv2D(512, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    # Second convolutional layer
    y = layers.Conv2D(512, (3, 3), padding='same')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    #skip
    y = layers.Add()([y1, y])
    y = layers.LeakyReLU()(y)
    y = layers.BatchNormalization()(y)
    ###################################
    y = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(y)  
    return y