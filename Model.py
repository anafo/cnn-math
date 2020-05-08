
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

r=1
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = r)

# MODEL PARAMS
batch_s = 16
n_epochs = 30
n_steps = X_train.shape[0]//batch_s
annealer = LearningRateScheduler(lambda x: 1e-5 * 0.8 ** x, verbose=1)

model = Sequential()
model.add(Conv2D(16, kernel_size=3, activation = "relu", input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=3, activation = "relu"))
model.add(BatchNormalization())
model.add(Conv2D(16, kernel_size=3, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation = "relu", input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation = "relu"))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.25))
model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation = "linear"))

model.compile(optimizer = Adam(lr=1e-5), loss = 'mean_squared_logarithmic_error', metrics = ['mean_absolute_error'])
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs = n_epochs,
                    callbacks=[annealer],
                    verbose=2).history