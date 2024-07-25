import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, Callback
import matplotlib.pyplot as plt


class TerminateOnThreshold(Callback):
    def __init__(self, monitor='val_accuracy', threshold=0.99):
        super(TerminateOnThreshold, self).__init__()
        self.monitor = monitor
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get(self.monitor) >= self.threshold:
            print(f"\nStopping training as {self.monitor} reached {self.threshold}")
            self.model.stop_training = True

def create_train_datagen():
    return ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.1, 1],
        channel_shift_range=150.0,
        preprocessing_function=None
    )

def create_val_datagen():
    return ImageDataGenerator()

def train_model(model, data, labels, batch_size=32, epochs=50):
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_datagen = create_train_datagen()
    val_datagen = create_val_datagen()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    def lr_schedule(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    terminate_on_threshold = TerminateOnThreshold()

    history = model.fit(train_generator, 
                        validation_data=val_generator, 
                        epochs=epochs, 
                        callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint, terminate_on_threshold])

    for epoch in range(len(history.history['loss'])):
        print(f"Epoch {epoch+1}/{epochs} - loss: {history.history['loss'][epoch]:.4f} - accuracy: {history.history['accuracy'][epoch]:.4f} - val_loss: {history.history['val_loss'][epoch]:.4f} - val_accuracy: {history.history['val_accuracy'][epoch]:.4f}")

    # Построение графиков
    plot_history(history)

    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # График точности на обучающей и проверочной выборках
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # График потерь на обучающей и проверочной выборках
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()