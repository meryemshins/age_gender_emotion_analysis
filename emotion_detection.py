from keras.preprocessing.image import ImageDataGenerator #  ImageDataGenerator: Görüntü verilerini işlemek için kullanılan bir veri artırma ve ön işleme aracıdır.
import tensorflow as tf
from tensorflow.keras.models import Sequential # Model oluşturmak için kullanılır.
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np
import random
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
"""
 TensorFlow kütüphanesini kullanarak sisteminizdeki GPU'ları tespit etmek için kullanılır.
 tf.config.list_physical_devices('GPU') ifadesi ile GPU cihazlarının listesi alınır ve len(physical_devices) ifadesi ile bu listedeki GPU sayısı hesaplanır.
 Sonuç, ekrana "Num GPUs: X" şeklinde yazdırılır, burada X, sisteminizdeki GPU sayısını temsil eder.
"""

IMG_HEIGHT=48 
IMG_WIDTH = 48
batch_size=32

train_data_dir='./fer2013/train/'
validation_data_dir='./fer2013/test/'

"""
Veri artırma (data augmentation),
 makine öğrenmesi ve derin öğrenme modellerinin eğitiminde kullanılan bir tekniktir. 
 Bu teknik, mevcut veri setinin çeşitliliğini artırarak modelin genelleme yeteneğini iyileştirmeyi hedefler.
"""
train_datagen = ImageDataGenerator(     # ImageDataGenerator sınıfını kullanarak bir görüntü veri artırma (data augmentation) yapılandırması oluşturuyor.
					rescale=1./255,  # Piksel değerlerinin 0-255 aralığından 0-1 aralığına yeniden ölçeklendirilmesini sağlar.
					rotation_range=30,  # Rastgele olarak en fazla 30 derece döndürme işlemi uygular.
					shear_range=0.3,  # Görüntüleri rastgele olarak en fazla 0.3 birim kaydırma işlemi uygular.
					zoom_range=0.3,  # Rastgele olarak en fazla %30 oranında yakınlaştırma veya uzaklaştırma işlemi uygular.
					horizontal_flip=True,  # Görüntüleri yatay olarak rastgele çevirme işlemi uygular.
					fill_mode='nearest')  # Döndürme veya kaydırma işlemleri sırasında oluşan boş piksellerin en yakın komşu piksellerle doldurulmasını sağlar.

validation_datagen = ImageDataGenerator(rescale=1./255) 

#train_generator: Veri üreteci nesnesidir ve eğitim sırasında modelin beslenmesi için kullanılabilecek akış halinde veriler sağlar.
train_generator = train_datagen.flow_from_directory(  # Dizindeki görüntüleri okumak, eğitim için uygun bir formatta veri akışı oluşturur.
					train_data_dir,  # Eğitim veri setinin bulunduğu dizin yolunu belirtir.
					color_mode='grayscale',  # Görüntülerin gray modda yükleneceğini belirtir.
					target_size=(IMG_HEIGHT, IMG_WIDTH),  # Hedef boyutu belirtir. 
					batch_size=batch_size,  
					class_mode='categorical',  # Kategorik olarak etiketlenmiş sınıfları kullanılacağını gösterir.
					shuffle=True)  # Eğitim verilerinin her bir batch de karıştırılacağını belirtir.

# Test veri seti için aynı işlemler uygulanır.
validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(IMG_HEIGHT, IMG_WIDTH),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


class_labels=['KIZGIN','IGRENME', 'KORKMUS', 'MUTLU','NOTR','UZGUN','SASKIN']

img, label = train_generator.__next__()  #__next__(): bir sonraki batch e geçer.

# Rastgele bir örneğin görüntüsünü ve etiketini görselleştirmek için kullanılır.
i=random.randint(0, (img.shape[0])-1) 
image = img[i]  
labl = class_labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

train_path = "./fer2013/train/"
test_path = "./fer2013/test/"

# train_path ve test_path olarak belirtilen dizinlerdeki eğitim ve test görüntüleri için toplam görüntü sayısını hesaplamak için kullanılır.
num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)


epochs=100

#model.fit(): eğitim verileriyle modeli eğitir ve eğitim sürecinin geçmişini (history) döndürür.
history=model.fit(train_generator,
                steps_per_epoch=num_train_imgs//batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=num_test_imgs//batch_size)

model.save('emotion_detection_model_100epochs.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


my_model = load_model('emotion_detection_model_100epochs.h5', compile=False)  # Eğitilen modeli yükler.


"""
validation_generator veri üretecinin bir sonraki doğrulama veri topluğunu alarak, my_model modeli kullanarak bu görüntülerin 
sınıflandırma tahminlerini yapar ve gerçek etiketleri ile karşılaştırır. 
"""
test_img, test_lbl = validation_generator.__next__() 
predictions=my_model.predict(test_img)  # Model için tahmin tutar.
predictions = np.argmax(predictions, axis=1) # Dizideki her bir tahmin için en yüksek olasılığa sahip olan sınıfın indeksini alır. 
test_labels = np.argmax(test_lbl, axis=1) 

print ("Accuracy = ", metrics.accuracy_score(test_labels, predictions))

cm = confusion_matrix(test_labels, predictions)  

sns.heatmap(cm, annot=True)

class_labels=['KIZGIN','IGRENME', 'KORKMUS', 'MUTLU','NOTR','UZGUN','SASKIN']


"""
rastgele bir örneğin görüntüsünü seçer, gerçek etiketini ve tahmin edilen etiketini alır ve seçilen görüntüyü
gri tonlamalı olarak görselleştirir. Bu şekilde, gerçek ve tahmin edilen etiketleri karşılaştırarak modelin
performansını değerlendirmek için kullanılır.
"""
n=random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.imshow(image[:,:,0], cmap='gray')
plt.title("Original label is:"+orig_labl+" Predicted is: "+ pred_labl)
plt.show()