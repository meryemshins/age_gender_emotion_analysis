import seaborn as sns #veri görselleştirme için kullanılan bir Python kütüphanesidir.
from sklearn.metrics import confusion_matrix  #Karışıklık matrisi, bir sınıflandırma modelinin performansını değerlendirmek için kullanılan bir metriktir.
#Gerçek ve tahmin edilen sınıflar arasındaki ilişkiyi gösteren bir tabloyu döndürür.
import tensorflow as tf
from sklearn import metrics # Sınıflandırma, regresyon, kümeleme ve diğer makine öğrenimi görevlerinin değerlendirilmesi için kullanılabilen metrikler sağlar.
from keras.models import load_model # Önceden eğitilmiş bir Keras modelini diskten yüklemek için kullanılır.
import pandas as pd  #veri analizi ve manipülasyonu için kullanılan güçlü bir Python kütüphanesidir.
#Verileri yüklemek, işlemek, temizlemek, dönüştürmek ve analiz etmek için çeşitli fonksiyonları sağlar.

import numpy as np
import os # Dosya/dizin işlemleri için kullanılır.
import matplotlib.pyplot as plt  #veri görselleştirme için kullanılan bir Python kütüphanesidir. plt olarak kısaltılarak kullanılır ve grafik çizme işlevlerini sağlar.
import cv2
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten, Input # Katmanlar
from sklearn.model_selection import train_test_split  #  Veri setini eğitim ve test alt kümelerine ayırmak için kullanılır. 


physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
"""
 TensorFlow kütüphanesini kullanarak sisteminizdeki GPU'ları tespit etmek için kullanılır.
 tf.config.list_physical_devices('GPU') ifadesi ile GPU cihazlarının listesi alınır ve len(physical_devices) ifadesi ile bu listedeki GPU sayısı hesaplanır.
 Sonuç, ekrana "Num GPUs: X" şeklinde yazdırılır, burada X, sisteminizdeki GPU sayısını temsil eder.
"""

path = "./UTKFace/UTKFace"  #dataset yolu.
images = []
age = []
gender = []
for img in os.listdir(path):  #belirtilen dizindeki dosya listesini döngüyle gezerek her bir dosya için işlem yapmayı sağlar.

    #img.split("_"): img isimli dosyanın adını '_' karakterine göre böler ve böldüğü parçaları bir liste olarak döndürür.
    #Örneğin, bir dosya adı "25_male.jpg" ise, bu ifade ['25', 'male', 'jpg'] listesini döndürür.
    ages = img.split("_")[0] #img isimli dosyanın adını '_' karakterine göre böldüğümüzde elde edilen parçalardan ilkini (0'ıncı indeksi) seçer. 
    #Bu örnekte, '25' değerini elde ederiz. Bu değer, dosyanın yaşa ilişkin bilgisini temsil eder.
    
    genders = img.split("_")[1]  #img isimli dosyanın adını '_' karakterine göre böldüğümüzde elde edilen parçalardan ikincisini (1'inci indeksi) seçer.
    #Bu örnekte, 'male' değerini elde ederiz. Bu değer, dosyanın cinsiyete ilişkin bilgisini temsil eder.
    
    img = cv2.imread(str(path)+"/"+str(img))  #path (dizin yolu) ve img (dosya adı) bilgilerini kullanarak belirtilen dosyayı okur. 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #görüntüyü BGR renk formatından RGB renk formatına dönüştürür. Bu dönüşüm, renk kanallarının sırasını değiştirir.
    images.append(np.array(img)) #dönüştürülmüş görüntüyü (img) bir NumPy dizisi olarak images listesine ekler.
    age.append(np.array(ages))  #yaş bilgisini (ages) bir NumPy dizisi olarak age listesine ekler.
    gender.append(np.array(genders))  #cinsiyet bilgisini (genders) bir NumPy dizisi olarak gender listesine ekler.

age = np.array(age, dtype=np.int64) #age listesini bir NumPy dizisine dönüştürür ve veri türünü 64-bit tamsayı olarak belirler.

images = np.array(images)
gender = np.array(gender, np.uint64)

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(
    images, age, random_state=42)
#images dizisini öğrenme verileri (x_train_age), age dizisini hedef değerler (y_train_age), rastgele bir şekilde karıştırarak eğitim ve test veri kümelerine böler.
#random_state=42 parametresi, rastgele bölme işleminin tekrarlanabilir olmasını sağlar, yani her çalıştırıldığında aynı bölme sonuçlarını verir.

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(
    images, gender, random_state=42)
#(cinsiyet için aynı işlemler)

#Yaş
age_model = Sequential() #Sequential(): Bu ifade, boş bir sıralı model oluşturur. Sıralı model, katmanları ardışık bir şekilde eklemenizi sağlayan basit bir model türüdür.
age_model.add(Conv2D(128, kernel_size=3, activation='relu',
              input_shape=(200, 200, 3)))  ##age_model modeline bir Conv2D katmanı ekler. Conv2D katmanı, evrişimli sinir ağı (CNN) modellerinde kullanılan bir tür konvolüsyon katmanıdır.
#128: katmandaki filtre sayısını belirtir, kernel_size=3: filtre boyutunu belirtir. Burada 3x3 boyutunda bir filtre kullanılır, activation='relu': aktivasyon fonksiyonunu belirtir. 
#input_shape=(200, 200, 3): Bu parametre, katmana giriş verisinin şeklini belirtir. (200, 200, 3) değeri, 200x200 piksel boyutunda renkli (RGB) bir görüntüyü temsil eder.
#Bu katman, 128 adet 3x3 boyutunda filtre kullanarak giriş verisini işler ve ReLU aktivasyon fonksiyonunu uygular.

age_model.add(MaxPool2D(pool_size=3, strides=2))#maksimum havuzlama (MaxPooling) katmanı ekler. pool_size=3 parametresi, havuzlama işlemi için kullanılan pencere boyutunu belirtirken, strides=2 parametresi adım boyutunu belirtir. 
#Bu katman, görüntü boyutunu küçültmeye ve özelliklerin özetini çıkarmaya yardımcı olur.
age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
#Yukarıdaki satırlar, bir havuzlama katmanını ve ardından bir konvolüsyon katmanını tekrarlar. Bu işlem, modelin daha derin ve karmaşık özellikler öğrenmesine yardımcı olur.
age_model.add(Flatten())  #verilerin düzleştirilerek vektör haline getirildiği düzleştirme (Flatten) katmanını ekler. Bu katman, 2D verileri (örneğin, görüntüler) 1D vektöre dönüştürür.
age_model.add(Dropout(0.2)) #overfittingi önlemek için dropout katmanı ekler. 0.2 parametresi, her bir ağırlık bağlantısının kapatılma olasılığını belirtir.
age_model.add(Dense(512, activation='relu')) # tam bağlantılı (fully connected) bir katman ekler. 512 parametresi, katmandaki nöron sayısını belirtir.
age_model.add(Dense(1, activation='linear', name='age')) #çıktı katmanını ekler. 1 parametresi, çıktı nöronu sayısını belirtir. 
age_model.compile(optimizer='adam', loss='mse', metrics=['mae']) #modelin derlenmesini sağlar. 
print(age_model.summary()) # modelin özetini ekrana yazdırır. Özet, modelin katmanları, çıkış şekilleri ve toplam parametre sayısı gibi bilgileri içerir.

history_age = age_model.fit(x_train_age, y_train_age,
                            validation_data=(x_test_age, y_test_age), epochs=100)  
 #Eğitim işlemi gerçekleştirildikten sonra, history_age değişkenine eğitim geçmişi kaydedilir.

age_model.save('age_model_100epochs.h5')  #age_model modelini "age_model_100epochs.h5" adlı bir H5 dosyası olarak kaydeder. H5 formatı, Keras tarafından kullanılan bir model kaydetme formatıdır.

#Yaş için yapılan tüm işlemler Cinsiyet için de yapılıyor.
gender_model = Sequential()
gender_model.add(Conv2D(36, kernel_size=3, activation='relu',
                 input_shape=(200, 200, 3)))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(512, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))
gender_model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_gender = gender_model.fit(x_train_gender, y_train_gender,
                                  validation_data=(x_test_gender, y_test_gender), epochs=100)
gender_model.save('gender_model_100epochs.h5')

history = history_age  #Bu ifade, history_age değişkenini history değişkenine kopyalar.

loss = history.history['loss']  #eğitim kaybı değerlerini loss değişkenine atar. history.history sözlüğü, eğitim sürecindeki kayıp ve metrik değerlerini içerir.
val_loss = history.history['val_loss']  #doğrulama kaybı değerlerini val_loss değişkenine atar.
epochs = range(1, len(loss) + 1)  # epoch (döngü) sayılarını temsil eden bir dizi oluşturur. len(loss) ile eğitim sürecindeki toplam epoch sayısı elde edilir.
plt.plot(epochs, loss, 'y', label='Training loss')  #eğitim kaybının epoch'e göre değişimini sarı renkte çizer.
plt.plot(epochs, val_loss, 'r', label='Validation loss') #doğrulama kaybının epoch'e göre değişimini kırmızı renkte çizer.
plt.title('Training and validation loss')  # grafiğin başlığını belirler.
plt.xlabel('Epochs') # x ekseninin etiketini belirler.
plt.ylabel('Loss')  # y ekseninin etiketini belirler.
plt.legend()
plt.show()

acc = history.history['accuracy'] 
# eğitim sürecindeki doğruluk metriklerini (accuracy) acc değişkenine atar. history.history sözlüğü, eğitim sürecindeki kayıp ve metrik değerlerini içerir.

val_acc = history.history['val_accuracy'] 
# doğrulama sürecindeki doğruluk metriklerini (val_accuracy) val_acc değişkenine atar.

#Bu şekilde, eğitim ve doğrulama süreçlerinde elde edilen doğruluk metriklerini kullanarak bir analiz veya görselleştirme yapılır.

plt.plot(epochs, acc, 'y', label='Training acc') # Train doğruluk oranı
plt.plot(epochs, val_acc, 'r', label='Validation acc')  # Doğrulama oranı
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()  #legend: açıklama kutusu
plt.show()

my_model = load_model('gender_model_100epochs.h5', compile=False)
#"gender_model_100epochs.h5" adlı model dosyasını yükler. compile=False parametresi, modelin derlenmemesini sağlar. Bu parametre, modelin önceden derlenmiş olduğunu varsayar ve yeniden derleme işlemine gerek olmadığını belirtir.

predictions = my_model.predict(x_test_gender) #my_model modelini kullanarak x_test_gender veri kümesi üzerinde tahminler yapar ve sonuçları predictions değişkenine atar. Tahmin sonuçları, olasılık değerlerini içeren bir dizi olarak elde edilir.
y_pred = (predictions >= 0.5).astype(int)[:, 0] # tahmin sonuçlarını (predictions) 0.5 eşik değeri üzerinden sınıflara dönüştürür. 0.5 veya daha büyük olan olasılıklar pozitif sınıfa, daha küçük olanlar ise negatif sınıfa atanır. .astype(int) ifadesi, sınıf etiketlerini tamsayıya dönüştürür. [:, 0] ifadesi ise tahmin sonuçlarının sadece ilk sütununu seçer (pozitif sınıfı temsil eder).
#Bu şekilde, my_model modeli üzerindeki tahmin sonuçlarını sınıflara dönüştürerek y_pred değişkeninde saklanır.
print("Accuracy = ", metrics.accuracy_score(y_test_gender, y_pred))
#y_test_gender gerçek etiketlerle y_pred tahmin etiketleri arasındaki doğruluk değerini hesaplar ve ekrana yazdırır. metrics.accuracy_score fonksiyonu, gerçek ve tahmin edilen etiketler arasındaki doğruluk değerini hesaplar.
cm = confusion_matrix(y_test_gender, y_pred) #y_test_gender gerçek etiketlerle y_pred tahmin etiketleri arasındaki karışıklık matrisini hesaplar. confusion_matrix fonksiyonu, gerçek ve tahmin edilen etiketler arasındaki karışıklık matrisini oluşturur.
sns.heatmap(cm, annot=True) #karışıklık matrisini ısı haritası olarak görselleştirir. sns.heatmap fonksiyonu, verilen karışıklık matrisini ısı haritası olarak çizer. annot=True parametresi, her hücreye değerleri yazdırmak için kullanılır.
