from keras.models import load_model   #Keras kütüphanesinde eğitilmiş bir modeli yüklemek için kullanılır. Bu işlev, modelin yapılandırmasını, ağırlıklarını ve eğitim durumunu bir dosyadan geri yükler.
#from time import sleep
from tensorflow.keras.utils import img_to_array  #bir görüntüyü NumPy dizisine dönüştürmek için kullanılır. 
                                                 #Bu işlev, genellikle görüntü verilerini Keras modellerine beslemek veya görüntü işleme işlemleri gerçekleştirmek için kullanılır.
from keras.preprocessing import image  #görüntü verilerini ön işleme ve veri artırma işlemleri için kullanılan işlevleri içerir.
import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')  #opecv kütüphanesini kullanarak yüz algılama için bir Cascade sınıflandırıcısı (classifier) oluşturur. Cascade sınıflandırıcısı, yüzleri tespit etmek için kullanılan önceden eğitilmiş bir modeldir. 
#Yüz tespiti için face_classifier nesnesini kullanırken, genellikle detectMultiScale yöntemi kullanılır. Bu yöntem, görüntüdeki yüzleri tespit etmek için kullanılır.

emotion_model = load_model('./emotion_detection_model_100epochs.h5')  #yüklemek istediğimiz eğitilmiş modelin kaydedildiği h5 dosyasının yolunu ve adını temsil eder.
age_model = load_model('./age_model_100epochs.h5')
gender_model = load_model('./gender_model_100epochs.h5')

class_labels=['KIZGIN','IGRENME', 'KORKMUS', 'MUTLU','NOTR','UZGUN','SASKIN']  #duygu sınıflandırma probleminde kullanılan sınıf etiketlerini temsil eden bir liste tanımlar. Her bir etiket, belirli bir duyguyu temsil eder.
gender_labels = ['KADIN', 'ERKEK']

cap=cv2.VideoCapture(0)  #cv2.VideoCapture() işlevi, bir video yakalama nesnesi oluşturur. 0 parametresi, ilk kamera aygıtını temsil eder. 
#Eğer birden fazla kamera aygıtı varsa, farklı bir indeks numarası kullanarak diğer aygıtları da seçebiliriz.
#cap değişkeni, oluşturulan video yakalama nesnesini temsil eder. Bu nesne, kameradan video akışını yakalamak ve işlemek için kullanılır.

while True:
    ret,frame=cap.read()  #cap nesnesi üzerinden read() yöntemini kullanarak video akışından kareler okunur.
    #ret değeri, karenin başarıyla okunup okunmadığını gösteren bir bayraktır. frame ise, okunan kareyi temsil eden bir dizi veya matris objesidir. Bu kareler üzerinde işlem yapılır, görüntü işleme teknikleri uygulanır.
    labels=[] #etiketler dizisi 
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #cvtcolor() işlevi bir renk dönüşümü gerçekleştirir ve görüntünün renk uzayını değiştirir.
    #frame dönüştürülmek istenen renkli görüntüyü temsil eder. BGR renk uzayından gri tonlamalı (grayscale) renk uzayına dönüşüm gerçekleştirilir.
    #Gri tonlamalı görüntü, bazı görüntü işleme yöntemlerinde kullanılan veya daha sonra modele beslemek için ön işleme adımı olarak kullanılan bir formattır.
    faces=face_classifier.detectMultiScale(gray,1.3,5)  #detectMultiScale() yöntemi, yüzleri tespit etmek için Cascade sınıflandırıcısını kullanır.
    #Bu yöntem, verilen görüntü üzerinde yüz tespiti yapar ve tespit edilen yüzlerin konumlarını dikdörtgen olarak döndürür.
    #faces değişkeni, tespit edilen yüzlerin konumlarını temsil eden bir dikdörtgen listesi olarak döndürülür. Her bir dikdörtgen, tespit edilen yüzün konumunu (x, y, width, height) temsil eder.
    
    for (x,y,w,h) in faces:
        #tespit edilen yüzlerin çevresine dikdörtgen çizme, ilgili alanı (ROI - Region of Interest) gri tonlamalı görüntü olarak alıp yeniden boyutlandırma işlemlerini gerçekleştirir.
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #mavi, 2 kalınlığı temsil eder.
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) #☻yeniden boyutlandırma.

        #Görüntünün tahmin için hazırlanması
        roi=roi_gray.astype('float')/255.0 #stype('float') yöntemi, ROI'yi veri tipini float olarak dönüştürür.
        #Bunu yapmanın nedeni, genellikle görüntü verilerini 0-255 aralığından 0-1 aralığına normalize etmektir. ROI'yi 255'e bölmek, piksel değerlerini 0 ile 1 arasında bir aralığa getirir.
        #Bu, modelin daha iyi öğrenme performansı göstermesine yardımcı olabilir.
        
        roi=img_to_array(roi)  #ROI görüntüsünü diziye dönüştürür. img_to_array() yöntemi, NumPy dizisine dönüştürmek için kullanılır. Bu, görüntüyü modelin girişine verebilmek için uygun formata getirir.
        roi=np.expand_dims(roi,axis=0) 
        #np.expand_dims() yöntemi, dizi boyutunu belirtilen eksende (burada ekseni 0 olarak belirtiyoruz) genişletir. Bu, modele giriş olarak tek bir örnek sağlamak için gereklidir.

        #Duygu
        preds=emotion_model.predict(roi)[0] 
        #predict() yöntemi, modelin verilen giriş üzerinde tahmin yapmasını sağlar. [0] indeksi, tahmin sonuçlarının ilk örneğini (yani tek bir tahmini) temsil eder.
        label=class_labels[preds.argmax()]  # tahmin sonuçlarından en yüksek olasılığa sahip duygu etiketini (label) belirler. preds.argmax() yöntemi, tahmin sonuçlarının en yüksek olasılığa sahip indeksini döndürür.
        #Bu indeks, class_labels listesindeki ilgili duygu etiketini belirlemek için kullanılır.
        
        label_position=(x,y)  # etiketin görüntü üzerindeki konumunu (label_position) belirler. Bu, dikdörtgenin sol üst köşesinin koordinatlarına (x, y) karşılık gelir.
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)  #rijinal görüntü üzerine yazdırmak için cv2.putText() işlevini kullanır. 
        #frame=görüntü, label=yazı, label_position=yazının konumu, cv2.FONT_HERSHEY_SIMPLEX= yazı tipi, 0,255,0=yeşil, 2=yazı kalınlığı.
        
        #Cisiyet
        roi_color=frame[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 
        gender_label_position=(x,y+h+50) #Etiketi yüzün dışına taşımak için 50 piksel aşağıda
        cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Yaş
        age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        age = round(age_predict[0,0])
        age_label_position=(x+h,y+h)
        cv2.putText(frame,"Yas="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
   
    cv2.imshow('Analiz', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  #cap.release() komutu, video yakalama cihazının kullanımını sonlandırır ve kaynakları serbest bırakır.
cv2.destroyAllWindows() #tüm açık pencereleri kapatır ve bellek sızıntılarını önlemek için kullanılan kaynakları serbest bırakır.