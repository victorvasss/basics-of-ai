import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка набора данных Reuters и индекса слов
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    path="reuters.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    test_split=0.2,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3
)

word_index = reuters.get_word_index(path="reuters_word_index.json")

class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

# Максимальная длина последовательности
maxlen = 200

# Подготовка данных
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Применение ранней остановки, чтобы минимизировать потери
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Создаем модель GRU
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128))
model.add(GRU(256, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(46, activation='softmax'))

# Компилируем модель
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
model.fit(x_train, y_train,
          epochs=10,
          batch_size=16,
          validation_split=0.1)

# Получение вероятностей для каждого класса
predicted_probabilities = model.predict(x_test)

# Преобразование вероятностей в предсказанные классы
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Оценка модели
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(y_test, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(y_test, predicted_labels, average='weighted', zero_division=0)
roc_auc_ovr = metrics.roc_auc_score(y_test, predicted_probabilities, average='weighted', multi_class='ovr')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC-AUC: {roc_auc_ovr}')

# Подготовка для построения ROC-кривых
num_classes = 10
fpr = dict()
tpr = dict()
roc_auc = dict()

# Вычисление ROC-кривых и ROC AUC для каждого класса
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, predicted_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Строим ROC-кривые
plt.figure(figsize=(15, 15))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class "{class_names[i]}" (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Вывод матрицы ошибок
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Создание тепловой карты (heatmap) на основе матрицы ошибок
plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, fmt='3d', cmap='Blues', cbar=True)
plt.xlabel('Predicted')
plt.ylabel('True')

# Установка имен классов для меток на осях
plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
plt.yticks(np.arange(len(class_names)), class_names, rotation=0)

plt.title('Confusion Matrix')
plt.show()