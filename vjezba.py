
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
 



 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('titanic.csv')
data_df = data.dropna()
data_df = pd.DataFrame(data, columns=['Pclass','Sex','Fare','Embarked','Survived']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['Survived']).to_numpy
y = data_df['Survived'].to_numpy



# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# a)
model = keras.Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(units=12, activation="relu"))
model.add(layers.Dense(units=8, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.summary()

# b)
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy", ])

# c)
history = model.fit(X_train, y_train, batch_size=10,
                    epochs=150, validation_split=0.1)


# d)
#model.save('Model/')

# e)
#model = load_model('Model/')
score = model.evaluate(X_test, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

# f)
y_predictions = model.predict(X_test)
y_predictions = np.around(y_predictions).astype(np.int32)
cm = confusion_matrix(y_test, y_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
# komentar u pdfu
