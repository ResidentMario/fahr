import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# generate dummy data
def sample_threeclass(n, ratio=0.8):
    np.random.seed(42)
    y_0 = np.random.randint(2, size=(n, 1))
    switch = (np.random.random(size=(n, 1)) <= ratio)
    y_1 = ~y_0 & switch
    y_2 = ~y_0 & ~switch
    y = np.concatenate([y_0, y_1, y_2], axis=1)
    
    X = y_0 + (np.random.normal(size=n) / 5)[np.newaxis].T
    return (X, y)

X_train, y_train = sample_threeclass(1000)
X_test, y_test = sample_threeclass(100)

# build a simple keras classifier
clf = Sequential()
clf.add(Dense(3, activation='linear', input_shape=(1,), name='hidden'))
clf.add(Dense(3, activation='softmax', name='out'))
clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

# fit
clf.fit(X_train, y_train, epochs=20, batch_size=128)

# predict
y_test_pred = clf.predict(X_test)

# print a score
print(
    f'Achieved accuracy score of '
    f'{(y_test_pred.argmax(axis=1) == y_test.argmax(axis=1)).sum() / len(y_test)}.'
)

# save a model artifact
clf.save("model.h5")