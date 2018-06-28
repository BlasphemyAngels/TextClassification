from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Convolution1D
from keras.layers import Convolution2D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.layers import concatenate

def train(x_train, y_train, x_test, y_test, vocab_size, embedding_size, batch_size, pred_data, epochs):

    length = len(x_train[0])
    main_input=Input(shape=(batch_size,),dtype='int32')

    embedder=Embedding(vocab_size + 1, embedding_size)

    embed=embedder(main_input)

    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)

    cnn1 = MaxPool1D(pool_size=4)

    cnn2 = Convolution1D(256, 4,padding='same',strides=1,activation='relu')(embed)

    cnn2 = MaxPool1D(pool_size=4)(cnn2)

    cnn3 = Convolution1D(256, 5, padding='same',strides=1,activation='relu')(embed)

    cnn3 = MaxPool1D(pool_size=4)(cnn3)

    # 合并三个模型的输出向量

    cnn = concatenate([cnn1,cnn2,cnn3],axis=-1)

    flat = Flatten()(cnn)

    drop = Dropout(0.2)(flat)

    main_output = Dense(2, activation='softmax')(drop)

    model = Model(inputs=main_input,outputs=main_output)

    model.summary()

    # try using different optimizers and different optimizer configs
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
    pred = model.predict_classes(pred_data)
    print('Test score:', score)
    print("pred", pred)
    print('Test accuracy:', acc)
