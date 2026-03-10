import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner as kt

# Create dataset
study_hours = np.array([2,4,6,8,5,7,3,9])
attendance = np.array([60,70,80,90,75,85,65,95])

X1 = study_hours.reshape(-1,1)
X2 = attendance.reshape(-1,1)

score = study_hours*10 + attendance*0.5
pass_fail = (score>60).astype(int)

Y1 = score.reshape(-1,1)
Y2 = pass_fail.reshape(-1,1)

# Model builder
def build_model(hp):

    input1 = tf.keras.Input(shape=(1,))
    input2 = tf.keras.Input(shape=(1,))

    merged = layers.concatenate([input1,input2])

    units = hp.Int("units",8,32,step=8)

    dense = layers.Dense(units,activation="relu")(merged)

    score_output = layers.Dense(1,name="score")(dense)
    pass_output = layers.Dense(1,activation="sigmoid",name="pass")(dense)

    model = tf.keras.Model(
        inputs=[input1,input2],
        outputs=[score_output,pass_output]
    )

    lr = hp.Choice("learning_rate",[0.01,0.001])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={"score":"mse","pass":"binary_crossentropy"}
    )

    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=3,
    directory="tuning"
)

tuner.search([X1,X2],[Y1,Y2],epochs=10,validation_split=0.2)

best_model = tuner.get_best_models(1)[0]

prediction = best_model.predict([[6],[85]])

print("Predicted Score:",prediction[0])
print("Pass Probability:",prediction[1])
