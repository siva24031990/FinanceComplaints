#To import required libs
import joblib
import data_preprocessing

#To load saved model and predict
model=joblib.load("logreg.joblib")
Y_pred2=model.predict(data_preprocessing.X_test)
print(Y_pred2)
