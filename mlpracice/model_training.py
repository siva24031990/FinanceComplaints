#To import required modules
import data_preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib
#To choose a model and train
"""
its a binomial or boolen classification of yes or no type
"""
logicreg=LogisticRegression()
logicreg.fit(data_preprocessing.X_train, data_preprocessing.Y_train)
Y_pred=logicreg.predict(data_preprocessing.X_test)
print(logicreg.score(data_preprocessing.X_train, data_preprocessing.Y_train))
print(confusion_matrix(data_preprocessing.Y_test, Y_pred))


#To test the selected model
print(Y_pred)
#To save or deploy the model
joblib.dump(logicreg, "logreg.joblib")