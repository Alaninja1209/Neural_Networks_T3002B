#Alan Alfredo Onofre Chavez A01632858
from sklearn import datasets
from sklearn.svm import SVC #Libreria de clasificacion

iris = datasets.load_iris() #Leyenda datos del conjunto iris

#Por comodidad, separar los datos
X = iris.data #Matriz de los predictores
Y = iris.target #Separar las etiquetas (vector)

model = SVC()

model.fit(X, Y) 

#Preguntarle cosas al modelo
print(model.predict([[1,2,3,4], [2,2,3,1]])) #De acuerdo a lo que aprendio de los valores de entrada