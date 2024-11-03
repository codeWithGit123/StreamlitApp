import streamlit as st
from sklearn.datasets import load_diabetes,load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

model = LogisticRegression(max_iter=2000,solver='lbfgs')

model.fit(X_train,y_train)

yp = model.predict(X_test)

st.title("Iris Plant Predictor")
st.subheader("Select the Values")

a = st.number_input("Enter Sepal Length")
b = st.number_input("Enter Sepal Width")
c = st.number_input("Enter Petal Length")
d = st.number_input("Enter Petat Width")


if st.button("Predict"):
    inp = [[a,b,c,d]]
    pred = iris.target_names[model.predict(inp)]
    st.write(f"The Plant is: {pred[0]}")
    st.image(os.path.join(os.getcwd(),"static",f"{pred[0]}.jpeg"))
    st.write(f"The Accuracy Score is:{accuracy_score(yp,y_test)}")
