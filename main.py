import numpy as np
import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import random
import statsmodels.api as sm

def main():
	
	st.title("Machine Learning Dashboard")
	st.sidebar.title("Dashboard")
	st.sidebar.markdown("This is for changing the model parameters")

	@st.cache(persist=True)
	def load_data():
		df=pd.read_csv('dataset.csv')
		val=df['Outcome'].value_counts()
		

		for i in range(val[0]-val[1]+1):

			new_row = {'Pregnancies': df[df['Outcome']==1]['Pregnancies'].mean()+random.randint(-2,2), 'Glucose':df[df['Outcome']==1]['Glucose'].mean()+random.randint(-5,5),
			 'BloodPressure':df[df['Outcome']==1]['BloodPressure'].mean()+random.randint(-2,2),
			 'SkinThickness':df[df['Outcome']==1]['SkinThickness'].mean()+random.randint(-2,2),
			 'Insulin':df[df['Outcome']==1]['Insulin'].mean()+random.randint(-2,2),
			 'DiabetesPedigreeFunction':df[df['Outcome']==1]['DiabetesPedigreeFunction'].mean()+random.uniform(-0.1,0.1),
			 'Age':df[df['Outcome']==1]['Age'].mean()+random.randint(-5,5),
			 'Outcome':1.0}
			df = df.append(new_row, ignore_index=True)

		df.fillna(df.mean(),inplace=True)
		df.Outcome = df.Outcome.astype(int)
		
		return df


	df=load_data()
	
	print(df.isnull().any().any())
	

	if  st.button("Show dataset", key='show df'):
		st.write(df)
		


	@st.cache(persist=True)
	def split(df):
		y = df['Outcome']
		x=df.drop(columns=['Outcome'])
		x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = 0.3)
		return x_train, x_test, y_train, y_test


	class_names = ['Diabeties', 'Normal']
	x_train, x_test, y_train, y_test = split(df)

	def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
			st.pyplot()

		if 'ROC Curve' in metrics_list:
			st.subheader("ROC Curve")
			plot_roc_curve(model, x_test, y_test)
			st.pyplot()
		
		if 'Precision-Recall Curve' in metrics_list:
			st.subheader('Precision-Recall Curve')
			plot_precision_recall_curve(model, x_test, y_test)
			st.pyplot()


	st.sidebar.subheader("Choose Classifier")
	classifier = st.sidebar.selectbox("Classifier", ("SVM", "Random Forest", "Logistic Regression","Decision Tree"))
	if classifier == 'SVM':
		st.sidebar.subheader("Model Hyperparameters")
		C = st.sidebar.number_input("Regularization (C)", 0.01, 10.0, step=0.01, key='C_SVM')
		kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
		gamma = st.sidebar.radio("Gamma (Kernel Cofficient)", ("scale", "auto"), key='gamma')
		metrics = st.sidebar.multiselect("Plot Metrices", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

		if st.sidebar.button("Classify", key='classify'):
			st.subheader("Results of SVM")
			model = SVC(C=C, kernel=kernel, gamma=gamma)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)

	if classifier == 'Random Forest':
		st.sidebar.subheader("Model Hyperparameters")
		n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
		max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
		bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
		metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

		if st.sidebar.button("Classify", key='classify'):
			st.subheader("Random Forest Results")
			model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)

	if classifier == 'Logistic Regression':
		metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
		if st.sidebar.button("Classify", key='classify'):
			st.subheader("Logistic Regression Results")
			model = LogisticRegression(random_state=0)
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)

	if classifier == 'Decision Tree':
		metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
		if st.sidebar.button("Classify", key='classify'):
			st.subheader("Decision Tree Results")
			model = DecisionTreeClassifier()
			model.fit(x_train, y_train)
			accuracy = model.score(x_test, y_test)
			y_pred = model.predict(x_test)
			st.write("Accuracy: ", accuracy.round(2))
			st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
			st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
			plot_metrics(metrics)


	st.sidebar.subheader('Hypothesis testing for Logistic Regression')
	if st.sidebar.button('Test'):
		logmodel=sm.GLM(df['Outcome'],sm.add_constant(df.drop(columns=['Outcome'])),family=sm.families.Binomial())
		summary=logmodel.fit().summary()
		print(summary)






if __name__ == '__main__':
	main()

