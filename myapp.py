from flask import Flask,render_template,send_from_directory,request
from mymodel import *
import os

f1=Flask(__name__,static_folder='static')

@f1.route("/")
def home():
    return render_template("page1.html", image_path='images/download.jpg')


@f1.route("/getpredict",methods=['GET','POST'])
def getpredict():
    if request.method=='POST':
        age=request.form['age']
        sex=request.form['sex']
        bmi=request.form['bmi']
        children=request.form['children']
        smoker=request.form['smoker']
        region=request.form['region']
        
        print(age)
        print(sex)
        print(bmi)
        print(children)
        print(smoker)
        print(region)
        

        newobs=np.array([[age,sex,bmi,children,smoker,region]],dtype=float)
        print(newobs)
        model=makepredict()
        yp=model.predict(newobs)[0]
        print(yp)
        return render_template("page2.html",data=yp)
    
if __name__=="__main__":
    f1.run(debug=True)