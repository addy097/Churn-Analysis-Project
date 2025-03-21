# Churn Analysis Project  

## 📌 Overview  
This project analyzes **customer churn** to identify key factors affecting retention. It extracts data using **SQL**, builds a **churn prediction model in Python**, and visualizes insights with **Power BI dashboards**.  

## 📌 Reference  
This project is based on a tutorial. Follow the original steps here:  
🔗 [Tutorial Link](https://pivotalstats.com/end-end-churn-analysis-portfolio-project/)  
## 📌 Project Documentation  
I have documented this updated project on medium if you wish to understand the insights and steps: 
🔗 [Medium link](https://medium.com/@addytalpade9/end-to-end-churn-prediction-sql-power-bi-ml-an-interactive-web-app-for-churn-probability-fe77d354dbeb)  

### 🔹 **Key Enhancements**  
- The **churn prediction model** is implemented in **Kaggle notebooks** instead of Jupyter. Since I have extended the referred project by building a web app, the code file in my project exports the predicted CSV as shown in the referred tutorial plus the code of the prediction probability web application I have built for prediction probability. Use code in file: Churn-prediction-application-code.py for Kaggle processing.
- A **web app** is added for predictions (screenshot in `screenshots/`).  

## 📂 **Project Structure**  
Churn-Analysis-Project/ │-- 📂 data/ # Contains datasets │ ├── original_data.csv # Raw customer data │ ├── prediction_data.csv # Views used as input for the model │ │-- 📂 sql/ # SQL queries for data processing │ ├── SQL.md # All SQL queries used in the analysis │ │-- 📂 code/ # Python scripts for churn prediction │ ├── Python.md # Full Python code for model training │ │-- 📂 screenshots/ # Visual representation of results │ ├── Churn_Dashboard_Screenshots.pdf # Power BI dashboard screenshots │ ├── Prediction_Web_App_Screenshot.pdf # Screenshot of prediction web app │ │-- README.md # Project documentation & setup instructions

## 🚀 **How to Use This Project**  
1. Follow **SQL.md** (`sql/`) to extract and transform data.  
2. Run **Python.md** (`code/`) to train and test the churn prediction model.  
3. Check **dashboard screenshots** (`screenshots/`) for insights.  

📊 **Tech Stack:** SQL | Python | Power BI | Kaggle Notebooks  

---
