import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from PIL import Image
import pgeocode





st.set_page_config(layout="wide")

# Create a title and a sub title
st.write("""
# Toronto Apartment Price Machine Learning Predictive Model
""")

st.write("""# About
This webapp was created for the 2022 Daisy Intelligence Hackathon. Given several parameters, it predicts Toronto apartment rental prices using XGBoost modelling through gradient boosting. The model is trained by first splitting data into training and validation and then the XGBoost parameters are adjusted to minimize the mean absolute error to obtain a model with a higher accuracy. The webapp is created and hosted through streamlit.
""")

Image = Image.open('logo.png')
st.image(Image, use_column_width= True)

#reading in the data
appartment_data_path = "Toronto_apartment_rentals_2018.csv"
appartment_data = pd.read_csv(appartment_data_path)

#define X and y variables
y = appartment_data.Price

appartment_features = ["Bedroom","Bathroom","Den","Lat","Long"]
X = appartment_data[appartment_features]

#splitting data into training and validation
train_X, val_X, train_y, val_y = train_test_split(X,y,train_size = 0.8, test_size = 0.2, random_state = 0)

model = XGBRegressor(n_estimators = 1000,max_depth= 6, colsample_bylevel = 1,colsample_bytree = 0.6,learning_rate = 0.1,subsample = 1, reg_alpha = 0.7, reg_lambda=0.12,gamma =0.5)
model.fit(train_X, train_y)

predictions = model.predict(val_X)
mae = mean_absolute_error(predictions,val_y)

#Getting User Input
user_data = {}

with st.form('Form 1'):
    user_data["Bedrooms"] = st.selectbox('Number of Bedrooms',[1,2,3])
    user_data["Bathrooms"] = st.selectbox('Number of Bathrooms',[1,2,3])
    user_data["Den"] = st.selectbox('Number of Dens',[0,1])
    user_data["postal"] = st.text_input(label="Postal Code")
    submit = st.form_submit_button(label="Calculate")

if submit:
    user_data_list = []
    for key in user_data:
        user_data_list.append(user_data[key])

    nomi = pgeocode.Nominatim("CA")
    postal_code_data = nomi.query_postal_code(user_data_list[-1][:4])
    del(user_data_list[-1])

    user_data_list.extend(postal_code_data[["latitude","longitude"]].tolist())


    predicted_price = (model.predict(pd.DataFrame([user_data_list])))
    st.write(f'The predicted price of the apartment is: ${int(predicted_price[0])}')

st.write(f"""# Toronto Apartment Rent Price Data""")
st.write(appartment_data)
st.write(f"""# Map of Apartments Used As Data""")
st.image("heat-map.PNG")


