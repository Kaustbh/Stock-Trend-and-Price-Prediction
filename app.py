# importing all necessary libraries
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
from nifty import niftyindex
from Nifty_stocks import stocks

   
def settings():
    st.title("Settings")
    st.write("Here you can configure your settings.")

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Stocks'], 
       menu_icon="cast", default_index=0,
       styles={
        "container": {"padding": "0!important", "background-color": "red"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "deepskyblue"},
        "nav-link-selected": {"background-color": "gold"},
    })

# Calling the File based on selected option

if selected == "Home":
    niftyindex()
elif selected == "Stocks":
    stocks()

