import streamlit as st

def show_popup():
    st.write("This is a pop-up message!")

def main():
    st.title("ASML 🫶")
    button_clicked = st.button("Start Signing!")
    if button_clicked:
        show_popup()

if __name__ == "__main__":
    main()
