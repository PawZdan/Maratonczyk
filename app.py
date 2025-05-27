import streamlit as st
import json
import pandas as pd  # type: ignore
from pycaret.regression import load_model, predict_model  # type: ignore
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel
from typing import Optional
import re
#import joblib
import os
from langfuse import Langfuse
from langfuse.decorators import observe

# ----------- MODEL & FILES -----------
MODEL_NAME = 'best_model'

# ----------- SESSION STATE INIT -----------
if "step" not in st.session_state:
    st.session_state.step = 0

# ----------- CACHED FUNCTIONS -----------
load_dotenv()

@st.cache_data
def go_to_next_step():
    st.session_state.step += 1
    st.experimental_rerun()  # nowa nazwa metody zamiast st.rerun()

@st.cache_resource
def load_model_1():
    return load_model(MODEL_NAME)
@st.cache_resource  # cacheujemy klienta, by nie tworzyć go za każdym razem
def get_langfuse():
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"]
    )

@st.cache_resource
def get_client():
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return instructor.from_openai(openai_client)

def format_seconds_to_hms(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

class Person(BaseModel):
    age: Optional[int]
    gender: Optional[bool]  # 0 = kobieta, 1 = mężczyzna
    tempo: Optional[float]  # np. "24 minuty"
# Pobieramy klienta Langfuse raz i przekazujemy go do dekoratora observe
langfuse = get_langfuse()

@observe()
def retrieve_structure(text: str) -> Person:
    res = get_client().chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_model=Person,
        messages=[
            {
                "role": "user",
                "content": (
                    "Wyodrębnij dane do obiektu Person:\n"
                    "- `age`: liczba całkowita (wiek)\n"
                    "- `gender`: bool (0 = kobieta, 1 = mężczyzna)\n"
                    "- `tempo`: tempo biegu w formacie min/km np 5:30 min/km, wyodrebnij wtedy 5.3 \n"
                    f"Tekst użytkownika:\n{text}"
                ),
            }
        ],
    )
    return res

def check_missing_fields(data) -> list[str]:
    missing = []
    if data.age is None:
        missing.append("wiek")
    if data.gender is None:
        missing.append("płeć")
    if data.tempo is None:
        missing.append("tempo (czas w sekundach)")
    return missing

# ----------- STEP 0: WELCOME -----------

if st.session_state.step == 0:
    model = load_model_1()
    st.title("aplikacja maratończyk")

    user_input = st.text_input("Podaj następujące dane: płeć, wiek i tempo na 5km:")

    if user_input:
        data = retrieve_structure(user_input)
        missing = check_missing_fields(data)

        if missing:
            st.warning(f"Brakuje następujących danych: {', '.join(missing)}. Spróbuj jeszcze raz.")
        else:
            st.success("Wszystkie dane zostały podane!")
            df = pd.DataFrame({
                "Płeć": [data.gender],
                "Wiek": [data.age],
                "5 km Tempo": [data.tempo]
            })
            pred = predict_model(model, data=df)
            #st.write(pred)

            #pred = model.predict_model(df)
            total_seconds = pred['prediction_label'].iloc[0]

            szacowany_czas = format_seconds_to_hms(total_seconds)

            st.write(f"Szacowany czas na przebiegnięcie półmaratonu to: {szacowany_czas}")

            #go_to_next_step()

# ----------- STEP 1 -----------

elif st.session_state.step == 1:
    st.write("Kolejny krok aplikacji...")
