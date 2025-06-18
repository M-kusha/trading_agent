## File: ui/utils.py
import os
import streamlit as st
import requests

API_URL = (
    st.secrets.get("API_URL")
    or os.getenv("API_URL")
    or "http://localhost:8000"
).rstrip("/")

def get_json(path: str):
    r = requests.get(f"{API_URL}{path}")
    r.raise_for_status()
    return r.json()

def put_json(path: str, payload: dict):
    r = requests.put(f"{API_URL}{path}", json=payload)
    r.raise_for_status()
    return r.json()
