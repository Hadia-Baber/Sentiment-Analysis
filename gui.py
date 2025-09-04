import customtkinter as ctk
from PIL import Image
import threading
import time
import joblib   
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter

# Load your trained model and vectorizer
model = joblib.load("sentiment_model_140.pkl")
vectorizer = joblib.load("tfidf_vectorizer_140.pkl")

# Load icons
plus_icon = ctk.CTkImage(light_image=Image.open("icons/plus.png"), size=(10, 10))
predict_icon = ctk.CTkImage(light_image=Image.open("icons/predictive.png"), size=(22, 22))
smile_icon = ctk.CTkImage(light_image=Image.open("icons/smile.png"), size=(24, 24))
upset_icon = ctk.CTkImage(light_image=Image.open("icons/sad.png"), size=(24, 24))

# Initialize CTk
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("sweetkind.json")  # Custom theme

app = ctk.CTk()
app.configure(fg_color="#3d6282")
app.geometry("650x580")
app.title("🌸 Sentiment Predictor")

# --- Main Frame ---
main_frame = ctk.CTkFrame(app, fg_color="#EBF1F2")
main_frame.pack(fill="both", expand=True, padx=30, pady=20)

# --- Title ---
header_icon = ctk.CTkImage(light_image=Image.open("icons/self-control.png"), size=(45, 45))
title_label = ctk.CTkLabel(
    main_frame,
    image=header_icon,
    text="   Sentiment Analyzer",
    compound="left",
    font=ctk.CTkFont(family="Georgia", size=26, weight="bold"),
    text_color="#3C3842",
)
title_label.pack(pady=(25, 10))

# --- Entry Frame ---
entry_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
entry_frame.pack(fill="both", expand=False)

# List to hold input entries
entry_widgets = []

def add_input_field():
    entry = ctk.CTkEntry(
        entry_frame,
        width=570,
        height=38,
        fg_color="white",
        text_color="#444444",
        border_color="#3d6282",
        border_width=1.1,
        font=ctk.CTkFont(size=15),
        placeholder_text="Enter a sentence...",
    )
    entry.pack(pady=6)
    entry_widgets.append(entry)

# Add the first input field
add_input_field()

# --- Result Section ---
results_heading = ctk.CTkLabel(
    main_frame,
    text="Results",
    font=("Segoe UI", 18, "bold"),
    text_color="#3C3842"
)

result_frame = ctk.CTkFrame(main_frame, fg_color="#FFFFFF", border_color="#CCCCCC", border_width=1)
chart_frame = ctk.CTkFrame(main_frame, fg_color="#FFFFFF", border_color="#CCCCCC", border_width=1)

result_label = ctk.CTkLabel(
    result_frame,
    text="",
    font=ctk.CTkFont(size=15, weight="normal"),
    wraplength=500,
    text_color="#1D1D1D",
    justify="left",
)
result_label.pack(pady=12, padx=10)

# --- Loading label ---
loading_label = ctk.CTkLabel(main_frame, text="", font=ctk.CTkFont(size=14, slant="italic"))

# --- Clear All ---
def clear_all():
    for entry in entry_widgets:
        try:
            entry.destroy()
        except:
            continue
    entry_widgets.clear()
    add_input_field()
    result_frame.pack_forget()
    results_heading.pack_forget()
    chart_frame.pack_forget()

# --- Chart Drawing ---
def show_bar_chart(counts):
    for widget in chart_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    sentiments = list(counts.keys())
    values = list(counts.values())

    ax.bar(sentiments, values, color=["#f44336", "#4caf50"])
    ax.set_title("Sentiment Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Sentiment")
    ax.set_ylim(0, max(values) + 1)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# --- Prediction Logic ---
import mysql.connector

def insert_sentiment(sentence, sentiment):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",         # Change if different
            password="",         # Change if you set a password
            database="sentiment_analysis"
        )

        cursor = connection.cursor()
        query = "INSERT INTO predictions (sentence, sentiment) VALUES (%s, %s)"
        values = (sentence, sentiment)
        cursor.execute(query, values)
        connection.commit()

        cursor.close()
        connection.close()

    except mysql.connector.Error as err:
        print("❌ Database Error:", err)

def simulate_prediction():
    loading_label.configure(text="⏳ Predicting...")
    loading_label.pack()
    time.sleep(0.5)

    inputs = []
    for entry in entry_widgets:
        try:
            text = entry.get().strip()
            if text:
                inputs.append(text)
        except:
            continue

    for widget in result_frame.winfo_children():
        widget.destroy()

    if not inputs:
        results_heading.pack(pady=(20, 5))
        result_frame.pack(pady=(15, 10), padx=10, fill="both", expand=False)
        loading_label.pack_forget()

        no_input_label = ctk.CTkLabel(result_frame, text="No input detected.", font=ctk.CTkFont(size=15))
        no_input_label.pack(pady=10)
        chart_frame.pack_forget()
        return

    X = vectorizer.transform(inputs)
    predictions = model.predict(X)

    label_map = {
        0: ("Negative", upset_icon),
        1: ("Positive", smile_icon)
    }

    counts = Counter()

    for text, sentiment in zip(inputs, predictions):
        sentiment_text, icon = label_map.get(sentiment, ("❓ Unknown", None))
        counts[sentiment_text] += 1

        insert_sentiment(text, sentiment_text)
        row = ctk.CTkFrame(result_frame, fg_color="transparent")
        row.pack(pady=5)
        row.pack_configure(anchor="center")

        emoji_label = ctk.CTkLabel(row, image=icon, text="")
        emoji_label.pack(side="left", padx=(0, 10))

        result_text = f"\"{text}\" → {sentiment_text}"
        text_label = ctk.CTkLabel(
            row,
            text=result_text,
            font=ctk.CTkFont(size=15),
            text_color="#1D1D1D",
            wraplength=500,
            justify="left"
        )
        text_label.pack(side="left")

    results_heading.pack(pady=(20, 5))
    result_frame.pack(pady=(15, 10), padx=10, fill="both", expand=False)
    chart_frame.pack(pady=(10, 10), padx=10, fill="both", expand=False)
    show_bar_chart(counts)
    loading_label.pack_forget()

def predict_sentiments():
    threading.Thread(target=simulate_prediction).start()

# --- Button Frame ---
button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
button_frame.pack(side="bottom", pady=20)

# Add More button
plus_button = ctk.CTkButton(
    button_frame,
    text="Add More",
    image=plus_icon,
    hover_color="#e3e3e3",
    compound="left",
    command=add_input_field,
    width=100,
    height=32,
    font=ctk.CTkFont(size=14),
    border_color="#3d6282",
    border_width=1.15
)
plus_button.pack(side="left", padx=12)

# Predict button
predict_button = ctk.CTkButton(
    button_frame,
    fg_color="#009688",
    text="Predict",
    text_color="#ffffff",
    hover_color="#00796b",
    image=predict_icon,
    compound="left",
    command=predict_sentiments,
    width=115,
    height=33,
    font=ctk.CTkFont(size=15),
    border_color="#7b9a72",
    border_width=1.15
)
predict_button.pack(side="left", padx=12)

# Clear All button
clear_button = ctk.CTkButton(
    button_frame,
    text="Clear All",
    fg_color="#e57373",
    hover_color="#d9534f",
    text_color="#ffffff",
    command=clear_all,
    width=100,
    height=32,
    font=ctk.CTkFont(size=14),
    border_width=0
)
clear_button.pack(side="left", padx=12)

app.mainloop()
