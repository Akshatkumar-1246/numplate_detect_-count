# ================= IMPORTS =================

import streamlit as st
import pandas as pd
import cv2
from ultralytics import YOLO
from pytesseract import pytesseract
from datetime import datetime
import re
import tempfile
import time

# ================= PAGE CONFIG =================

st.set_page_config(layout="wide")
st.title("🚗 Smart Garage Vehicle Entry/Exit System")

# ================= TESSERACT =================

pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= OCR =================

def plate_text(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 11, 17, 17)
    _, thresh = cv2.threshold(grey, 120, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    return text.strip()

def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# ================= FILE UPLOAD =================

video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if video_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    model = YOLO("best.pt")

    active_vehicles = {}
    logs = []

    in_count = 0
    out_count = 0
    frame_count = 0
    EXIT_DELAY = 40

    col1, col2 = st.columns([2,1])

    video_placeholder = col1.empty()
    table_placeholder = col2.empty()
    count_placeholder = col2.empty()

    search_query = col2.text_input("🔍 Search Plate")

    # ================= VIDEO LOOP =================

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 Process every 3rd frame (Smooth)
        if frame_count % 3 != 0:
            continue

        results = model.predict(frame, conf=0.3)

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate_img = frame[y1:y2, x1:x2]
                plate_img = cv2.resize(plate_img, (300,100))

                plate_id = clean_plate(plate_text(plate_img))

                if plate_id == "" or len(plate_id) < 6:
                    continue

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time_now = now.strftime("%H:%M:%S")

                # ================= IN =================
                if plate_id not in active_vehicles:
                    active_vehicles[plate_id] = now
                    logs.append([plate_id,"IN",date,time_now,"-"])
                    in_count += 1

                # Draw box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,plate_id,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # ================= AUTO OUT =================
        remove_list = []

        for plate_id, entry_time in active_vehicles.items():

            duration = (datetime.now() - entry_time).seconds

            if duration > 5:   # 5 seconds stay demo

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time_now = now.strftime("%H:%M:%S")

                logs.append([
                    plate_id,
                    "OUT",
                    date,
                    time_now,
                    f"{duration}"
                ])

                out_count += 1
                remove_list.append(plate_id)

        for plate_id in remove_list:
            del active_vehicles[plate_id]

        # ================= DISPLAY VIDEO =================

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # ================= DATAFRAME =================

        df = pd.DataFrame(
            logs,
            columns=["Car Number","Status","Date","Time","Duration"]
        )

        # 🔍 SEARCH FILTER
        if search_query:
            df = df[df["Car Number"].str.contains(search_query.upper())]

        # 🎨 COLOR FORMAT
        def color_status(val):
            if val == "IN":
                return "background-color: #28a745; color:white"
            elif val == "OUT":
                return "background-color: #dc3545; color:white"
            return ""

        styled_df = df.tail(30).style.applymap(
            color_status, subset=["Status"]
        )

        table_placeholder.dataframe(styled_df, height=400)

        # ================= COUNTS =================

        count_placeholder.markdown(f"""
        ### 📊 Live Counts
        🟢 **IN Count:** {in_count}  
        🔴 **OUT Count:** {out_count}  
        🚗 **Inside:** {in_count - out_count}
        """)

        time.sleep(0.03)  # smooth UI

    cap.release()

    # ================= DOWNLOAD BUTTON =================

    excel_path = "garage_logs.xlsx"
    df.to_excel(excel_path,index=False)

    with open(excel_path,"rb") as file:
        st.download_button(
            label="📥 Download Excel",
            data=file,
            file_name="garage_logs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )