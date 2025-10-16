
import base64
import cv2
import re
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from uvicorn import run
from ultralytics import YOLO
import easyocr
import difflib
from io import BytesIO
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

model_dir = "best.pt"
model = YOLO(model_dir)
print("Model loaded successfully.")

# โหลด EasyOCR
reader = easyocr.Reader(['th', 'en'])

# รายชื่อจังหวัดทั้งหมด
thai_provinces = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
    "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พังงา", "พัทลุง",
    "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์", "แพร่",
    "พะเยา", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี"
]

def correct_province(text):
    """พยายามจับชื่อจังหวัดที่คล้ายที่สุด"""
    match = difflib.get_close_matches(text, thai_provinces, n=1, cutoff=0.4)
    return match[0] if match else text

    return match[0] if match else text


# ---------------------- ฟังก์ชันปรับภาพ OCR ----------------------
def enhance_for_ocr(img):
    """
    ปรับภาพให้เหมาะกับ OCR ป้ายทะเบียนไทย
    เน้น: ขอบคมแต่ไม่แตก, ทนทุกสภาพแสง, เหมาะกับ EasyOCR/PaddleOCR
    """
    # 1 แปลงภาพเป็น grayscale จากช่อง Y (แสง)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        gray = img.copy()

    # 2 ลด noise แบบ soft (รักษาขอบตัวอักษร)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=40, sigmaSpace=40)

    # 3 Auto gamma ปรับตามความสว่างของภาพ
    mean_val = np.mean(gray)
    gamma = 1.5 if mean_val < 70 else (1.1 if mean_val < 120 else 0.9)
    gray = np.uint8(np.clip((gray / 255.0) ** gamma * 255, 0, 255))

    # 4 Adaptive contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5 Adaptive Threshold แบบ soft — ลด blockSize + เพิ่ม blur ก่อน
    blur_for_th = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur_for_th, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        61, 15
    )

    # 6 Auto invert ถ้าพื้นเข้ม
    white_ratio = np.sum(th == 255) / th.size
    if white_ratio < 0.45:
        th = cv2.bitwise_not(th)

    # 7 Morph close (ปิดช่องว่างตัวอักษร)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 8 median blur นิดนึงให้เส้นเรียบ
    th = cv2.medianBlur(th, 3)

    # 9 Soft sharpen — เน้นขอบแต่ไม่แตก
    blurred = cv2.GaussianBlur(th, (3, 3), 0)
    sharp_strength = 1.25 if np.std(gray) < 60 else 1.15
    sharpened = cv2.addWeighted(th, sharp_strength, blurred, -0.25, 0)

    # 10 Upscale ด้วย Lanczos (เก็บรายละเอียดดีสุด)
    upscale_factor = 2.0 if img.shape[1] < 400 else 1.5
    final = cv2.resize(
        sharpened, None,
        fx=upscale_factor, fy=upscale_factor,
        interpolation=cv2.INTER_LANCZOS4
    )

    # 11 ปรับ contrast สุดท้ายแบบ soft
    final = cv2.convertScaleAbs(final, alpha=1.05, beta=5)
    return gray

def extract_thai_license_plate(text):
    """
    ดึงทะเบียนรถไทย (พร้อมชื่อจังหวัดถ้ามี)
    """
    cleaned = re.sub(r"[^ก-ฮ0-9\s\-\.]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    pattern = (
        r"([0-9]{0,2}\s*[ก-ฮ]{1,3}[\s\-\.]*\d{1,4})"
        r"\s*([ก-ฮ]{2,}(?:จังหวัด)?[^0-9]*)?"
    )

    matches = re.findall(pattern, cleaned)
    results = []
    for plate, province in matches:
        plate = re.sub(r"[\s\-\.]", "", plate)
        province = correct_province(province.strip()) if province else None
        results.append({
            "plate": plate,
            "province": province
        })
    return results

@app.get("/")
async def root():
    return {"message": f"Welcome to the Fashion Model API!"}

@app.post("/predict")
async def predict(image: UploadFile):
    filename = image.filename
    contents = await image.read()

    # อ่านภาพ
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "ไม่สามารถอ่านภาพได้"}

    # ตรวจจับ YOLO
    results = model.predict(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    all_results = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        crop = frame[y1:y2, x1:x2]

        preprocessed = enhance_for_ocr(crop)
        scale = 2.0 if crop.shape[1] < 300 else 1.5
        preprocessed = cv2.resize(preprocessed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        preprocessed = cv2.convertScaleAbs(preprocessed, alpha=1.8, beta=30)

        ocr_result = reader.readtext(preprocessed, detail=1, paragraph=True)
        ocr_texts = [t[1].strip() for t in ocr_result if len(t) >= 2 and t[1].strip()]
        combined_text = re.sub(r"[^ก-ฮ0-9\s]", "", " ".join(ocr_texts))
        print(f"📷 OCR Raw: {combined_text}")

        results_found = extract_thai_license_plate(combined_text)
        if results_found:
            plate_info = results_found[0]
            final_label = plate_info["plate"]
            if plate_info["province"]:
                final_label += f" {plate_info['province']}"
                color = (0, 255, 0)  # เขียว = เจอป้ายและจังหวัด
            else:
                color = (0, 200, 200)  # สีฟ้า = เจอเลขป้าย
        else:
            final_label = "อ่านไม่ออก"
            color = (0, 0, 255)  # แดง = อ่านไม่ออก
        print(f"✅ Final: {final_label}")
        all_results.append(final_label)

        # วาดกรอบบนภาพ
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # แปลงภาพที่ตีกรอบแล้วเป็น base64
    _, buffer = cv2.imencode('.jpg', frame)
    # img_base64 = base64.b64encode(buffer).decode('utf-8')

    license_plate = ''.join(all_results)

    return {
        "filename": filename,
        "plates_detected": license_plate,
        # "image_base64": img_base64,
        "message": "Processing complete."
    }


import requests

def get_def_headers():
    return {
        "X-API-KEY": "123456"
    }
def get_base_api():
    return "localhost"

def post_json_api(url, data, headers=None):
    """POST API with JSON body"""
    response = requests.post(url, json=data, headers=headers)
    return response.json()

def post_image_api(url, image_path, extra_data=None, headers=None):
    """POST API with multipart/form-data and image upload"""
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = extra_data if extra_data else {}
        response = requests.post(url, files=files, data=data, headers=headers)
    return response.json()
