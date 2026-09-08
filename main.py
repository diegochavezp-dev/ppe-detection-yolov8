import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from tkinter import Tk, Button, filedialog
import threading
from tkinter import ttk
from tkinter import Frame, Label
from tkinter import font as tkfont
from sklearn.metrics import precision_score, recall_score, f1_score


# Entrenamiento del modelo
def entrenar_modelo():
    YOLO("yolov8n.pt").train(
        data="data.yaml",
        epochs=40,
        patience=10,
        imgsz=480,
        project="runs",
        name="epp_train",
        exist_ok=True
    )

# Lógica de verificación de EPP (usado tanto en cámara como en imagen)
def verificar_epp(frame, model, log):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    class_names = results[0].names  # {0: 'boots', 1: 'gloves', 2: 'helmet', 3: 'human', 4: 'vest'}

    personas = []
    epp_items = {
        0: [],  # boots
        1: [],  # gloves
        2: [],  # helmet
        4: []   # vest
    }

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        box = [int(x1), int(y1), int(x2), int(y2)]

        if cls == 3: 
            personas.append(box)
        elif cls in epp_items:
            epp_items[cls].append(box)

    for i, person_box in enumerate(personas):
        def cerca(boxes):
            for box in boxes:
                px1, py1, px2, py2 = person_box
                bx1, by1, bx2, by2 = box
                if not (px2 < bx1 or bx2 < px1 or py2 < by1 or by2 < py1):
                    return True
            return False

        tiene_botas = cerca(epp_items[0])
        tiene_guantes = cerca(epp_items[1])
        tiene_casco = cerca(epp_items[2])
        tiene_chaleco = cerca(epp_items[4])

        cumple = all([tiene_botas, tiene_casco, tiene_chaleco])
        color = (0, 255, 0) if cumple else (0, 0, 255)

        faltantes = []
        if not tiene_casco: faltantes.append("casco")
        if not tiene_chaleco: faltantes.append("chaleco")
        if not tiene_botas: faltantes.append("botas")
        if not tiene_guantes: faltantes.append("guantes")
        estado = "Cumple" if cumple else f"Falta: {', '.join(faltantes)}"

        cv2.rectangle(frame, (person_box[0], person_box[1]), (person_box[2], person_box[3]), color, 2)
        cv2.putText(frame, estado, (person_box[0], person_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not cumple:
            log.append({
                "persona_id": f"P{i+1}",
                "fecha": datetime.now().strftime("%Y-%m-%d"),
                "hora": datetime.now().strftime("%H:%M:%S"),
                "casco": tiene_casco,
                "chaleco": tiene_chaleco,
                "botas": tiene_botas,
                "guantes": tiene_guantes,
                "estado": "No cumple"
            })

    return frame

# Modo cámara en tiempo real
def modo_camara():
    cap = cv2.VideoCapture(0)
    log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = verificar_epp(frame, model, log)
        cv2.imshow("Verificación de EPP - Cámara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if log:
        pd.DataFrame(log).to_csv("reporte_EPP.csv", index=False)
        print("[INFO] Reporte generado: reporte_EPP.csv")
    else:
        print("[INFO] No se detectaron incumplimientos.")

# Modo imagen única
def modo_imagen():
    path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=[("Imagenes", "*.jpg *.png *.jpeg")])
    if not path:
        return

    image = cv2.imread(path)
    log = []
    resultado = verificar_epp(image, model, log)
    cv2.imshow("Verificación de EPP - Imagen", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if log:
        pd.DataFrame(log).to_csv("reporte_EPP.csv", index=False)
        print("[INFO] Reporte generado: reporte_EPP.csv")
    else:
        print("[INFO] No se detectaron incumplimientos.")

# Modo captura de imagen desde cámara
def capturar_y_verificar():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] No se pudo capturar imagen.")
        return

    log = []
    resultado = verificar_epp(frame, model, log)
    cv2.imshow("Verificación de EPP - Captura", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if log:
        pd.DataFrame(log).to_csv("reporte_EPP.csv", index=False)
        print("[INFO] Reporte generado: reporte_EPP.csv")
    else:
        print("[INFO] No se detectaron incumplimientos.")

#Evaluar el modelo en un conjunto de imágenes y etiquetas
def evaluar_modelo_general(model, img_dir, lbl_dir):
    clases = {0: "botas", 1: "guantes", 2: "casco", 3: "persona", 4: "chaleco"}
    y_true_all = []
    y_pred_all = []

    for img_name in os.listdir(img_dir):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            etiquetas = [int(line.strip().split()[0]) for line in f.readlines()]

        image = cv2.imread(image_path)
        results = model(image)
        detections = results[0].boxes.data.cpu().numpy()
        clases_predichas = [int(box[5]) for box in detections]

        for clase_id in clases.keys():
            y_true_all.append(1 if clase_id in etiquetas else 0)
            y_pred_all.append(1 if clase_id in clases_predichas else 0)

    # Calcular métricas generales
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    print("\n===== MÉTRICAS GENERALES DEL MODELO =====")
    print(f"Precisión:   {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")


if __name__ == '__main__':
    #entrenar_modelo()  # Descomentar si deseas entrenar el modelo
    model = YOLO("runs/epp_train/weights/best.pt")

    evaluar_modelo_general(
        model,
        "data/valid/images",
        "data/valid/labels"
    )

    root = Tk()
    root.title("Sistema de Verificación de EPP")
    root.geometry("350x270")
    root.configure(bg="#f0f4f8")  

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Segoe UI', 11), padding=6, background='#1976d2', foreground='white')
    style.map('TButton', background=[('active', '#1565c0')])

    frame = Frame(root, bg="#ffffff", bd=2, relief="groove")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=320, height=220)

    title_font = tkfont.Font(family="Segoe UI", size=15, weight="bold")
    Label(frame, text="Verificación de EPP", bg="#ffffff", fg="#1976d2", font=title_font).pack(pady=(18, 10))

    ttk.Button(frame, text="Verificar desde Cámara", width=28, command=lambda: threading.Thread(target=modo_camara).start()).pack(pady=6)
    ttk.Button(frame, text="Verificar Imagen", width=28, command=modo_imagen).pack(pady=6)
    ttk.Button(frame, text="Capturar Foto y Verificar", width=28, command=capturar_y_verificar).pack(pady=6)

    root.mainloop()
