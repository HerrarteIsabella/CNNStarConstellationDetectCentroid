import cv2
import numpy as np
import requests
import time
import json
import sys
import os
from star_detection_centroiding import *
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_detection_centroiding import run_baseline

API_KEY = "wnmzxqdgrsmercxg"
API_URL = "http://nova.astrometry.net/api"

def detect_stars(image_path):
    img_color = cv2.imread(image_path)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype('float32')
    
    tile_h, tile_w = 480, 640
    overlap = 64
    img_h, img_w = img.shape
    centroid_est = []

    for row in range(0, img_h, tile_h - overlap):
        for col in range(0, img_w, tile_w - overlap):
            r_end = min(row + tile_h, img_h)
            c_end = min(col + tile_w, img_w)
            patch = img[row:r_end, col:c_end]
            ph, pw = patch.shape
            if ph < 64 or pw < 64:
                continue
            margin = overlap // 2
            valid_r0 = margin if row > 0 else 0
            valid_c0 = margin if col > 0 else 0
            valid_r1 = ph - margin if r_end < img_h else ph
            valid_c1 = pw - margin if c_end < img_w else pw
            patch_centroids = run_baseline(patch.copy())
            for c in patch_centroids:
                lx, ly = c[0], c[1]
                if valid_c0 <= lx < valid_c1 and valid_r0 <= ly < valid_r1:
                    centroid_est.append([lx + col, ly + row])

    filtered = []
    for c in centroid_est:
        if all(np.sqrt((c[0]-f[0])**2 + (c[1]-f[1])**2) > 10 for f in filtered):
            filtered.append(c)

    print(f"Estrellas detectadas: {len(filtered)}")
    return filtered, img_color, img_h, img_w

def login():
    r = requests.post(f"{API_URL}/login", data={"request-json": json.dumps({"apikey": API_KEY})})
    session = r.json()["session"]
    print(f"Sesión iniciada: {session}")
    return session

def upload_image(session, image_path):
    print("Subiendo imagen a astrometry.net...")
    with open(image_path, "rb") as f:
        r = requests.post(f"{API_URL}/upload", files={"file": f}, data={
            "request-json": json.dumps({
                "session": session,
                "publicly_visible": "n",
                "allow_modifications": "n",
                "scale_units": "degwidth",
                "scale_lower": 8,   # FOV mínimo en grados
                "scale_upper": 12,  # FOV máximo en grados
            })
        })
    result = r.json()
    subid = result["subid"]
    print(f"Imagen subida, submission ID: {subid}")
    return subid

def wait_for_result(subid, timeout=120):
    print("Esperando resultado del plate solving", end="")
    for _ in range(timeout):
        time.sleep(1)
        r = requests.get(f"{API_URL}/submissions/{subid}")
        data = r.json()
        jobs = data.get("jobs", [])
        if jobs and jobs[0] is not None:
            job_id = jobs[0]
            r2 = requests.get(f"{API_URL}/jobs/{job_id}")
            status = r2.json().get("status")
            print(f"\nEstado: {status}")
            if status == "success":
                return job_id
            elif status == "failure":
                print("El plate solving falló.")
                return None
        print(".", end="", flush=True)
    print("\nTimeout esperando resultado.")
    return None

def get_results(job_id, centroids, img_w, img_h):
    # obtener información de calibración WCS
    r = requests.get(f"{API_URL}/jobs/{job_id}/calibration")
    cal = r.json()
    print("\n=== RESULTADO DEL PLATE SOLVING ===")
    print(f"  RA centro:  {cal['ra']:.4f}°")
    print(f"  Dec centro: {cal['dec']:.4f}°")
    print(f"  Orientación: {cal['orientation']:.2f}°")
    print(f"  Escala: {cal['pixscale']:.4f} arcsec/pixel")
    print(f"  Radio FOV: {cal['radius']:.4f}°")

    # obtener objetos identificados
    r2 = requests.get(f"{API_URL}/jobs/{job_id}/objects_in_field")
    objects = r2.json().get("objects_in_field", [])
    print(f"\n OBJETOS EN EL CAMPO ")
    for obj in objects:
        print(f"  {obj}")

    # convertir centroides a RA/Dec usando WCS simple
    print(f"\n COORDENADAS PARA TELESCOPIO ")
    ra_center = cal['ra']
    dec_center = cal['dec']
    pixscale = cal['pixscale'] / 3600.0  # arcsec -> grados por pixel
    orientation = cal['orientation']

    for i, c in enumerate(centroids[:20]):  # primeras 20 estrellas
        dx = c[0] - img_w / 2
        dy = -(c[1] - img_h / 2)  # invertir Y
        # rotación
        angle_rad = np.radians(orientation)
        dx_rot = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
        dy_rot = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        ra = ra_center + dx_rot * pixscale / np.cos(np.radians(dec_center))
        dec = dec_center + dy_rot * pixscale
        print(f"  Estrella {i:3d}: RA={ra:.4f}°  Dec={dec:+.4f}°  (x={c[0]:.0f}, y={c[1]:.0f})")

    return cal, objects

def draw_results(img_color, centroids, output_path):
    for i, c in enumerate(centroids):
        cx, cy = int(c[0]), int(c[1])
        cv2.circle(img_color, (cx, cy), 15, (0, 255, 0), 1)
        cv2.putText(img_color, str(i), (cx+5, cy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(output_path, img_color)
    print(f"\nImagen resultado guardada en: {output_path}")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test/Cassiopeia_apr_h0_fov10_atm001.png"
    
    centroids, img_color, img_h, img_w = detect_stars(image_path)
    session = login()
    subid = upload_image(session, image_path)
    job_id = wait_for_result(subid)
    
    if job_id:
        cal, objects = get_results(job_id, centroids, img_w, img_h)
        output = image_path.replace('.png', '_solved.png')
        draw_results(img_color, centroids, output)
    else:
        print("No se pudo resolver la imagen.")