#!/usr/bin/env python3
"""
Cartas de aceptación de prácticas (formato F-GC-015) -> CSV con el formato de tu directorio. 100% local: sin APIs ni llaves.

Motor OCR: RapidOCR v3 (ONNX, modelos incluidos en el paquete: no descarga nada). CPU, o GPU NVIDIA con --gpu.

Idea central: no se "entiende" la hoja, se aprovecha que la plantilla es fija.
  * Bloque de firma:  [nombre] / [puesto] / [empresa] / "Teléfono: ..."
  * Frase del cuerpo: "supervisadas por <nombre>, quien ocupa el puesto de <puesto>"  (verificación cruzada)
  * Membrete (arriba): dirección, correo, teléfonos  (también valida el nombre de la empresa)
  * Se hacen varias lecturas de la misma foto; un teléfono que sale en 2+ lecturas es confiable.
  * Los datos del alumno (nombre, ID) se IGNORAN a propósito.

Memoria de empresas (da consistencia y evita errores de escritura):
  * Si existe directorio_actual.csv (tu hoja exportada) se usa como base: sus empresas ya están bien escritas.
  * Cada empresa nueva que se lee se guarda en empresas.json, para las hojas siguientes.
  * Con cada hoja se calcula un % de coincidencia con las empresas guardadas:
        nombre 40 + misma calle y número 30 + teléfono en común 20 + mismo dominio de correo 10
    Con 70% o más (o 3 señales distintas) se reutilizan Empresa, Descripción, Ubicación y Teléfono/Correo
    guardados; de la hoja solo se toman Contacto y Puesto. Entre 40% y 69% solo se avisa en revision.txt.
  * Para empezar de cero borra empresas.json.

Instalar:
  pip install rapidocr onnxruntime opencv-python-headless rapidfuzz numpy
  GPU NVIDIA:  pip uninstall -y onnxruntime && pip install onnxruntime-gpu

Uso:
  Pon las fotos (.jpg/.jpeg/.png) en la carpeta inbox/ y corre:
  python directorio_local.py            # procesa inbox/
  python directorio_local.py --gpu      # usa la GPU
  python directorio_local.py --debug    # además imprime lo que leyó el OCR

Salida (carpeta salida/):
  para_pegar.csv  una fila por hoja, con EXACTAMENTE las 6 columnas de tu hoja: copia y pega
  revision.txt    (texto, NO es para pegar) por cada foto: confianza y notas de lo que hay que checar
Las fotos leídas pasan a procesadas/ y las que no se pudieron abrir a errores/.
"""
import csv
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz, process
from rapidocr import RapidOCR

# ------------------------------- CONFIG -------------------------------
BASE = Path(__file__).parent
INBOX, DONE, FAILED, OUT = (BASE / d for d in ("inbox", "procesadas", "errores", "salida"))
DIRECTORIO_CSV = BASE / "directorio_actual.csv"   # tu hoja exportada: base de la memoria de empresas
MEMORY_JSON = BASE / "empresas.json"              # empresas aprendidas de las hojas ya procesadas
REUSE_SCORE = 70   # % de coincidencia para reutilizar los datos guardados de una empresa (o 3 señales distintas)
HINT_SCORE = 40    # a partir de aquí solo se avisa en revision.txt: "posible misma empresa"
LONG_SIDE = 3000   # las fotos de celular se reducen a este lado largo (más grande no mejora el OCR)
MIN_SCORE = 0.50   # confianza mínima de RapidOCR para aceptar un fragmento de texto
HEAD6 = ["Empresa", "Descripción", "Ubicación", "Teléfono/Correo", "Contacto", "Puesto"]   # = tu hoja de Google
TITLES = r"(?:ing|lic|arq|dr|dra|mtro|mtra|m\.?c|m\.?i|m\.?b\.?a|c\.?p|t\.?s\.?u|q\.?f\.?b)"
DEBUG = "--debug" in sys.argv
USE_GPU = "--gpu" in sys.argv

# Anclas tolerantes a errores de OCR: "Teléfono", "Telefono", "Teiéfono", "Te1efono"...
TEL_ANCHOR = re.compile(r"\bt[eé]?[l1i][eé]?[fp]onos?\b", re.I)   # "Teléfono" y "Teléfonos"
NOMBRE_FIRMA = re.compile(r"nombre\s+y\s+firma", re.I)
WATERMARK = re.compile(r"^\W*(vigente|autorizado|vigente\s+autorizado)\W*$", re.I)
CITY_RE = re.compile(
    r"\b(?:tamp|altamira|madero|mante|reynosa|monterrey|m[eé]xico|cdmx)"
    r"|\b(?:nl|n\.\s?l\.|qro|jal)(?!\w)", re.I)


# ------------------------------- OCR (RapidOCR) -------------------------------
_engine = None


def engine() -> RapidOCR:
    global _engine
    if _engine is None:
        if USE_GPU:
            _engine = RapidOCR(params={"EngineConfig.onnxruntime.use_cuda": True})
        else:
            _engine = RapidOCR()
    return _engine


def load_resized(path: Path) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError("No pude abrir la imagen")
    s = LONG_SIDE / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im


def flatten(gray: np.ndarray) -> np.ndarray:
    """Quita sombras: divide entre el fondo estimado."""
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((25, 25), np.uint8)), 51)
    return cv2.divide(gray, bg, scale=255)


def boxes_to_lines(boxes, txts, scores) -> list[str]:
    """RapidOCR devuelve fragmentos con caja; los agrupo en renglones (por altura) y ordeno por x.
    Descarta fragmentos de baja confianza y la marca de agua diagonal 'VIGENTE AUTORIZADO'."""
    if boxes is None or txts is None:
        return []
    items = []
    for box, text, score in zip(boxes, txts, scores):
        text = str(text).strip()
        if not text or float(score) < MIN_SCORE:
            continue
        pts = np.array(box, dtype=float)
        dx, dy = pts[1] - pts[0]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if 20 < angle < 160 and len(text) > 3:   # texto diagonal = marca de agua
            continue
        if WATERMARK.match(text):
            continue
        h = max(float(np.linalg.norm(pts[3] - pts[0])), 1.0)
        items.append({"cy": pts[:, 1].mean(), "x": pts[:, 0].min(), "x1": pts[:, 0].max(), "h": h, "t": text})
    items.sort(key=lambda i: i["cy"])
    rows: list[dict] = []
    for it in items:
        if rows and abs(it["cy"] - rows[-1]["cy"]) < 0.5 * max(it["h"], rows[-1]["h"]):
            r = rows[-1]
            r["items"].append(it)
            r["cy"] = float(np.mean([i["cy"] for i in r["items"]]))
            r["h"] = max(r["h"], it["h"])
        else:
            rows.append({"cy": it["cy"], "h": it["h"], "items": [it]})
    out = []
    for r in rows:
        parts, prev = [], None
        for i in sorted(r["items"], key=lambda i: i["x"]):
            if prev is not None:   # hueco grande = otra columna/bloque: se separa con " | "
                parts.append(" | " if i["x"] - prev["x1"] > 1.5 * r["h"] else " ")
            parts.append(i["t"])
            prev = i
        out.append("".join(parts))
    return out


def ocr_text(img: np.ndarray) -> str:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res = engine()(img, use_cls=False)
    return "\n".join(boxes_to_lines(res.boxes, res.txts, res.scores))


def ocr_passes(path: Path) -> dict[str, str]:
    """Varias lecturas de la misma hoja. Cada una falla en lugares distintos."""
    im = load_resized(path)
    flat = flatten(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    h = flat.shape[0]
    return {
        "plano": ocr_text(flat),                        # hoja completa sin sombras
        "crudo": ocr_text(im),                          # hoja completa tal cual
        "membrete": ocr_text(flat[: int(h * 0.30)]),    # encabezado a más resolución
        "firma": ocr_text(flat[int(h * 0.45):]),        # mitad inferior (firma + teléfonos)
    }


# ------------------------------ PARSEO ------------------------------
# Cualquier racha de 10 dígitos con separadores libres (espacios, guiones, paréntesis),
# más extensión opcional. Así entiende "833 213 3416", "(833) 213-34-16", "833 213 34 16", etc.
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?52[ \t.-]*)?(\(?\d[\d \t().-]{8,16}\d)(?!\d)"
    r"(?:[ \t]*(?:ext(?:ensi[oó]n)?\.?)[ \t]*(\d{1,6}))?", re.I)
# "(833) 241-BENE (2363)": las letras son el teclado telefónico y los dígitos entre paréntesis las repiten
KEYPAD_RE = re.compile(r"(\(?\d{3}\)?[ \t.-]*\d{3}[ \t.-]*)[A-Za-z]{4}[ \t]*\(?(\d{4})\)?")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def phones_in(text: str) -> dict[str, str]:
    """{10 dígitos: teléfono formateado}. Si un mismo número sale con y sin extensión, gana el más largo."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = KEYPAD_RE.sub(r"\1\2", line)
        for m in PHONE_RE.finditer(line):
            d = re.sub(r"\D", "", m.group(1))
            if len(d) != 10:
                continue
            s = f"{d[:3]} {d[3:6]} {d[6:8]} {d[8:]}"
            if m.group(2):
                s += f" ext. {m.group(2)}"
            if len(s) > len(out.get(d, "")):
                out[d] = s
    return out


def emails_in(text: str) -> set[str]:
    return {e.lower().strip(".") for e in EMAIL_RE.findall(text.replace(" @", "@").replace("@ ", "@"))}


def lines_of(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def clean_company(name: str) -> str:
    s = strip_accents(name).upper()
    s = re.sub(r"[-/&]", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(S ?DE ?R ?L|S ?R ?L|S ?A ?P ?I|S ?A|S ?C|A ?C)\b( DE C ?V)?|\bDE C ?V\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def person_key(name: str) -> str:
    s = strip_accents(name).lower()
    s = re.sub(rf"\b{TITLES}\b\.?", " ", s)
    return re.sub(r"[^a-z\s]", "", s).strip()


def company_in_header(empresa: str, header_text: str) -> int:
    e, h = clean_company(empresa), clean_company(header_text)
    return int(fuzz.partial_ratio(e, h)) if e and h else 0


# Texto que NO es parte del bloque de firma: sellos, folios, "Nombre y firma", fechas de recibido...
NOISE = re.compile(
    r"^\W*inst\w{4,8}\s+de\W*$|estudios\s+superiores|^\W*de\s+tamaulipas\W*$|recibido|servicios\s+escolares"
    r"|nombre\s+y\s+firma|f-?gc|\brev\.?\s*[\d.o]|^\W*\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)", re.I)


def first_seg(line: str) -> str:
    """Si el renglón trae varios bloques separados por ' | ' (p. ej. firma + sello), me quedo con el de la izquierda."""
    return line.split(" | ")[0].strip()


def best_seg(line: str) -> str:
    """Primer bloque del renglón que sea texto real: ni sello/folio (NOISE) ni basura de logo (sin 3 letras seguidas)."""
    for seg in line.split(" | "):
        seg = seg.strip()
        if re.search(r"[A-Za-zÁ-úÑñ]{3,}", seg) and not NOISE.search(seg):
            return seg
    return ""


def phone_only(line: str) -> bool:
    """Renglón que es solo un teléfono (opcionalmente con 'Tel:' o 'Ext.')."""
    seg = first_seg(line)
    m = PHONE_RE.search(seg)
    if not m:
        return False
    rest = re.sub(r"(?i)\btel[eé]?[a-z]*\b|[:.\s\-]", "", seg[:m.start()] + seg[m.end():])
    return len(rest) <= 3


def valid_person(name: str | None) -> bool:
    """¿Parece nombre de persona? 2-6 palabras, solo letras (con título opcional)."""
    if not name:
        return False
    n = re.sub(rf"^{TITLES}\b\.?\s*", "", name.strip(), flags=re.I)
    words = n.split()
    return (2 <= len(words) <= 6 and bool(re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ.' ]+", n))
            and not any(w.startswith("'") or w.endswith("'") for w in words))


def parse_signature_block(lines: list[str], header_text: str) -> dict | None:
    """El bloque de firma son 3 renglones (nombre / puesto / empresa) seguidos del teléfono.
    Ancla: la palabra 'Teléfono' o, si la carta no la trae, el primer renglón que es solo un teléfono en la
    mitad inferior de la hoja. Si la empresa ocupa 2 renglones se detecta comparando contra el membrete."""
    firsts = [best_seg(l) for l in lines]
    idx = None
    for i in range(len(firsts) - 1, -1, -1):          # el último 'Teléfono(s)': el del bloque de firma
        if TEL_ANCHOR.search(firsts[i]):
            idx = i
            break
    if idx is None:
        idx = next((i for i in range(len(lines) // 2, len(lines)) if phone_only(lines[i])), None)
    before = [l for l in firsts[:idx] if len(l) > 2 and not NOISE.search(l)] if idx is not None else []
    if len(before) >= 3:
        nombre, puesto, empresa = before[-3:]
        score = company_in_header(empresa, header_text)
        if len(before) >= 4:
            empresa_b = before[-2] + " " + before[-1]
            score_b = company_in_header(empresa_b, header_text)
            if score_b >= 75 and score_b > score + 15:
                nombre, puesto, empresa, score = before[-4], before[-3], empresa_b, score_b
        after = [first_seg(lines[idx])]
        for l in lines[idx + 1: idx + 4]:
            if phone_only(l) or re.match(r"\s*ext", l, re.I):
                after.append(first_seg(l))
            else:
                break
        return {"nombre": nombre, "puesto": puesto, "empresa": empresa,
                "empresa_score": score, "tel_text": "\n".join(after)}

    # Sin ancla de teléfono: los renglones que siguen a "Nombre y firma" son nombre / puesto / empresa
    lab = next((i for i in range(len(firsts) - 1, -1, -1) if NOMBRE_FIRMA.search(firsts[i])), None)
    if lab is None:
        return None
    block = [l for l in firsts[lab + 1:] if len(l) > 2 and not NOISE.search(l)]
    block = [l for l in block if not TEL_ANCHOR.search(l)]
    if len(block) < 3:
        return None
    nombre, puesto, empresa = block[:3]
    score = company_in_header(empresa, header_text)
    tels = [first_seg(l) for l in lines[lab + 1: lab + 8] if phone_only(l)]
    return {"nombre": nombre, "puesto": puesto, "empresa": empresa,
            "empresa_score": score, "tel_text": "\n".join(tels)}


def parse_body_supervisor(lines: list[str]) -> tuple[str, str] | None:
    """Supervisor del alumno, según el cuerpo de la carta. Dos formas:
       a) con rayas para llenar: 'quien ocupa el puesto de' y en los 2 renglones siguientes nombre y puesto;
       b) texto corrido: 'supervisadas por <nombre>, quien ocupa el puesto de <puesto>.'"""
    for i, l in enumerate(lines):
        if re.search(r"quien\s+ocupa\s+el\s+puesto\s+de\s*$", first_seg(l) if " | " not in l else l.split(" | ")[-1], re.I):
            cand = [re.sub(r"[_|]+", " ", first_seg(x)).strip(" .,") for x in lines[i + 1: i + 3]]
            if len(cand) == 2 and valid_person(cand[0]) and cand[1]:
                return cand[0], cand[1]
    flat = re.sub(r"[_|]+", " ", re.sub(r"\s+", " ", " ".join(lines)))
    m = re.search(r"supervisadas\s+por\s+(.{5,80}?),?\s+quien\s+ocupa\s+el\s+puesto\s+de\s+(.{5,80}?)\.", flat, re.I)
    if m and valid_person(m.group(1).strip()):
        return m.group(1).strip(), m.group(2).strip()
    return None


def parse_address(lines: list[str]) -> str | None:
    """Segmento con código postal de 5 dígitos junto a una ciudad conocida; se prefiere el que empieza con calle."""
    segs = [seg.strip() for l in lines for seg in l.split(" | ")]
    cands = [l for l in segs if re.search(r"(?<!\d)\d{5}(?!\d)", l) and CITY_RE.search(l)
             and not re.search(r"ID\s*\d{5}", l)]
    street = [l for l in cands if re.match(r"(av|ave|avenida|calle|blvd|boulevard|carr|carretera|priv)", l, re.I)]
    pick = (street or cands or [None])[0]
    if pick:  # recorta texto ajeno pegado después de "CP Ciudad, Estado."
        m = re.match(r"(.*?(?<!\d)\d{5}(?!\d)(?:\s+[A-Za-zÁ-úÑñ.]+){0,2}(?:,\s*[A-Za-zÁ-úÑñ.]+)?)", pick)
        if m:
            pick = m.group(1)
    return pick


def format_address(a: str | None) -> str | None:
    if not a:
        return a
    a = a.strip().rstrip(",")
    if not re.search(r"C\.?\s?P\.?\s*\d{5}", a, re.I):
        a = re.sub(r"(?<!\d)(\d{5})(?!\d)", r"C.P. \1", a, count=1)
    return re.sub(r"\bTamps\.?$|\bTam$", "Tam.", a, flags=re.I)


# ------------------------- MAYÚSCULAS / MINÚSCULAS -------------------------
SMALL = {"de", "del", "la", "las", "los", "el", "y", "e", "en", "van", "von", "da", "di", "do", "dos", "du", "le"}
JOB_WORDS = ["director", "directora", "subdirector", "subdirectora", "gerente", "subgerente", "coordinador",
             "coordinadora", "supervisor", "supervisora", "ingeniero", "ingeniera", "administrador", "administradora",
             "responsable", "encargado", "encargada", "representante", "presidente", "presidenta", "propietario",
             "propietaria", "fundador", "fundadora", "titular", "analista", "especialista", "técnico", "técnica",
             "licenciado", "licenciada", "rector", "rectora", "residente", "superintendente", "desarrollador",
             "desarrolladora", "contador", "contadora", "auxiliar", "asistente", "líder"]
_JOB_KEYS = [strip_accents(j) for j in JOB_WORDS]


def cap_word(w: str) -> str:
    if re.fullmatch(r"(?:[A-Za-zÁ-úÑñ]\.){2,}", w):          # M.I.  C.P.  S.A.
        return w.upper()
    return "-".join(p[:1].upper() + p[1:].lower() for p in w.split("-"))


def title_text(s: str) -> str:
    """'ING. FROYLAN ALONSO GONZÁLEZ' -> 'Ing. Froylan Alonso González' (de/la/del en minúscula)."""
    out = []
    for i, w in enumerate(s.split()):
        out.append(w.lower() if i > 0 and w.lower() in SMALL else cap_word(w))
    return " ".join(out)


def mostly_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.7


def sentence_case(s: str) -> str:
    """'Jefe de Departamento de Comunicaciones' -> 'Jefe de departamento de comunicaciones'.
    Si el texto viene mezclado se respetan las siglas (TI, RH, IMSS); si viene todo en mayúsculas no se puede distinguir."""
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    all_up = s.isupper()
    words = [w if (not all_up and re.fullmatch(r"[A-ZÁÉÍÓÚÑ]{2,5}[.,;:)]?", w)) else w.lower() for w in s.split(" ")]
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def fix_puesto(p: str) -> str:
    """Corrige errores típicos del OCR en la primera palabra del puesto ('Difector' -> 'Director')."""
    words = p.split()
    if words and len(words[0]) >= 6 and strip_accents(words[0]).lower() not in _JOB_KEYS:
        m = process.extractOne(strip_accents(words[0]).lower(), _JOB_KEYS, scorer=fuzz.ratio)
        if m and m[1] >= 85:
            words[0] = JOB_WORDS[m[2]]
    return " ".join(words)


def trim_company(emp: str, header_lines: list[str]) -> str:
    """Si a la empresa se le pegó texto del sello ('Beneficencia Española INTRO ESPANOL OF TAMPICO'),
    la recorta a la parte que sí aparece como renglón/bloque del membrete."""
    ws = emp.split()
    if len(ws) < 3:
        return emp
    pieces = [clean_company(x) for l in header_lines for seg in l.split(" | ") for x in seg.split(",")]
    pieces = [x for x in pieces if x]

    def sc(t: str) -> int:
        c = clean_company(t)
        return int(max((fuzz.ratio(c, h) for h in pieces), default=0)) if c else 0

    full = sc(emp)
    best_n, best_v = 0, -1
    for n in range(1, len(ws)):
        v = sc(" ".join(ws[:n]))
        if v >= best_v:
            best_n, best_v = n, v
    return " ".join(ws[:best_n]) if best_v >= 90 and best_v > full + 10 else emp


def strip_company_from_address(addr: str | None, empresa: str | None) -> str | None:
    """'Manitas Para To2, Wisconsin 111, ...' -> 'Wisconsin 111, ...' (la carta a veces antepone la empresa)."""
    if addr and empresa and "," in addr:
        head, tail = addr.split(",", 1)
        if fuzz.token_set_ratio(clean_company(head), clean_company(empresa)) >= 85:
            return tail.strip()
    return addr


def extract(passes: dict[str, str]) -> dict:
    notes, conf_flags = [], []
    L = {k: lines_of(v) for k, v in passes.items()}
    header_text = " ".join(L["membrete"] + L["plano"][:15])
    order = ("plano", "firma", "crudo")

    # --- bloque de firma: se lee en cada pasada y gana el nombre que más se repite ---
    cands = [c for k in order if (c := parse_signature_block(L[k], header_text))]
    valid = [c for c in cands if valid_person(c["nombre"])]
    if valid:
        top = Counter(person_key(c["nombre"]) for c in valid).most_common(1)[0][0]
        sig = next(c for c in valid if person_key(c["nombre"]) == top)
    else:
        sig = cands[0] if cands else None
    body = next((b for k in ("plano", "crudo", "firma") if (b := parse_body_supervisor(L[k]))), None)

    nombre = puesto = empresa = None
    info = []
    if sig:
        nombre, puesto = sig["nombre"], sig["puesto"]
        empresa = trim_company(sig["empresa"], L["membrete"] + L["plano"][:15])
        if empresa != sig["empresa"]:
            sig["empresa_score"] = company_in_header(empresa, header_text)
        if not valid_person(nombre):
            notes.append(f"El contacto '{nombre}' no parece un nombre de persona (verificar)")
        if body:
            if fuzz.token_set_ratio(person_key(nombre), person_key(body[0])) >= 85:
                t = re.match(rf"({TITLES})\.?\s+", body[0], re.I)   # el título (Ing., Lic.) sale del cuerpo; el nombre, de la firma
                if t and not re.match(rf"{TITLES}\b", nombre, re.I):
                    nombre = t.group(0).strip() + " " + nombre
            else:  # quien firma y quien supervisa al alumno pueden ser personas distintas: solo se informa
                info.append(f"Supervisor del alumno (según el cuerpo de la carta): {body[0]} - {body[1]}")
        if sig["empresa_score"] < 60:
            notes.append(f"La empresa '{empresa}' no aparece claro en el membrete (verificar nombre)")
    elif body:
        nombre, puesto = body
        conf_flags.append("sin bloque de firma")
    else:
        conf_flags.append("no se halló contacto")

    if not empresa:
        conf_flags.append("sin empresa")

    # --- teléfonos: votación entre lecturas, contando por los 10 dígitos ---
    votes: Counter = Counter()
    best: dict[str, str] = {}
    for txt in passes.values():
        for key, s in phones_in(txt).items():
            votes[key] += 1
            if len(s) > len(best.get(key, "")):
                best[key] = s
    sig_phones = phones_in(sig["tel_text"]) if sig else {}
    for key, s in sig_phones.items():
        if len(s) > len(best.get(key, "")):
            best[key] = s
    keep = list(sig_phones) + [k for k, n in votes.items() if n >= 2 and k not in sig_phones]
    phones = [best[k] for k in keep]
    weak = [best[k] for k, n in votes.items() if n == 1 and k not in keep]
    if weak:
        notes.append("Teléfono visto en 1 sola lectura (verificar): " + ", ".join(weak))
    for k in sig_phones:
        if votes[k] < 2:
            notes.append(f"Teléfono con poca certeza: {best[k]}")

    if sig and not sig_phones:
        notes.append("La firma no trae teléfono (revisa si lo pusieron en otra parte de la carta)")

    emails: Counter = Counter()
    for txt in passes.values():
        for e in emails_in(txt):
            emails[e] += 1
    weak_mail = sorted(e for e, n in emails.items() if n == 1)
    if weak_mail and any(n >= 2 for n in emails.values()):   # hay uno confiable y otro leído una vez: probable error de OCR
        notes.append("Correo visto en 1 sola lectura (probable error de OCR, verificar): " + ", ".join(weak_mail))
        emails = Counter({e: n for e, n in emails.items() if n >= 2})

    ubic = None
    for k in ("membrete", "plano", "crudo"):
        ubic = parse_address(L[k])
        if ubic:
            break
    ubic = strip_company_from_address(ubic, empresa)
    if ubic and mostly_upper(ubic):
        ubic = title_text(ubic)
    ubic = format_address(ubic)
    if not ubic:
        notes.append("No se leyó la dirección")

    if nombre:   # "Ing.Alma" -> "Ing. Alma"; todo en Mayúscula Inicial
        nombre = title_text(re.sub(r"(?<=\.)(?=[A-Za-zÁ-úÑñ]{2})", " ", nombre))
    if puesto:
        puesto = sentence_case(fix_puesto(puesto))

    nivel = "alta" if not conf_flags and not notes else ("media" if len(conf_flags) + len(notes) <= 2 else "baja")
    if "sin bloque de firma" in conf_flags or "no se halló contacto" in conf_flags:
        nivel = "baja"
    return {
        "empresa": empresa, "ubicacion": ubic, "telefonos": phones,
        "correos": sorted(emails),
        "nombre": nombre, "puesto": puesto, "confianza": nivel, "notas": notes + conf_flags, "info": info,
    }


# ------------------------- MEMORIA DE EMPRESAS -------------------------
GENERIC_MAIL = {"gmail.com", "hotmail.com", "hotmail.es", "outlook.com", "outlook.es", "yahoo.com", "yahoo.com.mx",
                "live.com", "live.com.mx", "icloud.com", "msn.com", "prodigy.net.mx"}


def entry_from_cells(cells: list[str], origen: str) -> dict:
    c = (list(cells) + [""] * 6)[:6]
    return {"empresa": c[0].strip(), "descripcion": c[1].strip() or "N/A", "ubicacion": c[2].strip() or "N/A",
            "telcorreo": c[3].strip(), "contactos": [x.strip() for x in c[4].splitlines() if x.strip()],
            "origen": origen, "hojas": 0, "archivos": []}


def load_memory() -> list[dict]:
    memory: list[dict] = []
    if DIRECTORIO_CSV.exists():
        with open(DIRECTORIO_CSV, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        start = next((i for i, r in enumerate(rows) if r and r[0].strip().lower() == "empresa"), None)
        if start is None:
            raise RuntimeError("No encontré la fila de encabezados ('Empresa') en directorio_actual.csv.")
        memory += [entry_from_cells(r, "directorio") for r in rows[start + 1:] if r and r[0].strip()]
    else:
        print("[AVISO] No hay directorio_actual.csv: la memoria solo usará las hojas ya procesadas.", file=sys.stderr)
    if MEMORY_JSON.exists():
        try:
            memory += json.loads(MEMORY_JSON.read_text(encoding="utf-8")).get("empresas", [])
        except (ValueError, OSError):
            print("[AVISO] empresas.json está dañado: se ignora (bórralo para empezar de cero).", file=sys.stderr)
    return memory


def save_learned(memory: list[dict]) -> None:
    data = {"version": 1, "empresas": [e for e in memory if e["origen"] == "hojas"]}
    tmp = MEMORY_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MEMORY_JSON)


def name_variants(n: str) -> list[str]:
    """'ABSORMEX CMPC TISSUE S.A. DE C.V. (SOFTYS)' -> completo, sin paréntesis y el alias entre paréntesis."""
    out = [n]
    base = re.sub(r"\([^)]*\)", " ", n).strip()
    if base and base != n:
        out.append(base)
    out += [x for x in re.findall(r"\(([^)]*)\)", n) if len(clean_company(x)) >= 4]
    return out


def names_match(a: str, b: str) -> bool:
    """Mismo nombre de empresa. Una sola palabra en común ('HOSPITAL', 'GRUPO') NO basta."""
    ca = clean_company(a)
    for v in name_variants(b):
        cb = clean_company(v)
        if not ca or not cb:
            continue
        if fuzz.token_sort_ratio(ca, cb) >= 88:
            return True
        if fuzz.token_set_ratio(ca, cb) == 100 and len(set(ca.split()) & set(cb.split())) >= 2:
            return True
    return False


def street_key(a: str) -> tuple[str, str | None]:
    s = strip_accents(a).upper()
    m = re.search(r"(?<!\d)\d{5}(?!\d)", s)
    street = s[:m.start()] if m else s
    street = re.sub(r"\bC\.?\s?P\.?\s*$", "", street.strip())
    street = re.sub(r"[^\w\s]", " ", street)
    return re.sub(r"\s+", " ", street).strip(), (m.group(0) if m else None)


def addr_match(a: str | None, b: str | None) -> bool:
    """Misma calle y número (y mismo CP si ambos lo traen). Dos direcciones de Tampico no coinciden por la ciudad."""
    if not a or not b or "N/A" in (a.strip().upper(), b.strip().upper()):
        return False
    (ka, ca), (kb, cb) = street_key(a), street_key(b)
    if ca and cb and ca != cb:
        return False
    if not (set(re.findall(r"\d+", ka)) & set(re.findall(r"\d+", kb))):
        return False
    return fuzz.token_set_ratio(ka, kb) >= 85


def mail_domains(emails) -> set[str]:
    return {e.split("@", 1)[1] for e in emails if "@" in e} - GENERIC_MAIL


def score_company(rec: dict, e: dict) -> tuple[int, list[str]]:
    sig, pts = [], 0
    if rec["empresa"] and names_match(rec["empresa"], e["empresa"]):
        sig.append("nombre"); pts += 40
    if addr_match(rec["ubicacion"], e["ubicacion"]):
        sig.append("dirección"); pts += 30
    if {re.sub(r"\D", "", p)[:10] for p in rec["telefonos"]} & set(phones_in(e["telcorreo"])):
        sig.append("teléfono"); pts += 20
    if mail_domains(rec["correos"]) & mail_domains(emails_in(e["telcorreo"])):
        sig.append("correo"); pts += 10
    return pts, sig


def find_match(rec: dict, memory: list[dict]) -> tuple[dict | None, int, list[str]]:
    best, best_pts, best_sig = None, 0, []
    for e in memory:            # el directorio va primero: en empate gana
        pts, sig = score_company(rec, e)
        if pts > best_pts:
            best, best_pts, best_sig = e, pts, sig
    return best, best_pts, best_sig


def reuse_row(rec: dict, e: dict) -> tuple[list[str], list[str]]:
    """Fila con los datos guardados de la empresa + contacto y puesto de ESTA hoja.
    Teléfonos/correos de la carta que no estén guardados se agregan, salvo que sean casi iguales a uno guardado
    (1 dígito de diferencia / correo muy parecido = probable error de OCR)."""
    notes = []
    lines = [l for l in ("" if e["telcorreo"].strip().upper() == "N/A" else e["telcorreo"]).splitlines() if l.strip()]
    have = {}
    for i, l in enumerate(lines):
        for d in phones_in(l):
            have.setdefault(d, i)
    have_mail = {m for l in lines for m in emails_in(l)}
    for ph in rec["telefonos"]:
        d = re.sub(r"\D", "", ph)[:10]
        if d in have:
            i = have[d]
            if "ext" in ph.lower() and "ext" not in lines[i].lower():
                lines[i] = ph
                notes.append(f"Se completó la extensión de un teléfono guardado: {ph}")
        elif any(sum(x != y for x, y in zip(d, h)) <= 1 for h in have):
            notes.append(f"Teléfono de la carta ignorado, casi igual a uno guardado (probable error de OCR): {ph}")
        else:
            lines.append(ph)
            have[d] = len(lines) - 1
            notes.append(f"Teléfono nuevo agregado a la empresa: {ph}")
    for m in rec["correos"]:
        if m in have_mail:
            continue
        if any(fuzz.ratio(m, h) >= 85 for h in have_mail):
            notes.append(f"Correo de la carta ignorado, casi igual a uno guardado: {m}")
        else:
            lines.append(m)
            have_mail.add(m)
            notes.append(f"Correo nuevo agregado a la empresa: {m}")
    e["telcorreo"] = "\n".join(lines)
    if rec["nombre"]:
        if any(fuzz.token_set_ratio(person_key(rec["nombre"]), person_key(c)) >= 90 for c in e["contactos"]):
            notes.append("Ese contacto ya estaba guardado para esta empresa")
        else:
            e["contactos"].append(rec["nombre"])
    row = [e["empresa"], e["descripcion"], e["ubicacion"], e["telcorreo"] or "N/A",
           rec["nombre"] or "N/A", rec["puesto"] or "N/A"]
    return row, notes


def confidence_from(notas: list[str]) -> str:
    if any(n.startswith(("sin bloque", "no se halló")) for n in notas):
        return "baja"
    return "alta" if not notas else ("media" if len(notas) <= 2 else "baja")


# --------------------------------- SALIDA ---------------------------------
def rec_to_row6(rec: dict) -> list[str]:
    """Una fila con EXACTAMENTE el formato de tu hoja: varios teléfonos/correos = un renglón cada uno."""
    return [
        (rec["empresa"] or "N/A").upper(),
        "N/A",  # la carta no trae descripción de la empresa: se llena a mano
        rec["ubicacion"] or "N/A",
        "\n".join(rec["telefonos"] + rec["correos"]) or "N/A",
        rec["nombre"] or "N/A",
        rec["puesto"] or "N/A",
    ]


def main():
    for d in (INBOX, DONE, FAILED, OUT):
        d.mkdir(exist_ok=True)
    files = sorted(p for p in INBOX.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        print(f"[AVISO] No hay imágenes (.jpg/.jpeg/.png) en {INBOX}")
        return
    memory = load_memory()
    out_csv = OUT / "para_pegar.csv"
    nuevo = not out_csv.exists()
    cuenta = Counter()
    with open(out_csv, "a", newline="", encoding="utf-8-sig") as fh, \
            open(OUT / "revision.txt", "a", encoding="utf-8") as rv:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(HEAD6)
        for f in files:
            try:
                passes = ocr_passes(f)
                if DEBUG:
                    for k, v in passes.items():
                        print(f"\n--- {f.name} / {k} ---\n{v}")
                rec = extract(passes)
                entry, pts, sig = find_match(rec, memory)
                reused = entry is not None and (pts >= REUSE_SCORE or len(sig) >= 3)
                notas, extra, nivel, coinc = list(rec["notas"]), [], rec["confianza"], ""
                if reused:
                    origen = "directorio_actual.csv" if entry["origen"] == "directorio" else "hojas anteriores"
                    row6, extra = reuse_row(rec, entry)
                    # lo que la carta dijo de empresa/teléfonos/dirección ya no aplica: se usó lo guardado
                    notas = [n for n in notas if n.startswith(("El contacto", "sin bloque", "no se halló"))]
                    nivel = confidence_from(notas)
                    coinc = f'{pts}% con "{entry["empresa"]}" ({origen}) por {", ".join(sig)}: se reutilizaron sus datos'
                    if rec["empresa"] and clean_company(rec["empresa"]) != clean_company(entry["empresa"]):
                        extra.insert(0, f"La carta decía: {rec['empresa']}")
                    if entry["origen"] == "hojas":
                        entry["hojas"] += 1
                        entry["archivos"].append(f.name)
                        save_learned(memory)
                    cuenta["reconocidas"] += 1
                else:
                    row6 = rec_to_row6(rec)
                    if entry and pts >= HINT_SCORE:
                        coinc = (f'posible misma empresa que "{entry["empresa"]}" ({pts}% por {", ".join(sig)}); '
                                 f"NO se reutilizó (hace falta {REUSE_SCORE}% o 3 señales)")
                    e = entry_from_cells(row6, "hojas")
                    e["hojas"], e["archivos"] = 1, [f.name]
                    memory.append(e)
                    save_learned(memory)
                w.writerow(row6)
                fh.flush()
                rv.write(f"=== {f.name} ===\nEmpresa:   {row6[0]}\nConfianza: {nivel}\n"
                         + (f"Coincidencia: {coinc}\n" if coinc else "")
                         + "".join(f"  - {n}\n" for n in notas + extra)
                         + "".join(f"  (info) {n}\n" for n in rec["info"]) + "\n")
                rv.flush()
                shutil.move(str(f), DONE / f.name)
                cuenta["ok"] += 1
                print(f"[OK] {f.name}: {row6[0]} | confianza {nivel}" + (f" | reconocida {pts}%" if reused else ""))
            except Exception as ex:
                print(f"[ERROR] {f.name}: {ex}", file=sys.stderr)
                shutil.move(str(f), FAILED / f.name)
                cuenta["errores"] += 1
    print(f"\nListo: {cuenta['ok']} hojas en salida/para_pegar.csv ({cuenta['reconocidas']} de empresas ya conocidas), "
          f"{cuenta['errores']} con error (carpeta errores/). Qué revisar: salida/revision.txt")


if __name__ == "__main__":
    main()