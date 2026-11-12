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
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz
from rapidocr import RapidOCR

# ------------------------------- CONFIG -------------------------------
BASE = Path(__file__).parent
INBOX, DONE, FAILED, OUT = (BASE / d for d in ("inbox", "procesadas", "errores", "salida"))
LONG_SIDE = 3000   # las fotos de celular se reducen a este lado largo (más grande no mejora el OCR)
MIN_SCORE = 0.50   # confianza mínima de RapidOCR para aceptar un fragmento de texto
HEAD6 = ["Empresa", "Descripción", "Ubicación", "Teléfono/Correo", "Contacto", "Puesto"]   # = tu hoja de Google
TITLES = r"(?:ing|lic|arq|dr|dra|mtro|mtra|m\.?c|m\.?i|m\.?b\.?a|c\.?p|t\.?s\.?u|q\.?f\.?b)"
DEBUG = "--debug" in sys.argv
USE_GPU = "--gpu" in sys.argv

# Anclas tolerantes a errores de OCR: "Teléfono", "Telefono", "Teiéfono", "Te1efono"...
TEL_ANCHOR = re.compile(r"\bt[eé]?[l1i][eé]?[fp]ono\b", re.I)
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
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def phones_in(text: str) -> dict[str, str]:
    """{10 dígitos: teléfono formateado}. Si un mismo número sale con y sin extensión, gana el más largo."""
    out: dict[str, str] = {}
    for line in text.splitlines():
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
    r"instit|instt|estudios\s+superiores|de\s+tamaulipas|recibido|servicios\s+escolares|nombre\s+y\s+firma|f-?gc"
    r"|\brev\.?\s*[\d.o]|^\W*\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)", re.I)


def first_seg(line: str) -> str:
    """Si el renglón trae varios bloques separados por ' | ' (p. ej. firma + sello), me quedo con el de la izquierda."""
    return line.split(" | ")[0].strip()


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
    return 2 <= len(words) <= 6 and bool(re.fullmatch(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ.' ]+", n))


def parse_signature_block(lines: list[str], header_text: str) -> dict | None:
    """El bloque de firma son 3 renglones (nombre / puesto / empresa) seguidos del teléfono.
    Ancla: la palabra 'Teléfono' o, si la carta no la trae, el primer renglón que es solo un teléfono en la
    mitad inferior de la hoja. Si la empresa ocupa 2 renglones se detecta comparando contra el membrete."""
    firsts = [first_seg(l) for l in lines]
    idx = None
    for i in range(len(firsts) - 1, -1, -1):          # el último 'Teléfono': el del bloque de firma
        if TEL_ANCHOR.search(firsts[i]):
            idx = i
            break
    if idx is None:
        idx = next((i for i in range(len(lines) // 2, len(lines)) if phone_only(lines[i])), None)
    if idx is None:
        return None
    before = [l for l in firsts[:idx] if len(l) > 2 and not NOISE.search(l)]
    if len(before) < 3:
        return None
    nombre, puesto, empresa = before[-3:]
    score = company_in_header(empresa, header_text)
    if len(before) >= 4:
        empresa_b = before[-2] + " " + before[-1]
        score_b = company_in_header(empresa_b, header_text)
        if score_b >= 75 and score_b > score + 15:
            nombre, puesto, empresa, score = before[-4], before[-3], empresa_b, score_b
    after = [firsts[idx]]
    for l in lines[idx + 1: idx + 4]:
        if phone_only(l) or re.match(r"\s*ext", l, re.I):
            after.append(first_seg(l))
        else:
            break
    return {"nombre": nombre, "puesto": puesto, "empresa": empresa,
            "empresa_score": score, "tel_text": "\n".join(after)}


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
    return re.sub(r"\bTamps\.?$|\bTam$", "Tam.", a)


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

    # --- bloque de firma: primera lectura donde salga una persona válida ---
    sig, respaldo = None, None
    for k in order:
        cand = parse_signature_block(L[k], header_text)
        if not cand:
            continue
        if valid_person(cand["nombre"]):
            sig = cand
            break
        respaldo = respaldo or cand
    sig = sig or respaldo
    body = next((b for k in ("plano", "crudo", "firma") if (b := parse_body_supervisor(L[k]))), None)

    nombre = puesto = empresa = None
    info = []
    if sig:
        nombre, puesto, empresa = sig["nombre"], sig["puesto"], sig["empresa"]
        if not valid_person(nombre):
            notes.append(f"El contacto '{nombre}' no parece un nombre de persona (verificar)")
        if body:
            if fuzz.token_set_ratio(person_key(nombre), person_key(body[0])) >= 85:
                if re.match(rf"{TITLES}\b", body[0], re.I):   # conservar el título (Ing., Lic.) si el cuerpo lo trae
                    nombre = body[0]
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
    ubic = format_address(strip_company_from_address(ubic, empresa))
    if not ubic:
        notes.append("No se leyó la dirección")

    nivel = "alta" if not conf_flags and not notes else ("media" if len(conf_flags) + len(notes) <= 2 else "baja")
    if "sin bloque de firma" in conf_flags or "no se halló contacto" in conf_flags:
        nivel = "baja"
    return {
        "empresa": empresa, "ubicacion": ubic, "telefonos": phones,
        "correos": sorted(emails),
        "nombre": nombre, "puesto": puesto, "confianza": nivel, "notas": notes + conf_flags, "info": info,
    }


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
                row6 = rec_to_row6(rec)
                w.writerow(row6)
                fh.flush()
                rv.write(f"=== {f.name} ===\nEmpresa:   {row6[0]}\nConfianza: {rec['confianza']}\n"
                         + "".join(f"  - {n}\n" for n in rec["notas"])
                         + "".join(f"  (info) {n}\n" for n in rec["info"]) + "\n")
                rv.flush()
                shutil.move(str(f), DONE / f.name)
                cuenta["ok"] += 1
                print(f"[OK] {f.name}: {row6[0]} | confianza {rec['confianza']}")
            except Exception as ex:
                print(f"[ERROR] {f.name}: {ex}", file=sys.stderr)
                shutil.move(str(f), FAILED / f.name)
                cuenta["errores"] += 1
    print(f"\nListo: {cuenta['ok']} hojas en salida/para_pegar.csv, {cuenta['errores']} con error (carpeta errores/). "
          f"Qué revisar: salida/revision.txt")


if __name__ == "__main__":
    main()