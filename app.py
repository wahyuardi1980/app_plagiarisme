from flask import Flask, render_template, request
import pdfplumber, docx, os, math, re, time, uuid, requests
from collections import Counter
from werkzeug.utils import secure_filename
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================
# APP CONFIG
# =====================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
ALLOWED_EXT = (".pdf", ".docx", ".txt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# =====================
# STOPWORDS
# =====================
stop_factory = StopWordRemoverFactory()
STOPWORDS = set(stop_factory.get_stop_words())

# =====================
# READ FILE
# =====================
def read_file(path):
    text = ""
    try:
        if path.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + " "
        elif path.endswith(".docx"):
            doc = docx.Document(path)
            for p in doc.paragraphs:
                text += p.text + " "
        elif path.endswith(".txt"):
            with open(path, encoding="utf8") as f:
                text = f.read()
    except:
        pass
    return text.lower()

# =====================
# PREPROCESS
# =====================
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

# =====================
# KEYWORD EXTRACTION
# =====================
def extract_keywords(tokens, top_n=5):
    freq = Counter(tokens)
    return " ".join([w for w, _ in freq.most_common(top_n)])

# =====================
# FETCH API & STORE DATASET
# =====================
def fetch_and_store_journals(query, limit=5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json().get("data", [])

        for paper in data:
            abstract = paper.get("abstract")
            if abstract and len(abstract) > 200:
                fname = f"api_{uuid.uuid4().hex}.txt"
                with open(os.path.join(DATASET_FOLDER, fname), "w", encoding="utf8") as f:
                    f.write(abstract.lower())
    except Exception as e:
        print("API ERROR:", e)

# =====================
# NGRAM
# =====================
def ngrams(tokens, n=3):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# =====================
# RABIN-KARP
# =====================
def rabin_karp(ng1, ng2):
    s1, s2 = set(ng1), set(ng2)
    if not s1 or not s2:
        return 0
    return (len(s1 & s2) / len(s1)) * 100

# =====================
# COSINE SIMILARITY
# =====================
def cosine(t1, t2):
    f1, f2 = Counter(t1), Counter(t2)
    common = set(f1) & set(f2)
    dot = sum(f1[w] * f2[w] for w in common)
    mag1 = math.sqrt(sum(v*v for v in f1.values()))
    mag2 = math.sqrt(sum(v*v for v in f2.values()))
    return 0 if mag1 == 0 or mag2 == 0 else (dot / (mag1 * mag2)) * 100

# =====================
# ROUTE
# =====================
@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    details = []

    if request.method == "POST":
        start_time = time.time()

        file = request.files.get("doc")
        if not file or not file.filename.lower().endswith(ALLOWED_EXT):
            error = "Format file tidak didukung"
            return render_template("index.html", error=error)

        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # ===== DOKUMEN UJI =====
        text_uji = read_file(upload_path)
        tokens_uji = preprocess(text_uji)

        if len(tokens_uji) < 30:
            os.remove(upload_path)
            error = "Teks terlalu sedikit / PDF scan"
            return render_template("index.html", error=error)

        # ===== AUTO DATASET DARI API =====
        query = extract_keywords(tokens_uji)
        fetch_and_store_journals(query, limit=5)

        best_score = best_rk = best_cs = 0
        best_ref = None

        # ===== PERBANDINGAN =====
        for f in os.listdir(DATASET_FOLDER):
            if f.endswith(ALLOWED_EXT):
                ref_path = os.path.join(DATASET_FOLDER, f)
                text_ref = read_file(ref_path)
                tokens_ref = preprocess(text_ref)

                if len(tokens_ref) < 30:
                    continue

                rk = round(rabin_karp(ngrams(tokens_uji), ngrams(tokens_ref)), 2)
                cs = round(cosine(tokens_uji, tokens_ref), 2)
                hybrid = round((rk + cs) / 2, 2)

                details.append({
                    "ref": f,
                    "rk": rk,
                    "cs": cs,
                    "hybrid": hybrid
                })

                if hybrid > best_score:
                    best_score = hybrid
                    best_rk = rk
                    best_cs = cs
                    best_ref = f

        process_time = round(time.time() - start_time, 2)
        os.remove(upload_path)

        if not details:
            result = {"score": 0, "status": "DATASET KOSONG"}
        else:
            result = {
                "score": best_score,
                "rk": best_rk,
                "cs": best_cs,
                "ref": best_ref,
                "time": process_time,
                "status": "PLAGIARISME" if best_score >= 50 else "TIDAK PLAGIARISME"
            }

    return render_template("index.html", result=result, details=details, error=error)

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
