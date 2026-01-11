from flask import Flask, render_template, request
import pdfplumber, docx, os, math, re, shutil
from collections import Counter
from werkzeug.utils import secure_filename
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"
ALLOWED_EXT = (".pdf", ".docx")

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
        if path.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + " "
        elif path.lower().endswith(".docx"):
            doc = docx.Document(path)
            for p in doc.paragraphs:
                text += p.text + " "
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
# NGRAM
# =====================
def ngrams(tokens, n=3):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# =====================
# SIMILARITY
# =====================
def rabin_karp(ng1, ng2):
    s1, s2 = set(ng1), set(ng2)
    if not s1 or not s2:
        return 0
    return (len(s1 & s2) / len(s1)) * 100

def cosine(t1, t2):
    f1, f2 = Counter(t1), Counter(t2)
    common = set(f1) & set(f2)
    dot = sum(f1[w] * f2[w] for w in common)
    mag1 = math.sqrt(sum(v*v for v in f1.values()))
    mag2 = math.sqrt(sum(v*v for v in f2.values()))
    return 0 if mag1 == 0 or mag2 == 0 else (dot / (mag1 * mag2)) * 100

def hybrid(tokens_uji, tokens_ref):
    return round(
        (0.5 * rabin_karp(ngrams(tokens_uji), ngrams(tokens_ref))) +
        (0.5 * cosine(tokens_uji, tokens_ref)),
        2
    )

# =====================
# ROUTE
# =====================
@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None

    if request.method == "POST":
        file = request.files.get("doc")
        if not file or not file.filename.lower().endswith(ALLOWED_EXT):
            error = "Format file tidak didukung"
            return render_template("index.html", error=error)

        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        dataset_path = os.path.join(DATASET_FOLDER, filename)

        file.save(upload_path)

        # === PROSES FILE UJI ===
        text_uji = read_file(upload_path)
        tokens_uji = preprocess(text_uji)

        if len(tokens_uji) < 30:
            os.remove(upload_path)
            error = "Teks terlalu sedikit atau PDF berupa hasil scan"
            return render_template("index.html", error=error)

        scores = []
        best_match = None

        for f in os.listdir(DATASET_FOLDER):
            if f.lower().endswith(ALLOWED_EXT):
                ref_path = os.path.join(DATASET_FOLDER, f)
                text_ref = read_file(ref_path)
                tokens_ref = preprocess(text_ref)

                if len(tokens_ref) < 30:
                    continue

                score = hybrid(tokens_uji, tokens_ref)
                scores.append(score)

        # === SIMPAN KE DATASET ===
        if not os.path.exists(dataset_path):
            shutil.copy(upload_path, dataset_path)

        os.remove(upload_path)

        if not scores:
            result = {
                "score": 0,
                "status": "DATASET MASIH KOSONG"
            }
        else:
            max_score = max(scores)
            result = {
                "score": max_score,
                "status": "PLAGIARISME" if max_score >= 50 else "TIDAK PLAGIARISME"
            }

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
