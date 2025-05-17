import os, json, hashlib, re, requests, tqdm
from zipfile import ZipFile

DATA_DIR  = "/Data/MMEL/wiki_D"
IMG_DIR   = os.path.join(DATA_DIR, "wikinewsImgs")
ANNOT_ZIP = os.path.join(DATA_DIR, "annotated_data_V2.zip")
os.makedirs(IMG_DIR, exist_ok=True)

# Build a session with a proper User‑Agent header
session = requests.Session()
session.headers.update({
    "User-Agent": "WikiDiverseDownloader/0.1 (vijay.murugan@ip-paris.fr)"
})

def hash_name(url: str) -> str:
    m_img  = url.split("/")[-1]
    prefix = hashlib.md5(m_img.encode()).hexdigest()
    suffix = re.sub(
        r'(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))',
        "",
        m_img
    )
    name = prefix + suffix
    return name.replace(".svg", ".png").replace(".SVG", ".png")

# Collect all unique image URLs
urls = set()
with ZipFile(ANNOT_ZIP) as z:
    for member in z.namelist():
        if member.endswith(".json"):
            for caption, img_url, topic, mentions in json.loads(z.read(member)):
                urls.add(img_url)

# Download each
for url in tqdm.tqdm(sorted(urls)):
    fname = hash_name(url)
    fpath = os.path.join(IMG_DIR, fname)
    if os.path.exists(fpath):
        continue
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        print("FAILED", url, e)
