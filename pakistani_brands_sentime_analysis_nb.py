# -*- coding: utf-8 -*-
"""pakistani_brands_sentime_Analysis.nb


**Pipeline:** `YouTube Data API v3` → `Python + Google Colab (comments → CSV)` → `Sentiment Analysis (positive / negative / neutral)` → `Dashboard (audience reactions & brand/campaign comparison)`



**API key:** Google Cloud Console → new project → enable **YouTube Data API v3** → Credentials → API key. In Colab, add it in 🔑 *Secrets* as `YT_API_KEY`.

## Step 1 — Install & import
"""

!pip -q install google-api-python-client vaderSentiment plotly pandas

import re, os, random, html
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 200)

"""## Step 2 — Configuration
Each brand needs a **search name** (used once to find its official channel). If the auto-pick is wrong, paste the brand's `channel_id` (starts with `UC...`) to override it.

You can also add specific campaign videos in `EXTRA_VIDEOS` (e.g. a Lawn / Eid / Independence Day campaign) — they are tagged with the brand name so they appear in the comparison.
"""

# ---------- CONFIG ----------
DEMO_MODE = True               # True = synthetic sample data (no API key). False = real YouTube data.
VIDEOS_PER_BRAND = 5           # latest N uploads per brand channel
MAX_COMMENTS_PER_VIDEO = 400
INCLUDE_REPLIES = False
SENTIMENT_ENGINE = "vader"     # "vader" (fast, + Roman-Urdu lexicon)  or  "xlmr" (multilingual transformer, slower, more accurate)

BRANDS = {
    # brand label : {"query": search text for channel lookup, "channel_id": optional override}
    "Khaadi":     {"query": "Khaadi official",              "channel_id": None},
    "Outfitters": {"query": "Outfitters Pakistan official", "channel_id": None},
    "Sapphire":   {"query": "Sapphire Pakistan official",   "channel_id": None},
    "Gul Ahmed":  {"query": "Gul Ahmed official",           "channel_id": None},
}

# Optional: specific campaign videos -> {"Brand label": ["url or id", ...]}
EXTRA_VIDEOS = {
    # "Khaadi": ["https://www.youtube.com/watch?v=XXXXXXXXXXX"],
}

RAW_CSV  = "youtube_comments_raw.csv"
CSV_PATH = "youtube_comments_sentiment.csv"
HTML_PATH = "dashboard.html"

# ---------- API KEY ----------
API_KEY = None
if not DEMO_MODE:
    try:
        from google.colab import userdata
        API_KEY = userdata.get("YT_API_KEY")
    except Exception:
        import getpass
        API_KEY = os.environ.get("YT_API_KEY") or getpass.getpass("Enter YouTube API key: ")

"""## Step 3 — Collect comments with YouTube Data API v3"""

def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    raise ValueError(f"Could not parse a video ID from: {url_or_id}")


def find_channel(yt, query):
    """Look up a channel by name (costs 100 quota units). Returns (channel_id, title, subscribers)."""
    r = yt.search().list(part="snippet", q=query, type="channel", maxResults=1).execute()
    if not r["items"]:
        raise ValueError(f"No channel found for: {query}")
    cid = r["items"][0]["snippet"]["channelId"]
    return cid


def channel_info(yt, channel_id):
    r = yt.channels().list(part="snippet,statistics,contentDetails", id=channel_id).execute()
    it = r["items"][0]
    return {
        "channel_id": channel_id,
        "title": it["snippet"]["title"],
        "subs": int(it["statistics"].get("subscriberCount", 0)),
        "uploads": it["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def latest_video_ids(yt, uploads_playlist, n):
    ids, token = [], None
    while len(ids) < n:
        r = yt.playlistItems().list(part="contentDetails", playlistId=uploads_playlist,
                                    maxResults=min(50, n - len(ids)), pageToken=token).execute()
        ids += [i["contentDetails"]["videoId"] for i in r["items"]]
        token = r.get("nextPageToken")
        if not token:
            break
    return ids[:n]


def get_video_meta(yt, video_id):
    r = yt.videos().list(part="snippet,statistics", id=video_id).execute()
    if not r["items"]:
        return {"title": video_id, "views": None, "likes": None, "published": None}
    it = r["items"][0]
    st = it.get("statistics", {})
    return {"title": it["snippet"]["title"], "published": it["snippet"]["publishedAt"],
            "views": int(st.get("viewCount", 0)), "likes": int(st.get("likeCount", 0))}


def fetch_comments(yt, video_id, brand, max_comments=400, include_replies=False):
    meta = get_video_meta(yt, video_id)
    rows, token = [], None

    def add(s):
        rows.append({
            "brand": brand, "video_id": video_id, "video_title": meta["title"],
            "video_views": meta["views"], "video_published": meta["published"],
            "author": s.get("authorDisplayName"), "comment": s.get("textDisplay", ""),
            "likes": s.get("likeCount", 0), "published_at": s.get("publishedAt"),
        })

    while len(rows) < max_comments:
        try:
            resp = yt.commentThreads().list(
                part="snippet,replies" if include_replies else "snippet",
                videoId=video_id, maxResults=100, pageToken=token,
                textFormat="plainText", order="time").execute()
        except HttpError as e:
            print(f"   ⚠️  {video_id}: comments unavailable ({getattr(e, 'reason', e)})")
            break
        for item in resp["items"]:
            add(item["snippet"]["topLevelComment"]["snippet"])
            if include_replies:
                for rep in item.get("replies", {}).get("comments", []):
                    add(rep["snippet"])
        token = resp.get("nextPageToken")
        if not token:
            break
    return rows[:max_comments]

"""### Demo data
Synthetic comments (English + Roman Urdu) so the notebook runs without an API key.
> ⚠️ **These are generated, not real customer opinions.** Brand names are used only to demonstrate the layout — don't draw conclusions about any brand from demo mode.
"""

def make_demo_data(seed=7):
    random.seed(seed)
    T = {
     "Price": {
       "Positive": ["Price bohat reasonable hai, value for money", "Itni acha quality is price mein, loved it", "Sale mein kamaal deals thi"],
       "Negative": ["Bohat mehnga hai yaar, price kam karo", "Overpriced, quality ke hisaab se bilkul worth nahi", "Itni mehngai mein kon khareedega"],
       "Neutral":  ["Price kya hai is suit ka?", "Sale kab lag rahi hai?", "Price list kahan milegi?"]},
     "Quality": {
       "Positive": ["Fabric quality zabardast hai", "Stitching bohat acha hai, mashallah", "Cotton ki quality behtareen, long lasting"],
       "Negative": ["Fabric quality bekar ho gayi hai ab", "Stitching ghatia thi, ek wash mein colour fade", "Quality pehle jaisi nahi rahi, disappointed"],
       "Neutral":  ["Is mein lawn hai ya cotton?", "Fabric kaunsa hai?", "Size chart share kar dein"]},
     "Design": {
       "Positive": ["Colors kamaal ke hain, design bohat khoobsurat", "Prints are gorgeous, love the collection", "Ye design to ekdum perfect hai"],
       "Negative": ["Design boring hai, same cheez har saal", "Colors dull hain, expected better", "Collection mein kuch naya nahi, bakwas"],
       "Neutral":  ["Ye collection kab launch hogi?", "Is design ka colour aur kaunsa hai?", "Kya ye unstitched mein available hai?"]},
     "Delivery/Service": {
       "Positive": ["Delivery time pe aayi, customer service bohat acha", "Store staff bohat helpful tha, great service", "Exchange asaani se ho gaya, thank you"],
       "Negative": ["Delivery late aayi aur customer service ne reply nahi kiya", "Worst service, order cancel kar diya without notice", "Website hang hoti hai, bohat frustrating"],
       "Neutral":  ["Lahore store ka address kya hai?", "Kya Karachi mein delivery hoti hai?", "Return policy kya hai?"]},
     "Ad/Model": {
       "Positive": ["Ad bohat pyara hai, music zabardast", "Model looks stunning, beautiful shoot", "Best campaign this year, loved the concept"],
       "Negative": ["Ye ad bakwas hai, cringe", "Model ka selection ghalat hai, bilkul pasand nahi aya", "Boring ad, waste of money"],
       "Neutral":  ["Is ad mein song kaunsa hai?", "Model ka naam kya hai?", "First comment"]},
    }
    # P(positive), P(negative) per brand (rest neutral) – illustrative only
    mix = {"Khaadi": (.52,.22), "Outfitters": (.48,.28), "Sapphire": (.55,.20), "Gul Ahmed": (.42,.30)}
    asp_w = {"Khaadi":[.25,.2,.25,.15,.15], "Outfitters":[.3,.2,.2,.15,.15],
             "Sapphire":[.2,.2,.3,.15,.15], "Gul Ahmed":[.3,.25,.2,.15,.1]}
    aspects = list(T)
    rows, now = [], datetime.utcnow()
    for brand, (pp, pn) in mix.items():
        for v in range(5):
            vid = f"demo_{brand[:3]}_{v}"
            base = now - timedelta(days=random.randint(5, 150))
            for _ in range(random.randint(70, 110)):
                a = random.choices(aspects, asp_w[brand])[0]
                r = random.random()
                s = "Positive" if r < pp else "Negative" if r < pp + pn else "Neutral"
                rows.append({
                    "brand": brand, "video_id": vid, "video_title": f"{brand} – Campaign Video {v+1}",
                    "video_views": random.randint(50_000, 2_000_000), "video_published": base.isoformat() + "Z",
                    "author": f"user_{random.randint(1000, 9999)}", "comment": random.choice(T[a][s]),
                    "likes": int(random.expovariate(1/5)),
                    "published_at": (base + timedelta(days=random.randint(0, 25), hours=random.randint(0, 23))).isoformat() + "Z",
                })
    return rows


all_rows = []
if DEMO_MODE:
    print("🧪 DEMO MODE – synthetic comments (not real opinions).")
    all_rows = make_demo_data()
else:
    yt = build("youtube", "v3", developerKey=API_KEY)
    for brand, cfg in BRANDS.items():
        try:
            cid = cfg.get("channel_id") or find_channel(yt, cfg["query"])
            info = channel_info(yt, cid)
            print(f"\n🔎 {brand}: using channel '{info['title']}' ({info['subs']:,} subscribers) – {cid}")
            print("   ↳ If this is not the official channel, set channel_id for this brand in BRANDS.")
            vids = latest_video_ids(yt, info["uploads"], VIDEOS_PER_BRAND)
        except Exception as e:
            print(f"❌ {brand}: {e}")
            vids = []
        vids += [extract_video_id(v) for v in EXTRA_VIDEOS.get(brand, [])]
        for vid in dict.fromkeys(vids):
            rows = fetch_comments(yt, vid, brand, MAX_COMMENTS_PER_VIDEO, INCLUDE_REPLIES)
            print(f"   ✅ {vid}: {len(rows)} comments")
            all_rows += rows

df = pd.DataFrame(all_rows)
print(f"\nTotal comments collected: {len(df):,}")
df.head()

"""## Step 4 — Clean & save raw comments to CSV"""

def clean_text(t: str) -> str:
    t = html.unescape(str(t))
    t = re.sub(r"<[^>]+>", " ", t)             # HTML tags
    t = re.sub(r"http\S+|www\.\S+", " ", t)    # URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["clean_comment"] = df["comment"].map(clean_text)
df = (df[df["clean_comment"].str.len() > 1]
        .drop_duplicates(subset=["video_id", "author", "clean_comment"])
        .reset_index(drop=True))
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
df["video_published"] = pd.to_datetime(df["video_published"], errors="coerce", utc=True)

df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")
print(f"Saved raw data → {RAW_CSV} ({len(df):,} rows)")

"""## Step 5 — Sentiment analysis (Positive / Negative / Neutral)
**VADER + Roman-Urdu lexicon.** VADER is built for short social-media text; here it is extended with common Roman-Urdu / Urdu-script words and a simple *"<word> nahi"* → *"not <word>"* rule.
Compound score: `≥ 0.05` Positive · `≤ −0.05` Negative · otherwise Neutral.

For the best accuracy on very Urdu-heavy comments, set `SENTIMENT_ENGINE = "xlmr"` in Step 2 (multilingual transformer, slower).
"""

# ---- Roman-Urdu / Urdu lexicon (score range -4 … +4, VADER style) ----
URDU_LEXICON = {
    # positive
    "acha": 2.0, "achi": 2.0, "achay": 2.0, "achha": 2.0, "behtareen": 3.0, "zabardast": 3.2, "kamaal": 3.0, "kamal": 3.0,
    "shandar": 3.0, "khoobsurat": 2.8, "khubsurat": 2.8, "pyara": 2.4, "pyari": 2.4, "pasand": 2.0, "mashallah": 2.6,
    "masha": 1.5, "allah": 0.0, "lajawab": 3.0, "umda": 2.4, "behtar": 2.0, "maza": 2.0, "mazay": 2.0, "mazedar": 2.4,
    "shukriya": 1.8, "jazakallah": 2.0, "wah": 2.0, "zbrdst": 3.0, "best": 2.8, "perfect": 2.8,
    "reasonable": 1.8, "helpful": 2.2, "worth": 1.5, "mubarak": 1.8, "pyaara": 2.4,
    # negative
    "bekar": -2.8, "bakwas": -3.2, "ghatia": -3.0, "bura": -2.4, "buri": -2.4, "mehnga": -2.0, "mehngi": -2.0, "mehngai": -2.2,
    "mahenga": -2.0, "faltu": -2.8, "afsos": -2.0, "afsosnak": -2.6, "dhoka": -3.2, "fraud": -3.2, "jhoot": -2.6,
    "thakawat": -1.5, "ghalat": -2.2, "kharab": -2.6, "kharaab": -2.6, "pareshan": -2.2, "mayoos": -2.4, "mayus": -2.4,
    "cringe": -2.6, "overpriced": -2.8, "late": -1.6, "boring": -2.2, "dull": -1.8, "fade": -1.6, "disappointed": -2.6,
    "frustrating": -2.6, "hang": -1.4, "rude": -2.6, "nafrat": -3.0, "bewaqoof": -2.4, "bewakoof": -2.4,
    # Urdu script
    "زبردست": 3.0, "اچھا": 2.0, "اچھی": 2.0, "بہترین": 3.0, "خوبصورت": 2.8, "کمال": 3.0, "شاندار": 3.0, "ماشاءاللہ": 2.6,
    "پسند": 2.0, "لاجواب": 3.0, "بکواس": -3.2, "گھٹیا": -3.0, "بیکار": -2.8, "مہنگا": -2.0, "مہنگی": -2.0, "برا": -2.4,
    "خراب": -2.6, "فضول": -2.8, "دھوکہ": -3.2,
}
NEGATORS_UR = r"(?:nahi|nahin|nhi|nai|نہیں|نہی)"

sia = SentimentIntensityAnalyzer()
sia.lexicon.update(URDU_LEXICON)

# Roman-Urdu words that VADER mistakes for English sentiment words (e.g. "lag" = "apply/feel", not "lag")
sia.lexicon.update({"lag": 0.0, "lagta": 0.0, "laga": 0.0, "lagi": 0.0, "like": 0.5})

PHRASES = {  # multi-word expressions that carry sentiment
    "expected better": "disappointed", "not worth": "overpriced", "bilkul worth nahi": "overpriced",
    "kuch naya nahi": "boring", "pehle jaisi nahi": "disappointed", "waste of money": "bakwas",
}

def prep(text: str) -> str:
    t = text.lower()
    for k, v in PHRASES.items():
        t = t.replace(k, v)
    # Urdu word order puts the negator AFTER the word: "acha nahi" -> "not acha".
    # Only flip when the word is a known sentiment word; otherwise just drop the negator.
    def _neg(m):
        w = m.group(1)
        return f"not {w}" if w in sia.lexicon else w
    return re.sub(rf"\b(\w+)\s+{NEGATORS_UR}\b", _neg, t)

def vader_score(text):
    return sia.polarity_scores(prep(text))["compound"]

def label(c, pos_th=0.05, neg_th=-0.05):
    return "Positive" if c >= pos_th else "Negative" if c <= neg_th else "Neutral"

if SENTIMENT_ENGINE == "xlmr":
    !pip -q install transformers torch
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                   truncation=True, max_length=256, device=-1)
    preds = clf(df["clean_comment"].tolist(), batch_size=32)
    df["sentiment"] = [p["label"].capitalize() for p in preds]
    sign = {"Positive": 1, "Negative": -1, "Neutral": 0}
    df["compound"] = [sign[s] * p["score"] for s, p in zip(df["sentiment"], preds)]
else:
    df["compound"] = df["clean_comment"].map(vader_score)
    df["sentiment"] = df["compound"].map(label)

print("Sentiment engine:", SENTIMENT_ENGINE)
df["sentiment"].value_counts()

"""## Step 6 — Topic tagging (what are people talking about?)
Keyword rules (English + Roman Urdu) tag each comment with a topic so you can see, e.g., that *Price* drives most negativity for one brand while *Design* drives it for another.
"""

ASPECTS = {
    "Price":            r"price|pricing|mehn?g|mahen?g|expensive|costly|overpriced|afford|sasta|sasti|rate|paisa|paise|rupee|rs\.?\s?\d|value for money|قیمت|مہنگ",
    "Quality/Fabric":   r"quality|fabric|cotton|lawn|stitch|silai|material|kapra|kapray|colou?r fade|fade|wash|shrink|texture|کپڑ|معیار",
    "Design":           r"design|print|colou?rs?|rang|collection|pattern|style|fashion|cutting|embroider|neckline|ڈیزائن|رنگ",
    "Delivery/Service": r"deliver|courier|order|service|staff|store|outlet|website|app\b|exchange|return|refund|support|reply|complain|stock",
    "Ad/Model":         r"\bad\b|advert|commercial|campaign|model|music|song|shoot|video|concept|ambassador|tvc",
    "Sale/Discount":    r"\bsale\b|discount|offer|off\b|clearance|voucher|deal|promo",
}
_compiled = {k: re.compile(v, re.I) for k, v in ASPECTS.items()}

def tag_aspects(text):
    hits = [k for k, rx in _compiled.items() if rx.search(text)]
    return hits or ["General"]

df["topics"] = df["clean_comment"].map(tag_aspects)
df_topics = df.explode("topics").rename(columns={"topics": "topic"})
df_topics["topic"].value_counts()

"""## Step 7 — Save results & summary tables"""

df_out = df.copy()
df_out["topics"] = df_out["topics"].map(lambda x: ", ".join(x))
df_out.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
print(f"Saved → {CSV_PATH}")

order = ["Positive", "Neutral", "Negative"]

# ---- Brand comparison ----
summary = df.groupby(["brand", "sentiment"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
summary["Total"] = summary[order].sum(axis=1)
for s in order:
    summary[f"{s} %"] = (summary[s] / summary["Total"] * 100).round(1)
summary["Avg score"] = df.groupby("brand")["compound"].mean().round(3)
summary["Net sentiment"] = (summary["Positive %"] - summary["Negative %"]).round(1)
summary = summary.sort_values("Net sentiment", ascending=False)
display(summary)

# ---- Per-video (campaign) view ----
by_video = (df.groupby(["brand", "video_title"])
              .agg(comments=("sentiment", "size"),
                   positive_pct=("sentiment", lambda s: round((s == "Positive").mean() * 100, 1)),
                   negative_pct=("sentiment", lambda s: round((s == "Negative").mean() * 100, 1)),
                   avg_score=("compound", "mean"))
              .round(3).reset_index().sort_values("avg_score", ascending=False))
print("\nBest and worst received videos:")
display(pd.concat([by_video.head(3), by_video.tail(3)]))

"""## Step 8 — Interactive dashboard"""

COLORS = {"Positive": "#2e9e6b", "Neutral": "#9aa3ad", "Negative": "#d9534f"}
brands = list(summary.index)
TITLE = "Pakistani Fashion Brands – YouTube Sentiment Dashboard" + ("  (DEMO – synthetic data)" if DEMO_MODE else "")

STOP = set("""a an the and or but if is are was were be been to of in on at for with this that it its i you we they he she
me my your our their from as by not no so very just too have has had do does did will would can could video comment first what when
where who how which there here about all more some any one like get got really hai hain ho ka ki ke ko se mein me par pe yeh ye woh wo
aur bhi kya kuch bohat bahut tha thi thay kar karo kar di raha rahi rahe nahi nahin hi tou toh jo jis ab phir lekin magar""".split())

def top_words(sub, n=10):
    words = re.findall(r"[a-zA-Z']{3,}", " ".join(sub["clean_comment"]).lower())
    return Counter(w for w in words if w not in STOP).most_common(n)

fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"type": "domain"}, {"type": "xy"}],
           [{"type": "xy", "colspan": 2}, None],
           [{"type": "xy"}, {"type": "xy"}]],
    subplot_titles=("Overall audience reaction", "Brand comparison (% of comments)",
                    "Sentiment trend (weekly average score)",
                    "Topic sentiment heatmap (avg score)", "Top words in negative comments"),
    vertical_spacing=0.13, horizontal_spacing=0.12,
)

# 1) Overall donut
overall = df["sentiment"].value_counts().reindex(order, fill_value=0)
fig.add_trace(go.Pie(labels=overall.index, values=overall.values, hole=.55,
                     marker=dict(colors=[COLORS[s] for s in overall.index]),
                     textinfo="percent+label", showlegend=False), row=1, col=1)

# 2) Brand comparison – stacked %
for s in order:
    fig.add_trace(go.Bar(x=brands, y=summary.loc[brands, f"{s} %"], name=s, marker_color=COLORS[s],
                         text=summary.loc[brands, f"{s} %"].round(0).astype(int).astype(str) + "%",
                         textposition="inside"), row=1, col=2)
fig.update_yaxes(title_text="% of comments", row=1, col=2)

# 3) Weekly trend per brand
t = df.dropna(subset=["published_at"]).copy()
t["week"] = t["published_at"].dt.tz_localize(None).dt.to_period("W").dt.start_time
trend = t.groupby(["week", "brand"])["compound"].mean().reset_index()
for b in brands:
    d = trend[trend["brand"] == b].sort_values("week")
    fig.add_trace(go.Scatter(x=d["week"], y=d["compound"], mode="lines+markers", name=b, legendgroup="brands"), row=2, col=1)
fig.add_trace(go.Scatter(x=[trend["week"].min(), trend["week"].max()], y=[0, 0], mode="lines",
                         line=dict(dash="dot", color="#999"), showlegend=False, hoverinfo="skip"), row=2, col=1)
fig.update_yaxes(title_text="avg sentiment", row=2, col=1)

# 4) Topic x brand heatmap
topics_order = [k for k in list(ASPECTS) + ["General"] if k in set(df_topics["topic"])]
heat = df_topics.pivot_table(index="brand", columns="topic", values="compound", aggfunc="mean").reindex(index=brands, columns=topics_order)
fig.add_trace(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="RdYlGn", zmid=0,
                         text=np.round(heat.values, 2), texttemplate="%{text}", showscale=False), row=3, col=1)

# 5) Top negative words
neg_words = top_words(df[df["sentiment"] == "Negative"])
if neg_words:
    w, n = zip(*neg_words)
    fig.add_trace(go.Bar(x=list(n)[::-1], y=list(w)[::-1], orientation="h",
                         marker_color=COLORS["Negative"], showlegend=False), row=3, col=2)

fig.update_layout(barmode="stack", height=1150, template="plotly_white", title_text=TITLE, title_x=0.5,
                  legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
fig.show()

"""## Step 9 — Most-liked comments & problem areas"""

for s in ["Positive", "Negative"]:
    print(f"\n=== Most-liked {s} comments ===")
    display(df[df["sentiment"] == s].sort_values("likes", ascending=False)
            [["brand", "clean_comment", "likes", "compound"]].head(6))

# Which topic hurts each brand most?
worst = (df_topics.groupby(["brand", "topic"])
           .agg(comments=("compound", "size"), avg_score=("compound", "mean"))
           .query("comments >= 10").round(3).reset_index()
           .sort_values(["brand", "avg_score"]).groupby("brand").head(2))
print("\n=== Weakest topics per brand (min 10 comments) ===")
display(worst)

"""## Step 10 — Export the dashboard & download"""

kpi_html = summary.reset_index().to_html(index=False, border=0, classes="kpi")
vid_html = by_video.to_html(index=False, border=0, classes="kpi")
banner = '<p style="background:#fff3cd;padding:10px;border-radius:6px;text-align:center">⚠️ DEMO – synthetic sample comments, not real audience opinions.</p>' if DEMO_MODE else ""
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{TITLE}</title>
<style>body{{font-family:system-ui,Segoe UI,Arial;margin:24px;max-width:1250px;margin-inline:auto;color:#222}}
h1,h2{{text-align:center}} table.kpi{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}
.kpi th,.kpi td{{padding:7px 10px;border-bottom:1px solid #e5e5e5;text-align:center}} .kpi th{{background:#f6f6f6}}</style></head>
<body><h1>🛍️ {TITLE}</h1>{banner}
<p style="text-align:center">{len(df):,} comments · {len(brands)} brands · generated {datetime.now():%Y-%m-%d %H:%M}</p>
<h2>Brand scorecard</h2>{kpi_html}
{fig.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Video / campaign breakdown</h2>{vid_html}</body></html>"""
with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(page)
print("Saved:", HTML_PATH, "|", CSV_PATH, "|", RAW_CSV)

try:
    from google.colab import files
    for p in (HTML_PATH, CSV_PATH):
        files.download(p)
except Exception:
    pass

"""## Notes
- **Quota:** looking up a channel by name (`search.list`) costs 100 units; everything else is ~1 unit per call. Four brands ≈ 400 units for lookups — well inside the free 10,000/day. Paste `channel_id`s into `BRANDS` after the first run to skip the lookups.
- **Check the channel:** the notebook prints which channel it picked for each brand. Verify it's the official one.
- **Comments disabled:** some videos turn comments off; they are skipped with a warning.
- **Limits of lexicon sentiment:** sarcasm and mixed-language comments will sometimes be mislabelled. Use the `xlmr` engine and spot-check a sample of 50–100 comments before reporting numbers.
- **Responsible use:** YouTube comments are public, but report aggregated results and don't republish usernames.
- **Ideas to extend:** Streamlit app, BERTopic for automatic topics, spike alerts on negative sentiment during Eid/Lawn launches, correlate sentiment with video views.
"""
