"""Big Data Analytics — tujuh aktivitas Recommendation System (materi dosen)."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Recommendation System | Big Data Analytics", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1440px}
[data-testid="stSidebar"] {background: #101a2d}
[data-testid="stSidebar"] .stRadio label {padding: .17rem .1rem}
h1,h2,h3 {letter-spacing:-.035em}
.hero {background:linear-gradient(115deg,#1B2B53,#152B39 75%); padding:26px 30px; border:1px solid #354567; border-radius:18px; margin-bottom:20px}
.hero h1 {font-size:2.0rem; margin:0 0 8px 0;color:#F3F7FF}
.hero p {color:#CFDBEF; margin:0; font-size:.98rem}
.eyebrow {color:#A7C6FF; font-size:12px; font-weight:700; letter-spacing:.11em;margin-bottom:9px}
.note {padding:12px 16px; background:#17243A; border-left:3px solid #7C9AFF; border-radius:8px; margin:14px 0; color:#CDD9ED}
.smallcap {color:#A4B7D7;font-size:13px}
</style>
""", unsafe_allow_html=True)

NAV = [
    "🏠  Overview",
    "🛒  1. Apriori",
    "🧺  2. Market Basket Analysis",
    "🎬  3. Content-Based (Contoh)",
    "🎞️  4. Movie Recommendation",
    "👥  5. User-Based Collaborative",
    "📚  6. Item-Based Collaborative",
    "🧩  7. Hybrid Recommendation",
]

with st.sidebar:
    st.markdown("### 🧠 Big Data Analytics")
    st.caption("RECOMMENDATION SYSTEM · WEEK 7")
    st.divider()
    page = st.radio("Pilih aktivitas", NAV, label_visibility="collapsed")
    st.divider()
    st.caption("**Sumber:** notebook dan dataset praktik dosen. Bukan dataset contoh Kaggle pengganti.")
    st.caption("**Platform:** Streamlit · GitHub · Python 3.12")


def hero(tag, title, detail):
    st.markdown(f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{detail}</p></div>', unsafe_allow_html=True)


def note(message):
    st.markdown(f'<div class="note">{message}</div>', unsafe_allow_html=True)


def metrics(entries):
    cols = st.columns(len(entries))
    for col, (label, value) in zip(cols, entries):
        col.metric(label, value)


def show_table(df, filename="hasil_analisis.csv", height=320):
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)
    st.download_button("⬇️ Unduh hasil CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv")


def chart_layout(fig, height=380):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15,r=15,t=40,b=25), font=dict(color="#DCE7F6"))
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def groceries_raw():
    return pd.read_csv(DATA / "groceries.csv", header=None, dtype=str)


@st.cache_data(show_spinner=False)
def movies_raw():
    return pd.read_csv(DATA / "movie.csv")


@st.cache_data(show_spinner=False)
def ratings_raw():
    return pd.read_csv(DATA / "rating.csv")


@st.cache_data(show_spinner=False)
def book_interactions():
    return pd.read_csv(DATA / "book_knn_interactions.csv.gz", compression="gzip")


def transactions_from_frame(frame):
    return [list(dict.fromkeys(str(x).strip() for x in row.dropna().tolist() if str(x).strip() and str(x).strip().lower() != 'nan')) for _,row in frame.iterrows() if row.notna().any()]


def apriori_tables(transactions, minimum_support, minimum_confidence, maximum_length=None):
    encoder=TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    basket = pd.DataFrame(encoded, columns=encoder.columns_)
    itemsets = apriori(basket, min_support=minimum_support, use_colnames=True, max_len=maximum_length)
    if len(itemsets):
        rules = association_rules(itemsets, metric="confidence", min_threshold=minimum_confidence)
    else:
        rules = pd.DataFrame()
    return itemsets, rules, basket


@st.cache_data(show_spinner="Menghitung Apriori dan aturan asosiasi...")
def run_groceries(support, confidence, max_len, uploaded_bytes=None):
    frame = pd.read_csv(BytesIO(uploaded_bytes), header=None, dtype=str) if uploaded_bytes else groceries_raw()
    transactions = transactions_from_frame(frame)
    itemsets, rules, basket = apriori_tables(transactions, support, confidence, None if max_len == 0 else max_len)
    freq = basket.sum().sort_values(ascending=False)
    return itemsets, rules, freq, len(transactions), basket.shape[1]


def pretty_itemsets(frame):
    d=frame.copy()
    if 'itemsets' in d.columns:
        d['itemsets'] = d['itemsets'].apply(lambda s: ', '.join(sorted(s)))
    return d


def pretty_rules(frame):
    if frame.empty:
        return pd.DataFrame(columns=['Jika membeli', 'Maka membeli', 'Support', 'Confidence', 'Lift'])
    d=frame.copy()
    d['Jika membeli']=d['antecedents'].apply(lambda s:', '.join(sorted(s)))
    d['Maka membeli']=d['consequents'].apply(lambda s:', '.join(sorted(s)))
    return d[['Jika membeli','Maka membeli','support','confidence','lift']].rename(columns={'support':'Support','confidence':'Confidence','lift':'Lift'}).sort_values(['Lift','Confidence'],ascending=False).reset_index(drop=True)


EXAMPLE_BASKET = [
    ['Roti', 'Susu', 'Mentega'], ['Roti', 'Mentega'],
    ['Roti', 'Susu', 'Selai'], ['Apel', 'Pisang'],
    ['Roti', 'Susu', 'Mentega', 'Selai']
]

EXAMPLE_CONTENT = pd.DataFrame({
    'Judul':['The Matrix','Inception','Interstellar','The Notebook','Titanic'],
    'Deskripsi':[
        'Action sci-fi hacker virtual reality computer illusion',
        'Action sci-fi dream mind control heist illusion',
        'Sci-fi space travel black hole time relativity',
        'Romance love story drama beautiful couple memory',
        'Romance drama ship tragedy beautiful love ocean']
})

EXAMPLE_RATINGS = pd.DataFrame({
    'Pengguna':['Andi','Andi','Andi','Budi','Budi','Citra','Citra','Citra','Citra'],
    'Film':['The Matrix','Inception','Avatar','The Matrix','Inception','The Matrix','Inception','Avatar','Titanic'],
    'Rating':[5,4,1,4,5,4,5,1,5]
})
EXAMPLE_HYBRID = pd.DataFrame({
    'Movie':['The Matrix','Inception','Avatar','Titanic','The Notebook'],
    'Genre':['Action Sci-Fi','Action Sci-Fi','Sci-Fi Adventure','Romance Drama','Romance Drama']
})


def ratings_matrix(frame):
    matrix=frame.pivot_table(index='Pengguna', columns='Film',values='Rating').fillna(0)
    sim=pd.DataFrame(cosine_similarity(matrix), index=matrix.index,columns=matrix.index)
    return matrix,sim


@st.cache_data(show_spinner=False)
def prep_movies():
    movies=movies_raw().copy()
    movies['title_year']=movies['title'].str.strip()
    movies['year']=pd.to_numeric(movies['title_year'].str.extract(r'\((\d{4})\)\s*$')[0],errors='coerce')
    movies['title_clean']=movies['title_year'].str.replace(r'\s*\(\d{4}\)\s*$', '',regex=True)
    movies=movies[movies.genres.fillna('(no genres listed)')!='(no genres listed)'].copy().reset_index(drop=True)
    genre_text=movies.genres.str.replace('|',' ',regex=False).str.replace('Sci-Fi','Scifi',regex=False).str.replace('Film-Noir','Noir',regex=False)
    vector=TfidfVectorizer(stop_words='english')
    matrix=vector.fit_transform(genre_text)
    return movies,matrix


@st.cache_resource(show_spinner="Menyiapkan matriks buku dan model kNN...")
def prep_books():
    books=book_interactions()
    pivot = books.pivot_table(index='bookTitle',columns='userID',values='bookRating',aggfunc='mean',fill_value=0)
    sparse=csr_matrix(pivot.to_numpy(dtype=np.float32))
    knn=NearestNeighbors(metric='cosine',algorithm='brute')
    knn.fit(sparse)
    return pivot,sparse,knn


def render_overview():
    hero('DASHBOARD AKADEMIK · MMT ITS', 'Recommendation System', 'Satu dashboard interaktif untuk tujuh aktivitas praktik, empat pendekatan rekomendasi, dan dataset yang disediakan dosen.')
    metrics([('Materi utama','4'),('Aktivitas','7'),('Halaman','8'),('Sumber praktik','Dosen')])
    st.subheader('Peta pembelajaran')
    modules=[
        ('01–02','Association Rule Mining','Apriori dan Market Basket Analysis','Frequent itemsets · Support · Confidence · Lift'),
        ('03–04','Content-Based Filtering','Konten contoh dan MovieLens','TF-IDF · Cosine similarity · Rekomendasi film'),
        ('05–06','Collaborative Filtering','Rating pengguna dan Book-Crossing','User similarity · Item-based kNN'),
        ('07','Hybrid Recommendation','Konten + preferensi pengguna','Skor CB · Skor CF · Pembobotan')]
    for n,title,subtitle,detail in modules:
        with st.container(border=True):
            a,b=st.columns([1,5])
            a.markdown(f'### {n}')
            b.markdown(f'**{title}** — {subtitle}  \n<span class="smallcap">{detail}</span>',unsafe_allow_html=True)
    st.subheader('Dataset sumber')
    groceries=groceries_raw()
    movies=movies_raw()
    books=book_interactions()
    dataset_info=pd.DataFrame([
        ['Apriori contoh','Didefinisikan di notebook',5,'Transaksi sederhana'],
        ['Groceries','data_groceries.csv',len(groceries),'Baris transaksi'],
        ['MovieLens','movie.csv',len(movies),'Film'],
        ['MovieLens rating','rating.csv',len(ratings_raw()),'Rating (data tambahan dalam ZIP)'],
        ['Book-Crossing — setelah filter','BX-Users (1).zip',len(books),'Interaksi rating terpilih'],
        ['User-Based / Hybrid','Didefinisikan di notebook',len(EXAMPLE_RATINGS),'Baris rating contoh'],
    ],columns=['Dataset','Sumber','Jumlah baris','Keterangan'])
    st.dataframe(dataset_info,hide_index=True,use_container_width=True)
    note('Data <b>music.csv</b> dan <b>Online Retail.xlsx</b> disertakan dosen, tetapi tidak dipanggil oleh tujuh script praktik yang menjadi dasar dashboard ini. Keduanya tidak otomatis dimasukkan sebagai aktivitas baru.')
    st.caption('Gunakan menu sebelah kiri untuk menjalankan metode. Tidak semua metode memerlukan pembagian data training/testing.')


def render_apriori():
    hero('AKTIVITAS 01 · ASSOCIATION RULE', 'Apriori pada transaksi contoh', 'Mencari kombinasi barang yang sering dibeli bersama menggunakan lima transaksi dari notebook dosen.')
    up=st.file_uploader('Opsional: unggah CSV transaksi sendiri (tiap baris = satu transaksi; tanpa header)', type=['csv'], key='up_a1')
    if up:
        tx=transactions_from_frame(pd.read_csv(BytesIO(up.getvalue()),header=None,dtype=str))
    else: tx=EXAMPLE_BASKET
    if not tx:
        st.warning('CSV tidak memiliki transaksi yang dapat dianalisis.');return
    a,b=st.columns(2)
    with a: support=st.slider('Minimum support',0.1,1.0,0.4,0.05,key='sup_a1')
    with b: confidence=st.slider('Minimum confidence',0.1,1.0,0.7,0.05,key='conf_a1')
    itemsets,rules,basket=apriori_tables(tx,support,confidence)
    metrics([('Transaksi',f'{len(tx):,}'),('Jenis barang',str(basket.shape[1])),('Frequent itemsets',str(len(itemsets))),('Association rules',str(len(rules)))])
    t1,t2,t3=st.tabs(['🧾 Data transaksi','🔍 Frequent itemsets','🔗 Aturan asosiasi'])
    with t1:
        st.dataframe(pd.DataFrame({'Transaksi':[i+1 for i in range(len(tx))], 'Isi keranjang':[', '.join(t) for t in tx]}), hide_index=True,use_container_width=True)
        st.dataframe(basket.head(15),use_container_width=True)
    with t2:
        st.markdown('Kombinasi barang yang memiliki nilai support ≥ batas pilihan.')
        show_table(pretty_itemsets(itemsets),'apriori_frequent_itemsets.csv')
    with t3:
        display=pretty_rules(rules)
        if display.empty: st.info('Tidak ada aturan untuk parameter yang dipilih. Coba turunkan minimum confidence atau support.')
        else:
            show_table(display,'apriori_rules.csv')
            chart_layout(px.bar(display.head(12),x='Lift',y='Jika membeli',color='Confidence',orientation='h',hover_data=['Maka membeli','Support'],title='Aturan dengan nilai lift terbesar'))
    note('<b>Interpretasi:</b> support menunjukkan proporsi transaksi yang memuat kombinasi barang; confidence menunjukkan peluang barang konsekuen muncul saat anteseden muncul; lift &gt; 1 menunjukkan asosiasi positif relatif terhadap peluang dasarnya.')


def render_groceries():
    hero('AKTIVITAS 02 · ASSOCIATION RULE', 'Market Basket Analysis — Groceries', 'Eksplorasi transaksi belanja dan pembentukan aturan Apriori menggunakan file Groceries dari dosen.')
    up=st.file_uploader('Opsional: ganti data dengan CSV transaksi (tanpa header)', type=['csv'],key='up_gro')
    c1,c2,c3=st.columns(3)
    with c1: support=st.number_input('Minimum support',min_value=0.001,max_value=0.1,value=0.001,step=0.001,format='%.3f',key='sup_g')
    with c2: confidence=st.slider('Minimum confidence',min_value=0.05,max_value=1.0,value=0.8,step=0.05,key='conf_g')
    with c3: max_choice=st.selectbox('Panjang kombinasi maksimal',['Tidak dibatasi','2','3','4'],index=0,help='Tanpa batas mengikuti prinsip script Apriori dalam contoh R.')
    limit=0 if max_choice=='Tidak dibatasi' else int(max_choice)
    try: itemsets,rules,frequency,n_orders,n_items=run_groceries(float(support),float(confidence),limit,up.getvalue() if up else None)
    except Exception as e:
        st.error(f'Perhitungan gagal: {e}');return
    metrics([('Transaksi',f'{n_orders:,}'),('Jenis barang',f'{n_items:,}'),('Frequent itemsets',f'{len(itemsets):,}'),('Aturan asosiasi',f'{len(rules):,}')])
    t1,t2,t3=st.tabs(['📊 Frekuensi barang','🔗 Aturan asosiasi','🎯 Analisis barang tertentu'])
    with t1:
        top=frequency.head(15).rename_axis('Barang').reset_index(name='Frekuensi')
        chart_layout(px.bar(top,x='Frekuensi',y='Barang',orientation='h',title='15 jenis barang paling sering muncul'))
        st.dataframe(top,hide_index=True,use_container_width=True)
    with t2:
        display=pretty_rules(rules)
        if display.empty: st.info('Belum ada aturan yang memenuhi batas support dan confidence. Turunkan parameternya.')
        else:
            show_table(display.head(250), 'groceries_rules_top250.csv')
            graph=display.head(300)
            chart_layout(px.scatter(graph,x='Support',y='Confidence',size='Lift',hover_data=['Jika membeli','Maka membeli'],title='Pola support–confidence–lift'))
    with t3:
        display=pretty_rules(rules)
        prod=st.selectbox('Pilih barang',list(frequency.index),key='target_g')
        direction=st.radio('Posisi barang', ['Barang yang disarankan (RHS)','Barang yang sudah dipilih (LHS)'],horizontal=True)
        if display.empty: st.info('Belum ada aturan yang memenuhi parameter.')
        else:
            col='Maka membeli' if direction.startswith('Barang yang disarankan') else 'Jika membeli'
            filtered=display[display[col].apply(lambda s:prod in [x.strip() for x in s.split(',')])]
            if len(filtered):show_table(filtered.head(30),'aturan_barang_spesifik.csv')
            else: st.info('Tidak ditemukan aturan untuk barang tersebut pada parameter saat ini.')
    note('Script R dosen memanggil dataset <b>Groceries</b> bawaan paket <code>arules</code> untuk perhitungan aturannya, dan memuat CSV <code>data_groceries.csv</code> untuk eksplorasi frekuensi. Dashboard ini menghitung langsung dari CSV yang dikirim dosen; angka dapat sedikit berbeda jika isi file dan opsi algoritma tidak identik dengan objek R.')


def render_content():
    hero('AKTIVITAS 03 · CONTENT-BASED', 'Content-Based Recommendation', 'Rekomendasi berdasarkan kemiripan deskripsi lima film menggunakan TF-IDF dan cosine similarity.')
    st.caption('Default: lima film contoh yang ditulis langsung dalam notebook Content_Based_Recommendation.ipynb.')
    up=st.file_uploader('Opsional: unggah CSV dengan kolom Judul dan Deskripsi',type=['csv'],key='up_cb')
    df=pd.read_csv(up) if up else EXAMPLE_CONTENT.copy()
    if not {'Judul','Deskripsi'}.issubset(df.columns): st.error('CSV memerlukan kolom Judul dan Deskripsi.');return
    df=df[['Judul','Deskripsi']].dropna().drop_duplicates(subset=['Judul']).reset_index(drop=True)
    if len(df)<2: st.warning('Setidaknya diperlukan 2 judul film.');return
    try:
        matrix=TfidfVectorizer(stop_words='english').fit_transform(df['Deskripsi'].astype(str))
        similarity=cosine_similarity(matrix)
    except ValueError as e:st.error(f'Deskripsi belum bisa diolah: {e}');return
    left,right=st.columns([2,1])
    with left: title=st.selectbox('Pilih film yang disukai',df.Judul)
    with right: n=st.slider('Jumlah rekomendasi',1,min(len(df)-1,15),min(2,len(df)-1))
    idx=df.index[df.Judul==title][0]
    nearest=np.argsort(-similarity[idx],kind='stable')
    nearest=[i for i in nearest if i!=idx][:n]
    result=pd.DataFrame({'Film rekomendasi':df.loc[nearest,'Judul'].to_numpy(),'Kemiripan':similarity[idx,nearest]})
    metrics([('Jumlah film',len(df)),('Istilah TF-IDF',matrix.shape[1]),('Rekomendasi',len(result))])
    t1,t2,t3=st.tabs(['🎯 Rekomendasi','📊 Matriks kemiripan','📄 Data deskripsi'])
    with t1:
        show_table(result,'content_based_rekomendasi.csv')
        chart_layout(px.bar(result,x='Kemiripan',y='Film rekomendasi',orientation='h',title=f'Film mirip dengan {title}'))
    with t2:
        if len(df)<=30:
            chart_layout(px.imshow(similarity,x=df.Judul.tolist(),y=df.Judul.tolist(),zmin=0,zmax=1,aspect='auto',title='Cosine similarity antarfilm'),height=420)
        else: st.info('Matriks heatmap dibatasi untuk maksimal 30 film; rekomendasi tetap diproses.')
    with t3: st.dataframe(df,use_container_width=True,hide_index=True)
    note('TF-IDF mengubah deskripsi menjadi representasi angka, lalu cosine similarity mengukur kedekatan konten. Skor tinggi berarti deskripsi lebih mirip, bukan jaminan film akan disukai pengguna.')


def render_movie():
    hero('AKTIVITAS 04 · CONTENT-BASED', 'Movie Recommendation — MovieLens', 'Rekomendasi film berdasarkan kemiripan genre dari dataset Movie.zip yang diberikan dosen.')
    up=st.file_uploader('Opsional: unggah CSV dengan kolom movieId, title, genres',type=['csv'],key='up_mov')
    if up:
        frame=pd.read_csv(up)
        if not {'title','genres'}.issubset(frame.columns):st.error('CSV harus memiliki minimal kolom title dan genres.');return
        frame=frame[frame.genres.fillna('(no genres listed)')!='(no genres listed)'].dropna(subset=['title','genres']).reset_index(drop=True)
        frame['title_year']=frame.title.astype(str)
        frame['year']=pd.to_numeric(frame.title.str.extract(r'\((\d{4})\)$')[0],errors='coerce')
        frame['title_clean']=frame.title_year.str.replace(r'\s*\(\d{4}\)\s*$', '',regex=True)
        vect=TfidfVectorizer(stop_words='english')
        X=vect.fit_transform(frame.genres.astype(str).str.replace('|',' ',regex=False).str.replace('Sci-Fi','Scifi',regex=False).str.replace('Film-Noir','Noir',regex=False))
    else:frame,X=prep_movies()
    if len(frame)<2:st.warning('Data film tidak cukup.');return
    title=st.selectbox('Pilih judul film (bisa diketik untuk mencari)',frame.title_year.tolist())
    n=st.slider('Jumlah rekomendasi',3,20,10)
    ix=int(frame.index[frame.title_year==title][0]); scores=linear_kernel(X[ix],X).ravel();scores[ix]=-1
    top=np.argsort(-scores,kind='stable')[:n]
    result=frame.loc[top,['title_year','genres','year']].copy()
    result['Kesamaan genre']=scores[top]
    result=result.rename(columns={'title_year':'Film rekomendasi','genres':'Genre','year':'Tahun'})
    metrics([('Film valid',f'{len(frame):,}'),('Fitur genre TF-IDF',str(X.shape[1])),('Rekomendasi',len(result))])
    tab1,tab2,tab3=st.tabs(['🎞️ Rekomendasi film','📊 Analisis genre','🗃️ Dataset'])
    with tab1:
        show_table(result,'movie_recommendations.csv')
        chart_layout(px.bar(result.head(12),x='Kesamaan genre',y='Film rekomendasi',orientation='h',title=f'Film paling mirip dengan {title}'),height=430)
    with tab2:
        counts=Counter(g.strip() for val in frame.genres.dropna() for g in str(val).split('|') if g.strip())
        top_genres=pd.DataFrame(counts.most_common(18),columns=['Genre','Jumlah film'])
        chart_layout(px.bar(top_genres,x='Jumlah film',y='Genre',orientation='h',title='18 genre paling umum'))
        st.dataframe(top_genres,hide_index=True,use_container_width=True)
    with tab3:
        st.dataframe(frame[['title_year','genres','year']].head(200),hide_index=True,use_container_width=True)
        st.caption('Rating.csv juga tersedia dari Movie.zip, tetapi notebook Content_Based_Filtering_ML menggunakan fitur genre dari movie.csv untuk metode ini.')
    note('Untuk menghemat memori, dashboard menghitung kemiripan film yang dipilih terhadap semua film, bukan membangun matriks kemiripan penuh 9.000 × 9.000. Rumus cosine similarity tetap setara.')


def render_user_cf():
    hero('AKTIVITAS 05 · COLLABORATIVE FILTERING', 'User-Based Collaborative Filtering', 'Mencari tetangga pengguna paling mirip dari pola rating film yang dicontohkan dalam notebook dosen.')
    up=st.file_uploader('Opsional: unggah CSV dengan kolom Pengguna, Film, Rating',type=['csv'],key='up_ub')
    df=pd.read_csv(up) if up else EXAMPLE_RATINGS.copy()
    if not {'Pengguna','Film','Rating'}.issubset(df.columns):st.error('Butuh kolom Pengguna, Film, Rating.');return
    df=df[['Pengguna','Film','Rating']].dropna()
    df['Rating']=pd.to_numeric(df['Rating'],errors='coerce');df=df.dropna(subset=['Rating'])
    if df.Pengguna.nunique()<2:st.warning('Dibutuhkan minimal dua pengguna.');return
    matrix, sim=ratings_matrix(df)
    left,right=st.columns(2)
    with left: user=st.selectbox('Pilih pengguna',matrix.index)
    with right: threshold=st.slider('Minimal rating dari pengguna serupa',1,5,4)
    neighbours=sim[user].drop(user).sort_values(ascending=False)
    nearest=neighbours.index[0]
    unseen=(matrix.loc[user]==0)&(matrix.loc[nearest]>=threshold)
    recommendations=pd.DataFrame({'Film':matrix.columns[unseen], 'Rating pengguna serupa':matrix.loc[nearest,unseen].values})
    metrics([('Pengguna',df.Pengguna.nunique()),('Film',df.Film.nunique()),('Tetangga terdekat',nearest),('Kemiripan',f'{float(neighbours.iloc[0]):.3f}')])
    t1,t2,t3=st.tabs(['👥 Hasil rekomendasi','📈 Kemiripan pengguna','🗂️ Matriks rating'])
    with t1:
        st.write(f'Pengguna **{user}** paling mirip dengan **{nearest}**.')
        if len(recommendations):show_table(recommendations,'user_based_rekomendasi.csv')
        else:st.info('Tidak ada film baru dengan rating tetangga yang memenuhi batas ini. Kondisi ini mungkin terjadi pada dataset contoh yang sangat kecil.')
    with t2:
        chart_layout(px.imshow(sim,x=sim.columns,y=sim.index,zmin=0,zmax=1,title='Cosine similarity antar pengguna'))
        st.dataframe(neighbours.rename('Skor kesamaan').reset_index().rename(columns={'index':'Pengguna lain'}),hide_index=True,use_container_width=True)
    with t3:st.dataframe(matrix,use_container_width=True)
    note('Angka 0 pada matriks berarti <b>belum ada rating</b> (missing), bukan rating negatif. Penentuan rekomendasi mengikuti notebook: film yang belum dinilai pengguna target tetapi diberi rating tinggi oleh tetangga terdekat.')


def render_item_cf():
    hero('AKTIVITAS 06 · COLLABORATIVE FILTERING', 'Item-Based kNN — Book-Crossing', 'Menemukan judul buku yang serupa dari pola rating pengguna menggunakan k-Nearest Neighbors dengan jarak cosine.')
    st.caption('Dataset siap pakai telah diproses dari tiga CSV dalam BX-Users.zip menggunakan filter yang tercantum di notebook dosen.')
    up=st.file_uploader('Opsional: unggah CSV siap olah dengan kolom userID, bookTitle, bookRating',type=['csv'],key='up_book')
    if up:
        df=pd.read_csv(up)
        if not {'userID','bookTitle','bookRating'}.issubset(df.columns): st.error('CSV harus memiliki kolom userID, bookTitle, bookRating.');return
        df=df[['userID','bookTitle','bookRating']].dropna()
        df['bookRating']=pd.to_numeric(df.bookRating,errors='coerce')
        df=df.dropna(subset=['bookRating']).drop_duplicates(['userID','bookTitle'])
        pivot=df.pivot_table(index='bookTitle',columns='userID',values='bookRating',aggfunc='mean',fill_value=0)
        sparse=csr_matrix(pivot.to_numpy(dtype=np.float32));model=NearestNeighbors(metric='cosine',algorithm='brute').fit(sparse)
    else:
        df=book_interactions()
        pivot,sparse,model=prep_books()
    if len(pivot)<2:st.warning('Jumlah buku tidak cukup.');return
    name=st.selectbox('Pilih buku yang dijadikan acuan',pivot.index.tolist(),index=min(25,len(pivot)-1))
    n=st.slider('Jumlah rekomendasi',1,min(15,len(pivot)-1),min(5,len(pivot)-1))
    index=int(pivot.index.get_loc(name))
    dists,indices=model.kneighbors(sparse[index],n_neighbors=min(n+1,len(pivot)))
    rec=[]
    for j,d in zip(indices[0],dists[0]):
        if j!=index: rec.append({'Buku rekomendasi':pivot.index[j],'Kemiripan (1 − jarak)':round(float(1-d),4),'Jarak cosine':round(float(d),4)})
    result=pd.DataFrame(rec).head(n)
    metrics([('Interaksi setelah filter',f'{len(df):,}'),('Buku',f'{len(pivot):,}'),('Pengguna',f'{len(pivot.columns):,}'),('Rekomendasi',len(result))])
    t1,t2,t3=st.tabs(['📚 Rekomendasi buku','📊 Pola rating','ℹ️ Proses filter data'])
    with t1:
        if len(result):
            show_table(result,'item_based_buku.csv')
            chart_layout(px.bar(result,x='Kemiripan (1 − jarak)',y='Buku rekomendasi',orientation='h',title=f'Buku yang mirip dengan {name[:65]}'),height=420)
    with t2:
        counts=df.bookRating.value_counts().sort_index().rename_axis('Rating').reset_index(name='Jumlah')
        chart_layout(px.bar(counts,x='Rating',y='Jumlah',title='Distribusi rating buku'))
        st.caption('Rating 0 berasal dari data Book-Crossing dan merepresentasikan rating implisit / tanpa penilaian numerik eksplisit.')
    with t3:
        st.markdown('''**Mengikuti tahapan notebook dosen:**

1. Memilih pengguna dengan setidaknya 200 interaksi rating pada data asli.
2. Mempertahankan nilai rating yang muncul sedikitnya 100 kali (sesuai script).
3. Menggabungkan data rating dan judul buku; memilih judul dengan setidaknya 50 interaksi rating.
4. Menyaring pengguna dengan lokasi mengandung `usa` atau `canada`.
5. Menghapus pasangan pengguna–judul yang duplikat, lalu membentuk matriks buku × pengguna untuk kNN cosine.

**Catatan:** agar repo ringan, dashboard menyertakan data hasil filter dari file dosen, bukan keseluruhan arsip mentah. Jumlah buku dapat berbeda bila CSV dibaca dengan kebijakan baris rusak yang berbeda.''')
    note('Jarak cosine yang lebih kecil berarti pola rating buku lebih mirip. Skor ini mengukur kemiripan berdasarkan interaksi, bukan kesamaan isi cerita buku.')


def render_hybrid():
    hero('AKTIVITAS 07 · HYBRID', 'Hybrid Recommendation System', 'Mengombinasikan content-based filtering (kemiripan genre) dan collaborative filtering (pola rating pengguna).')
    movies=EXAMPLE_HYBRID.copy()
    ratings=EXAMPLE_RATINGS.rename(columns={'Pengguna':'User','Film':'Movie'}).copy()
    a,b=st.columns(2)
    with a: target=st.selectbox('Pengguna target',sorted(ratings.User.unique()),index=1)
    with b: cb_weight=st.slider('Bobot Content-Based',0.0,1.0,0.5,0.05)
    cf_weight=1-cb_weight
    tfidf=TfidfVectorizer()
    X=tfidf.fit_transform(movies.Genre)
    item_sim=pd.DataFrame(cosine_similarity(X),index=movies.Movie,columns=movies.Movie)
    matrix=ratings.pivot_table(index='User',columns='Movie',values='Rating').fillna(0)
    user_sim=pd.DataFrame(cosine_similarity(matrix),index=matrix.index,columns=matrix.index)
    liked=ratings.loc[(ratings.User==target)&(ratings.Rating>=4),'Movie'].tolist()
    rated=ratings.loc[ratings.User==target,'Movie'].tolist()
    candidate=[movie for movie in movies.Movie if movie not in rated]
    scores=[]
    for movie in candidate:
        cb=float(np.mean([item_sim.loc[movie,x] for x in liked])) if liked else 0.0
        cf=0.0
        for other,sim in user_sim[target].drop(target).sort_values(ascending=False).items():
            if movie in matrix.columns and matrix.loc[other,movie]>0:
                cf=float(matrix.loc[other,movie]*sim)
                break
        scores.append({'Film':movie,'CB Score':cb,'CF Score':cf})
    score_df=pd.DataFrame(scores)
    if not score_df.empty:
        score_df[['CB Normalisasi','CF Normalisasi']]=MinMaxScaler().fit_transform(score_df[['CB Score','CF Score']])
        score_df['Hybrid Score']=cb_weight*score_df['CB Normalisasi']+cf_weight*score_df['CF Normalisasi']
        score_df=score_df.sort_values('Hybrid Score',ascending=False).reset_index(drop=True)
    metrics([('Pengguna',target),('Film kandidat',len(candidate)),('Bobot CB',f'{cb_weight:.0%}'),('Bobot CF',f'{cf_weight:.0%}')])
    t1,t2,t3=st.tabs(['🏆 Peringkat rekomendasi','⚖️ Kontribusi metode','📖 Data dan rumus'])
    with t1:
        if score_df.empty: st.info('Pengguna sudah memberi rating pada semua film; tidak ada kandidat baru.')
        else:
            show_table(score_df,'hybrid_rekomendasi.csv')
            chart_layout(px.bar(score_df,x='Hybrid Score',y='Film',orientation='h',title=f'Rekomendasi untuk {target}'))
    with t2:
        if not score_df.empty:
            plot=score_df[['Film','CB Normalisasi','CF Normalisasi']].melt(id_vars='Film',var_name='Komponen',value_name='Skor normalisasi')
            chart_layout(px.bar(plot,x='Film',y='Skor normalisasi',color='Komponen',barmode='group',title='Skor tiap metode sebelum diberi bobot'))
    with t3:
        st.markdown('''**Perhitungan (sesuai notebook):**

- **CB Score** = rata-rata kesamaan genre film kandidat terhadap film yang disukai pengguna (rating ≥ 4).
- **CF Score** = rating tetangga terdekat yang pernah menilai film kandidat × kemiripan pengguna.
- **Normalisasi** = MinMaxScaler, dilakukan terpisah untuk CB dan CF pada film kandidat.
- **Hybrid Score** = `(bobot CB × CB Normalisasi) + (bobot CF × CF Normalisasi)`.

Skor merupakan peringkat internal contoh ini, **bukan probabilitas** seorang pengguna akan menyukai film.''')
        st.dataframe(movies,hide_index=True,use_container_width=True)
        st.dataframe(ratings,hide_index=True,use_container_width=True)
    note('Data dan formula dipertahankan dari notebook Hybrid dosen. Bobot dapat diganti untuk melihat pengaruh pendekatan berbasis konten dan pengguna terhadap urutan rekomendasi.')


if page==NAV[0]:render_overview()
elif page==NAV[1]:render_apriori()
elif page==NAV[2]:render_groceries()
elif page==NAV[3]:render_content()
elif page==NAV[4]:render_movie()
elif page==NAV[5]:render_user_cf()
elif page==NAV[6]:render_item_cf()
elif page==NAV[7]:render_hybrid()
