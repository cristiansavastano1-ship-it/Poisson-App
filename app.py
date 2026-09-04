import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import date, datetime
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Poisson Betting Model", page_icon="🔮", layout="centered")

# =====================================================================
# 📊 DATABASE CAMPIONATI
# I campionati "solo_previsione" (Champions/Europa League) usano football-data.org
# (serve una chiave API gratuita, vedi sidebar) e NON hanno colonne di quote
# bookmaker: previsione Poisson sì, value finder/backtest/CLV no — la fonte
# dati semplicemente non li fornisce.
# =====================================================================
CAMPIONATI = {
    "Italia - Serie A":              {"id_fd": "I1"},
    "Italia - Serie B":               {"id_fd": "I2"},
    "Inghilterra - Premier League":   {"id_fd": "E0"},
    "Inghilterra - Championship":     {"id_fd": "E1"},
    "Spagna - La Liga":               {"id_fd": "SP1"},
    "Spagna - Segunda Division":      {"id_fd": "SP2"},
    "Germania - Bundesliga":          {"id_fd": "D1"},
    "Germania - 2. Bundesliga":       {"id_fd": "D2"},
    "Francia - Ligue 1":              {"id_fd": "F1"},
    "Francia - Ligue 2":              {"id_fd": "F2"},
    "Olanda - Eredivisie":            {"id_fd": "N1"},
    "Portogallo - Primeira Liga":     {"id_fd": "P1"},
    "Belgio - Pro League":            {"id_fd": "B1"},
    "Turchia - Süper Lig":            {"id_fd": "T1"},
    "🌍 Champions League (solo previsione)": {"id_fdorg": "CL", "solo_previsione": True},
    "🌍 Europa League (solo previsione)":    {"id_fdorg": "EL", "solo_previsione": True},
}

FILE_CLV_PERSONALE = "clv_personale.json"

EWMA_SPAN = 6
GIORNI_EMIVITA_DECADIMENTO = 180

# =====================================================================
# 🧠 MOTORE — stessa logica del tool Jupyter (Dixon-Coles, EWMA, time decay)
# =====================================================================
def codici_stagione(oggi=None):
    oggi = oggi or date.today()
    anno_inizio_corrente = oggi.year if oggi.month >= 7 else oggi.year - 1
    anno_inizio_precedente = anno_inizio_corrente - 1
    fmt = lambda a: f"{a % 100:02d}{(a + 1) % 100:02d}"
    return fmt(anno_inizio_corrente), fmt(anno_inizio_precedente)


def tau_dixon_coles(gc, gt, lc, lt, rho):
    if gc == 0 and gt == 0: return 1 - (lc * lt * rho)
    elif gc == 0 and gt == 1: return 1 + (lc * rho)
    elif gc == 1 and gt == 0: return 1 + (lt * rho)
    elif gc == 1 and gt == 1: return 1 - rho
    return 1.0


def media_ewma(serie):
    serie = serie.dropna()
    if len(serie) == 0: return None
    return serie.ewm(span=EWMA_SPAN, min_periods=1).mean().iloc[-1]


def media_pesata_decadimento(df, colonna, data_riferimento, emivita_giorni=GIORNI_EMIVITA_DECADIMENTO):
    sub = df[[colonna, 'Date_parsed']].dropna()
    if len(sub) == 0: return None
    giorni = (data_riferimento - sub['Date_parsed']).dt.days.clip(lower=0)
    pesi = 0.5 ** (giorni / emivita_giorni)
    tot = pesi.sum()
    return sub[colonna].mean() if tot <= 0 else (sub[colonna] * pesi).sum() / tot


def calcola_modello(giocate, squadra_casa, squadra_trasferta, rho, data_riferimento=None):
    n_storico = len(giocate)
    if n_storico < 15:
        return None
    if data_riferimento is None or pd.isna(data_riferimento):
        data_riferimento = giocate['Date_parsed'].max()

    m_gol_casa = media_pesata_decadimento(giocate, 'FTHG', data_riferimento)
    m_gol_trasf = media_pesata_decadimento(giocate, 'FTAG', data_riferimento)
    if m_gol_casa is None or m_gol_trasf is None:
        return None

    forma_casa = giocate[giocate['HomeTeam'] == squadra_casa]
    forma_trasf = giocate[giocate['AwayTeam'] == squadra_trasferta]

    gf_casa_rec = media_ewma(forma_casa['FTHG']) or m_gol_casa
    gs_casa_rec = media_ewma(forma_casa['FTAG']) or m_gol_trasf
    gf_trasf_rec = media_ewma(forma_trasf['FTAG']) or m_gol_trasf
    gs_trasf_rec = media_ewma(forma_trasf['FTHG']) or m_gol_casa

    tiri_casa = (media_ewma(forma_casa['HST']) if 'HST' in forma_casa.columns else None) or 4.0
    corner_casa = (media_ewma(forma_casa['HC']) if 'HC' in forma_casa.columns else None) or 5.0
    tiri_trasf = (media_ewma(forma_trasf['AST']) if 'AST' in forma_trasf.columns else None) or 3.5
    corner_trasf = (media_ewma(forma_trasf['AC']) if 'AC' in forma_trasf.columns else None) or 4.5

    pericolo_casa = (tiri_casa + corner_casa * 0.3) / 5.5
    pericolo_trasf = (tiri_trasf + corner_trasf * 0.3) / 4.8

    m_gf_c = media_pesata_decadimento(forma_casa, 'FTHG', data_riferimento)
    m_gs_c = media_pesata_decadimento(forma_casa, 'FTAG', data_riferimento)
    m_gf_t = media_pesata_decadimento(forma_trasf, 'FTAG', data_riferimento)
    m_gs_t = media_pesata_decadimento(forma_trasf, 'FTHG', data_riferimento)

    att_casa_st = (m_gf_c / m_gol_casa) if m_gf_c is not None else 1.0
    dif_casa_st = (m_gs_c / m_gol_trasf) if m_gs_c is not None else 1.0
    att_trasf_st = (m_gf_t / m_gol_trasf) if m_gf_t is not None else 1.0
    dif_trasf_st = (m_gs_t / m_gol_casa) if m_gs_t is not None else 1.0

    # FIX #19 — SHRINKAGE verso la media di lega (1.0) per campioni piccoli.
    # Con poche partite specifiche (neopromosse, inizio stagione) il rapporto
    # attacco/difesa calcolato sopra è rumoroso. Lo "tiriamo" verso 1.0 (media
    # di lega) in proporzione a quante partite specifiche abbiamo: K_SHRINKAGE
    # è il numero di partite dopo il quale il dato specifico pesa metà e metà
    # con la media di lega — sotto K conta di più la media, sopra K conta di
    # più il dato osservato.
    K_SHRINKAGE = 10
    peso_casa = len(forma_casa) / (len(forma_casa) + K_SHRINKAGE)
    peso_trasf = len(forma_trasf) / (len(forma_trasf) + K_SHRINKAGE)
    att_casa_st = peso_casa * att_casa_st + (1 - peso_casa) * 1.0
    dif_casa_st = peso_casa * dif_casa_st + (1 - peso_casa) * 1.0
    att_trasf_st = peso_trasf * att_trasf_st + (1 - peso_trasf) * 1.0
    dif_trasf_st = peso_trasf * dif_trasf_st + (1 - peso_trasf) * 1.0

    att_casa = att_casa_st * 0.70 + ((gf_casa_rec / max(0.1, m_gol_casa)) * pericolo_casa) * 0.30
    dif_casa = dif_casa_st * 0.70 + (gs_casa_rec / max(0.1, m_gol_trasf)) * 0.30
    att_trasf = att_trasf_st * 0.70 + ((gf_trasf_rec / max(0.1, m_gol_trasf)) * pericolo_trasf) * 0.30
    dif_trasf = dif_trasf_st * 0.70 + (gs_trasf_rec / max(0.1, m_gol_casa)) * 0.30

    lam_c = att_casa * dif_trasf * m_gol_casa
    lam_t = att_trasf * dif_casa * m_gol_trasf

    prob_1, prob_x, prob_2, prob_goal, prob_nogoal = 0.0, 0.0, 0.0, 0.0, 0.0
    limiti = [0.5, 1.5, 2.5, 3.5, 4.5]
    prob_under = {l: 0.0 for l in limiti}
    risultati = []

    for gc in range(8):
        for gt in range(8):
            p = poisson.pmf(gc, lam_c) * poisson.pmf(gt, lam_t) * tau_dixon_coles(gc, gt, lam_c, lam_t, rho) * 100
            segno = 'X' if gc == gt else ('1' if gc > gt else '2')
            if segno == '1': prob_1 += p
            elif segno == 'X': prob_x += p
            else: prob_2 += p
            if gc > 0 and gt > 0: prob_goal += p
            else: prob_nogoal += p
            for l in limiti:
                if gc + gt < l: prob_under[l] += p
            risultati.append({'res': f"{gc}-{gt}", 'p': p, 'segno': segno})

    tot = sum(r['p'] for r in risultati)
    if tot > 0:
        f = 100.0 / tot
        for r in risultati: r['p'] *= f
        prob_1, prob_x, prob_2 = prob_1*f, prob_x*f, prob_2*f
        prob_goal, prob_nogoal = prob_goal*f, prob_nogoal*f
        prob_under = {l: v*f for l, v in prob_under.items()}

    return {"n_storico": n_storico, "lambda_casa": lam_c, "lambda_trasferta": lam_t,
            "prob_1": prob_1, "prob_X": prob_x, "prob_2": prob_2,
            "prob_goal": prob_goal, "prob_nogoal": prob_nogoal,
            "prob_under": prob_under, "risultati": risultati,
            "n_partite_casa": len(forma_casa), "n_partite_trasferta": len(forma_trasf)}


def classifica_colonne_quote(colonne):
    ap_h = [c for c in colonne if c.endswith('H') and not c.endswith('CH') and c not in ['FTHG','HTHG','PTHG']]
    ap_d = [c for c in colonne if c.endswith('D') and not c.endswith('CD') and c not in ['FTHG','FTAG','HTHG','HTAG']]
    ap_a = [c for c in colonne if c.endswith('A') and not c.endswith('CA') and c not in ['FTAG','HTAG','PTAG']]
    ch_h = [c for c in colonne if c.endswith('CH')]
    ch_d = [c for c in colonne if c.endswith('CD')]
    ch_a = [c for c in colonne if c.endswith('CA')]
    return {"apertura": (ap_h, ap_d, ap_a), "chiusura": (ch_h, ch_d, ch_a)}


def quote_mercato_normalizzate(riga, colonne_h, colonne_d, colonne_a):
    vh = [riga[c] for c in colonne_h if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    vd = [riga[c] for c in colonne_d if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    va = [riga[c] for c in colonne_a if pd.notna(riga.get(c)) and isinstance(riga.get(c), (int, float))]
    if not (vh and vd and va): return None
    qh, qd, qa = sum(vh)/len(vh), sum(vd)/len(vd), sum(va)/len(va)
    pih, pid, pia = 1/qh, 1/qd, 1/qa
    over = pih + pid + pia
    return {"q_casa_grezza": qh, "q_x_grezza": qd, "q_trasf_grezza": qa,
            "q_casa_equa": over/pih, "q_x_equa": over/pid, "q_trasf_equa": over/pia,
            "overround": over, "n_bookmakers": len(vh)}


def valuta_affidabilita(quota_book_norm, quota_macchina, n_storico):
    if quota_book_norm <= quota_macchina:
        return 0, "❌ Nessun valore"
    margine = ((quota_book_norm - quota_macchina) / quota_macchina) * 100
    fattore = min(1.0, n_storico / 40)
    score = max(0, min(100, margine * 3.0 * fattore))
    if score >= 70: return score, "🔥 ALTA AFFIDABILITÀ"
    elif score >= 50: return score, "⚠️ MEDIA AFFIDABILITÀ"
    return score, "🛑 BASSA AFFIDABILITÀ"


# =====================================================================
# 📥 CARICAMENTO DATI (cache 1 ora — evita di riscaricare a ogni click)
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def carica_dati_campionato(id_fd):
    codice_corrente, codice_precedente = codici_stagione()
    frames = []
    for codice, label in [(codice_precedente, 'precedente'), (codice_corrente, 'corrente')]:
        try:
            url = f"https://www.football-data.co.uk/mmz4281/{codice}/{id_fd}.csv"
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            df['Stagione'] = label
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    dati = pd.concat(frames, ignore_index=True, sort=False)
    dati['Date_parsed'] = pd.to_datetime(dati['Date'], errors='coerce', dayfirst=True)
    dati = dati.sort_values('Date_parsed').reset_index(drop=True)
    dati = dati.drop_duplicates(subset=['Date_parsed', 'HomeTeam', 'AwayTeam'], keep='last').reset_index(drop=True)
    return dati


@st.cache_data(ttl=1800, show_spinner=False)
def carica_fixture_future(id_fd):
    try:
        fx = pd.read_csv("https://www.football-data.co.uk/fixtures.csv")
        fx.columns = fx.columns.str.strip()
        if 'Div' not in fx.columns: return pd.DataFrame()
        fx = fx[fx['Div'] == id_fd].copy()
        if len(fx) == 0: return pd.DataFrame()
        fx['Date_parsed'] = pd.to_datetime(fx['Date'], errors='coerce', dayfirst=True)
        oggi = pd.Timestamp(date.today())
        fx = fx[fx['Date_parsed'] >= oggi]
        fx['FTHG'] = np.nan
        fx['FTAG'] = np.nan
        return fx.sort_values('Date_parsed').reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# =====================================================================
# 🌍 CHAMPIONS/EUROPA LEAGUE — solo previsione, via football-data.org
# Nessuna colonna di quote bookmaker in questa fonte: value finder, backtest
# e CLV non sono disponibili per queste competizioni, solo la previsione
# Poisson pura (1X2, gol/no gol, over/under, risultato esatto).
# =====================================================================
@st.cache_data(ttl=900, show_spinner=False)
def carica_dati_fdorg(codice_comp, api_key):
    if not api_key:
        return None
    url = f"https://api.football-data.org/v4/competitions/{codice_comp}/matches"
    headers = {"X-Auth-Token": api_key}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        dati_json = resp.json()
        righe = []
        for m in dati_json.get("matches", []):
            finita = m.get("status") == "FINISHED"
            righe.append({
                "HomeTeam": m["homeTeam"]["name"],
                "AwayTeam": m["awayTeam"]["name"],
                "Date": m["utcDate"][:10],
                "FTHG": m["score"]["fullTime"]["home"] if finita else np.nan,
                "FTAG": m["score"]["fullTime"]["away"] if finita else np.nan,
            })
        if not righe:
            return None
        df = pd.DataFrame(righe)
        df['Date_parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Stagione'] = 'corrente'
        return df.sort_values('Date_parsed').reset_index(drop=True)
    except Exception:
        return None


# =====================================================================
# 📌 REGISTRO CLV PERSONALE — "impara nel tempo"
# Per le partite future, football-data.co.uk non ha ancora una vera quota di
# chiusura (la partita non è stata giocata). Qui teniamo traccia noi stessi:
# alla PRIMA volta che analizzi una partita, salviamo la quota vista in quel
# momento. Ogni volta che la riguardi (anche dopo che la partita è stata
# giocata e ha una chiusura ufficiale), il sistema confronta con quella prima
# rilevazione, mostrandoti il movimento reale sui TUOI tempi di consultazione.
#
# LIMITE — su hosting gratuito (Streamlit Community Cloud) questo file può
# azzerarsi se il server si riavvia dopo inattività prolungata o dopo un
# aggiornamento del codice: non è un bug, è un limite del piano gratuito.
# =====================================================================
def carica_registro_clv():
    if os.path.exists(FILE_CLV_PERSONALE):
        try:
            with open(FILE_CLV_PERSONALE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def salva_registro_clv(registro):
    try:
        with open(FILE_CLV_PERSONALE, "w") as f:
            json.dump(registro, f)
    except Exception:
        pass


def chiave_partita(id_camp, partita):
    return f"{id_camp}|{partita.get('HomeTeam','?')}|{partita.get('AwayTeam','?')}|{str(partita.get('Date','?'))}"


# =====================================================================
# 🖥️ INTERFACCIA
# =====================================================================
st.title("🔮 Poisson Betting Model")
st.caption("Dixon-Coles + EWMA + Time Decay — versione web del tool Jupyter V16")

if "rho" not in st.session_state:
    st.session_state.rho = -0.10

with st.sidebar:
    st.subheader("⚙️ Impostazioni")
    api_key_fdorg = st.text_input(
        "Chiave API football-data.org (solo per Champions/Europa League)",
        type="password",
        help="Gratuita: registrati su football-data.org/client/register. "
             "Non serve per i campionati domestici, solo per le competizioni europee."
    )
    if st.button("🔄 Aggiorna quote/fixture ora (forza dati freschi)"):
        carica_fixture_future.clear()
        carica_dati_fdorg.clear()
        st.success("Cache svuotata: al prossimo caricamento i dati saranno quelli attuali.")

campionato = st.selectbox("Torneo", list(CAMPIONATI.keys()))
info_campionato = CAMPIONATI[campionato]
solo_previsione = info_campionato.get("solo_previsione", False)

if solo_previsione:
    st.warning("🌍 Competizione europea: solo previsione Poisson. Niente quote di mercato, "
               "value finder, backtest o CLV — football-data.org non fornisce quote bookmaker.")
    with st.spinner("Caricamento dati..."):
        dati = carica_dati_fdorg(info_campionato["id_fdorg"], api_key_fdorg)
    fixture_future = pd.DataFrame()
    if dati is None:
        if not api_key_fdorg:
            st.info("Inserisci una chiave API di football-data.org nella barra laterale per usare questa competizione.")
        else:
            st.error("Impossibile scaricare i dati (chiave non valida o servizio non raggiungibile).")
        st.stop()
    # Le fixture future per le competizioni europee sono le partite non ancora
    # giocate presenti nello stesso dataset (status diverso da FINISHED).
    fixture_future = dati[dati['FTHG'].isna()].copy()
    dati = dati[dati['FTHG'].notna()].copy()
    id_fd = info_campionato["id_fdorg"]
else:
    id_fd = info_campionato["id_fd"]
    with st.spinner("Caricamento dati..."):
        dati = carica_dati_campionato(id_fd)
        fixture_future = carica_fixture_future(id_fd)
    if dati is None:
        st.error("Impossibile scaricare i dati per questo campionato.")
        st.stop()

if 'Stagione' in dati.columns:
    n_corrente = int(((dati['Stagione'] == 'corrente') & (dati['FTHG'].notna())).sum())
    st.info(f"📊 Partite di stagione corrente già giocate e disponibili: **{n_corrente}**")

tab_analisi, tab_rho, tab_backtest = st.tabs(["🔮 Analisi Partita", "🎯 Stima ρ", "📈 Backtest"])

# ---------- TAB ANALISI ----------
with tab_analisi:
    opzioni = []
    partite_map = []

    if len(fixture_future) > 0:
        for _, r in fixture_future.iterrows():
            opzioni.append(f"⚽ FUTURA ({r['Date']}): {r['HomeTeam']} vs {r['AwayTeam']}")
            partite_map.append(r.to_dict())

    storiche = dati[dati['FTHG'].notna()].tail(20)
    for _, r in storiche.iterrows():
        opzioni.append(f"➡️ RISCONTRO ({r.get('Date','?')}): {r['HomeTeam']} vs {r['AwayTeam']}")
        partite_map.append(r.to_dict())

    if not opzioni:
        st.warning("Nessuna partita disponibile per questo campionato al momento.")
    else:
        scelta = st.selectbox("Partita", opzioni)
        idx = opzioni.index(scelta)
        partita = partite_map[idx]

        if st.button("🔮 Elabora Analisi", type="primary"):
            data_rif = partita.get('Date_parsed')
            if pd.notna(data_rif):
                giocate = dati[(dati['FTHG'].notna()) & (dati['Date_parsed'] < data_rif)].copy()
            else:
                giocate = dati[dati['FTHG'].notna()].copy()

            modello = calcola_modello(giocate, partita['HomeTeam'], partita['AwayTeam'],
                                       st.session_state.rho, data_riferimento=data_rif)

            if modello is None:
                st.warning("Pochi dati storici per un'analisi accurata.")
            else:
                st.subheader(f"{partita['HomeTeam']} vs {partita['AwayTeam']}")
                if pd.notna(partita.get('FTHG')):
                    st.success(f"Risultato reale: {int(partita['FTHG'])} - {int(partita['FTAG'])}")

                # FIX APP #2 — avviso trasparenza dati: se una delle due squadre ha
                # pochissime partite specifiche nello storico (es. neopromossa,
                # inizio stagione), la stima per quella squadra è meno affidabile
                # anche se il campione di lega complessivo (n_storico) è ampio.
                SOGLIA_AVVISO_POCHE_PARTITE = 5
                squadre_scarse = []
                if modello["n_partite_casa"] < SOGLIA_AVVISO_POCHE_PARTITE:
                    squadre_scarse.append(f"{partita['HomeTeam']} (solo {modello['n_partite_casa']} partite in casa nello storico)")
                if modello["n_partite_trasferta"] < SOGLIA_AVVISO_POCHE_PARTITE:
                    squadre_scarse.append(f"{partita['AwayTeam']} (solo {modello['n_partite_trasferta']} partite in trasferta nello storico)")
                if squadre_scarse:
                    st.warning("⚠️ Stima poco affidabile: " + " e ".join(squadre_scarse) +
                               " — probabile neopromossa o inizio stagione. Il modello compensa "
                               "parzialmente con le medie di lega, ma con così pochi dati specifici "
                               "la previsione va presa con più cautela del solito.")

                c1, c2, c3 = st.columns(3)
                c1.metric("1 (Casa)", f"{modello['prob_1']:.1f}%")
                c2.metric("X (Pareggio)", f"{modello['prob_X']:.1f}%")
                c3.metric("2 (Trasferta)", f"{modello['prob_2']:.1f}%")

                top5 = sorted(modello['risultati'], key=lambda x: x['p'], reverse=True)[:5]
                st.write("**Top 5 risultati esatti:**")
                st.table(pd.DataFrame([{"Risultato": r['res'], "Probabilità %": f"{r['p']:.1f}"} for r in top5]))

                c1, c2 = st.columns(2)
                c1.metric("GOAL", f"{modello['prob_goal']:.1f}%")
                c2.metric("NO GOAL", f"{modello['prob_nogoal']:.1f}%")

                st.write("**Under/Over:**")
                st.table(pd.DataFrame([{"Soglia": l, "Under %": f"{v:.1f}", "Over %": f"{100-v:.1f}"}
                                        for l, v in modello['prob_under'].items()]))

                colonne_quote = classifica_colonne_quote(dati.columns)
                quote = quote_mercato_normalizzate(partita, *colonne_quote["apertura"])
                if quote:
                    st.write(f"**Analisi Valore** (media {quote['n_bookmakers']} bookmaker, margine {(quote['overround']-1)*100:.1f}%)")
                    qm_1 = 100/max(1, modello['prob_1'])
                    qm_x = 100/max(1, modello['prob_X'])
                    qm_2 = 100/max(1, modello['prob_2'])
                    sc1, msg1 = valuta_affidabilita(quote['q_casa_equa'], qm_1, modello['n_storico'])
                    scx, msgx = valuta_affidabilita(quote['q_x_equa'], qm_x, modello['n_storico'])
                    sc2, msg2 = valuta_affidabilita(quote['q_trasf_equa'], qm_2, modello['n_storico'])
                    st.table(pd.DataFrame([
                        {"Segno": "1", "Mercato": f"{quote['q_casa_equa']:.2f}", "Modello": f"{qm_1:.2f}", "Score": f"{sc1:.0f}", "Giudizio": msg1},
                        {"Segno": "X", "Mercato": f"{quote['q_x_equa']:.2f}", "Modello": f"{qm_x:.2f}", "Score": f"{scx:.0f}", "Giudizio": msgx},
                        {"Segno": "2", "Mercato": f"{quote['q_trasf_equa']:.2f}", "Modello": f"{qm_2:.2f}", "Score": f"{sc2:.0f}", "Giudizio": msg2},
                    ]))

                    # FIX APP #1 — Registro CLV personale ("impara nel tempo").
                    # Alla prima volta che guardi questa partita salviamo la quota;
                    # ogni volta dopo (anche a partita giocata) confrontiamo con quella
                    # prima rilevazione, sui TUOI tempi reali di consultazione.
                    registro = carica_registro_clv()
                    chiave = chiave_partita(id_fd, partita)
                    if chiave not in registro:
                        registro[chiave] = {
                            "q1": quote['q_casa_grezza'], "qx": quote['q_x_grezza'], "q2": quote['q_trasf_grezza'],
                            "rilevata_il": datetime.now().isoformat(timespec="minutes"),
                        }
                        salva_registro_clv(registro)
                        st.caption("📌 Prima rilevazione quote salvata per questa partita — "
                                   "al prossimo controllo vedrai qui il movimento rispetto a ora.")
                    else:
                        prima = registro[chiave]
                        mov1 = (quote['q_casa_grezza']/prima['q1'] - 1)*100
                        movx = (quote['q_x_grezza']/prima['qx'] - 1)*100
                        mov2 = (quote['q_trasf_grezza']/prima['q2'] - 1)*100
                        st.write(f"**📌 Movimento quote dalla tua prima rilevazione** ({prima['rilevata_il']}):")
                        st.table(pd.DataFrame([
                            {"Segno": "1", "Prima rilevazione": f"{prima['q1']:.2f}", "Ora": f"{quote['q_casa_grezza']:.2f}", "Variazione": f"{mov1:+.1f}%"},
                            {"Segno": "X", "Prima rilevazione": f"{prima['qx']:.2f}", "Ora": f"{quote['q_x_grezza']:.2f}", "Variazione": f"{movx:+.1f}%"},
                            {"Segno": "2", "Prima rilevazione": f"{prima['q2']:.2f}", "Ora": f"{quote['q_trasf_grezza']:.2f}", "Variazione": f"{mov2:+.1f}%"},
                        ]))
                        if pd.notna(partita.get('FTHG')):
                            st.caption("La partita è già stata giocata: questo è il movimento completo "
                                       "dalla tua prima rilevazione fino alla chiusura del mercato.")
                else:
                    st.caption("Quote di mercato non disponibili per questa partita.")

# ---------- TAB STIMA RHO ----------
with tab_rho:
    if solo_previsione:
        st.caption("ℹ️ Su questa competizione la stima ρ userà tutti i dati disponibili come train "
                   "(football-data.org non fornisce qui una stagione precedente separata) — "
                   "nessuna validazione out-of-sample per ora.")
    st.write(f"ρ attuale in uso: **{st.session_state.rho:+.3f}**")
    st.caption("Calibrato per massima verosimiglianza SOLO sulla stagione precedente (train). "
               "Se disponibile, valuta anche la generalizzazione sulla stagione corrente (test).")

    if st.button("🎯 Stima ρ ottimale"):
        with st.spinner("Stima in corso..."):
            train_df = dati[(dati['FTHG'].notna()) & (dati['Stagione'] == 'precedente')].reset_index(drop=True)
            tutte = dati[dati['FTHG'].notna()].reset_index(drop=True)

            if len(train_df) < 50:
                st.warning("Campione di train insufficiente per stimare ρ.")
            else:
                idx_start = max(15, len(train_df) - 300)
                campione_train = []
                for i in range(idx_start, len(train_df)):
                    r = train_df.iloc[i]
                    prec = train_df.iloc[:i]
                    m = calcola_modello(prec, r['HomeTeam'], r['AwayTeam'], 0.0, data_riferimento=r.get('Date_parsed'))
                    if m: campione_train.append((int(r['FTHG']), int(r['FTAG']), m['lambda_casa'], m['lambda_trasferta']))

                if len(campione_train) < 30:
                    st.warning("Campione di train insufficiente dopo il filtro.")
                else:
                    def neg_ll(rho):
                        ll = 0.0
                        for gc, gt, lc, lt in campione_train:
                            p = max(poisson.pmf(gc, lc) * poisson.pmf(gt, lt) * tau_dixon_coles(gc, gt, lc, lt, rho), 1e-10)
                            ll += np.log(p)
                        return -ll

                    ris = minimize_scalar(neg_ll, bounds=(-0.30, 0.30), method='bounded')
                    st.session_state.rho = ris.x
                    st.success(f"ρ stimato: {ris.x:+.3f} (train: {len(campione_train)} partite)")

                    idx_test = tutte.index[tutte['Stagione'] == 'corrente'].tolist()
                    idx_test = [i for i in idx_test if i >= 15]
                    if idx_test:
                        campione_test = []
                        for i in idx_test:
                            r = tutte.iloc[i]
                            prec = tutte.iloc[:i]
                            m = calcola_modello(prec, r['HomeTeam'], r['AwayTeam'], 0.0, data_riferimento=r.get('Date_parsed'))
                            if m: campione_test.append((int(r['FTHG']), int(r['FTAG']), m['lambda_casa'], m['lambda_trasferta']))
                        if campione_test:
                            ll_rho = -sum(np.log(max(poisson.pmf(gc,lc)*poisson.pmf(gt,lt)*tau_dixon_coles(gc,gt,lc,lt,ris.x),1e-10)) for gc,gt,lc,lt in campione_test)/len(campione_test)
                            ll_zero = -sum(np.log(max(poisson.pmf(gc,lc)*poisson.pmf(gt,lt)*tau_dixon_coles(gc,gt,lc,lt,0.0),1e-10)) for gc,gt,lc,lt in campione_test)/len(campione_test)
                            st.write(f"Validazione out-of-sample ({len(campione_test)} partite mai viste in train):")
                            st.write(f"- Log-verosimiglianza con ρ stimato: {-ll_rho:.4f}")
                            st.write(f"- Log-verosimiglianza con ρ=0: {-ll_zero:.4f}")
                            if -ll_rho > -ll_zero:
                                st.success("Il ρ stimato generalizza meglio di nessuna correzione.")
                            else:
                                st.warning("Il ρ stimato NON migliora sul test — possibile overfitting sul train.")
                    else:
                        st.info("Nessuna partita di stagione corrente ancora disponibile per la validazione.")

# ---------- TAB BACKTEST ----------
with tab_backtest:
    if solo_previsione:
        st.info("📊 Backtest non disponibile per le competizioni europee: football-data.org "
                "non fornisce quote di mercato, quindi non c'è nulla su cui calcolare valore o ROI.")
        st.stop()
    st.caption("Versione semplificata: solo percentuale di vittoria (nessuna gestione della puntata).")
    soglia = st.slider("Soglia score", 0, 90, 50, 10)
    usa_oos = st.checkbox("Valida SOLO su stagione corrente (out-of-sample, consigliato)", value=True)

    if st.button("📈 Esegui Backtest", type="primary"):
        with st.spinner("Backtest in corso..."):
            colonne_quote = classifica_colonne_quote(dati.columns)
            qh, qd, qa = colonne_quote["apertura"]
            ch_h, ch_d, ch_a = colonne_quote["chiusura"]
            clv_disp = bool(ch_h and ch_d and ch_a)

            tutte = dati[dati['FTHG'].notna()].reset_index(drop=True)
            ha_stagione = 'Stagione' in tutte.columns

            if usa_oos and ha_stagione:
                indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
            else:
                indici = list(range(15, len(tutte)))

            if not indici:
                st.warning("⚠️ Nessuna partita di stagione corrente disponibile per la validazione "
                            "out-of-sample. Disattiva l'opzione per un test provvisorio in-sample.")
            else:
                n_bet, n_win = 0, 0
                somma_clv, n_clv_pos, n_clv = 0.0, 0, 0

                for i in indici:
                    partita = tutte.iloc[i]
                    prec = tutte.iloc[:i]
                    m = calcola_modello(prec, partita['HomeTeam'], partita['AwayTeam'],
                                         st.session_state.rho, data_riferimento=partita.get('Date_parsed'))
                    if m is None: continue
                    quote = quote_mercato_normalizzate(partita, qh, qd, qa)
                    if quote is None: continue
                    quote_ch = quote_mercato_normalizzate(partita, ch_h, ch_d, ch_a) if clv_disp else None

                    qm1, qmx, qm2 = 100/max(1,m['prob_1']), 100/max(1,m['prob_X']), 100/max(1,m['prob_2'])
                    sc1,_ = valuta_affidabilita(quote['q_casa_equa'], qm1, m['n_storico'])
                    scx,_ = valuta_affidabilita(quote['q_x_equa'], qmx, m['n_storico'])
                    sc2,_ = valuta_affidabilita(quote['q_trasf_equa'], qm2, m['n_storico'])
                    esito = '1' if partita['FTHG']>partita['FTAG'] else ('2' if partita['FTHG']<partita['FTAG'] else 'X')

                    opzioni_bet = [
                        ('1', sc1, quote_ch['q_casa_grezza'] if quote_ch else None),
                        ('X', scx, quote_ch['q_x_grezza'] if quote_ch else None),
                        ('2', sc2, quote_ch['q_trasf_grezza'] if quote_ch else None),
                    ]
                    quote_grezze = {'1': quote['q_casa_grezza'], 'X': quote['q_x_grezza'], '2': quote['q_trasf_grezza']}
                    for segno, score, q_chiusura in opzioni_bet:
                        if score < soglia: continue
                        n_bet += 1
                        if segno == esito:
                            n_win += 1
                        if q_chiusura:
                            clv_pct = (quote_grezze[segno]/q_chiusura - 1)*100
                            somma_clv += clv_pct
                            n_clv += 1
                            if clv_pct > 0: n_clv_pos += 1

                if n_bet == 0:
                    st.warning(f"Nessuna scommessa avrebbe superato la soglia {soglia} sulle {len(indici)} partite valutate.")
                else:
                    win_rate = (n_win/n_bet)*100
                    c1, c2 = st.columns(2)
                    c1.metric("Scommesse valutate", n_bet)
                    c2.metric("Win rate", f"{win_rate:.1f}%")
                    if n_clv > 0:
                        clv_medio = somma_clv/n_clv
                        st.write(f"**CLV medio**: {clv_medio:+.2f}% ({n_clv_pos/n_clv*100:.1f}% scommesse con CLV positivo)")
                    st.caption("⚠️ Percentuale di vittoria calcolata sulle quote storiche. "
                               "Nota: le colonne PSCH/PSCD/PSCA (Pinnacle closing) sono segnalate come inaffidabili "
                               "da football-data.co.uk dal 23/07/2025 — interpreta il CLV con cautela.")

    st.divider()
    st.write("**📊 Confronta soglie diverse in un colpo solo**")
    st.caption("Se lo score porta segnale reale, salendo con la soglia il win rate/CLV dovrebbero "
               "migliorare. Se restano piatti, lo score non sta filtrando nulla di utile.")

    if st.button("📊 Confronta soglie"):
        with st.spinner("Calcolo in corso (una sola passata sui dati per tutte le soglie)..."):
            colonne_quote = classifica_colonne_quote(dati.columns)
            qh, qd, qa = colonne_quote["apertura"]
            ch_h, ch_d, ch_a = colonne_quote["chiusura"]
            clv_disp = bool(ch_h and ch_d and ch_a)

            tutte = dati[dati['FTHG'].notna()].reset_index(drop=True)
            ha_stagione = 'Stagione' in tutte.columns

            if usa_oos and ha_stagione:
                indici = [i for i in tutte.index[tutte['Stagione'] == 'corrente'].tolist() if i >= 15]
            else:
                indici = list(range(15, len(tutte)))

            SOGLIE_CONFRONTO = [0, 30, 50, 70]
            stat = {s: {"n_bet": 0, "n_win": 0, "somma_clv": 0.0, "n_clv": 0, "n_clv_pos": 0} for s in SOGLIE_CONFRONTO}

            if not indici:
                st.warning("⚠️ Nessuna partita di stagione corrente disponibile per il confronto.")
            else:
                for i in indici:
                    partita = tutte.iloc[i]
                    prec = tutte.iloc[:i]
                    m = calcola_modello(prec, partita['HomeTeam'], partita['AwayTeam'],
                                         st.session_state.rho, data_riferimento=partita.get('Date_parsed'))
                    if m is None: continue
                    quote = quote_mercato_normalizzate(partita, qh, qd, qa)
                    if quote is None: continue
                    quote_ch = quote_mercato_normalizzate(partita, ch_h, ch_d, ch_a) if clv_disp else None

                    qm1, qmx, qm2 = 100/max(1,m['prob_1']), 100/max(1,m['prob_X']), 100/max(1,m['prob_2'])
                    sc1,_ = valuta_affidabilita(quote['q_casa_equa'], qm1, m['n_storico'])
                    scx,_ = valuta_affidabilita(quote['q_x_equa'], qmx, m['n_storico'])
                    sc2,_ = valuta_affidabilita(quote['q_trasf_equa'], qm2, m['n_storico'])
                    esito = '1' if partita['FTHG']>partita['FTAG'] else ('2' if partita['FTHG']<partita['FTAG'] else 'X')

                    quote_grezze = {'1': quote['q_casa_grezza'], 'X': quote['q_x_grezza'], '2': quote['q_trasf_grezza']}
                    quote_chiusura = {
                        '1': quote_ch['q_casa_grezza'] if quote_ch else None,
                        'X': quote_ch['q_x_grezza'] if quote_ch else None,
                        '2': quote_ch['q_trasf_grezza'] if quote_ch else None,
                    }
                    score_per_segno = {'1': sc1, 'X': scx, '2': sc2}

                    for segno, score in score_per_segno.items():
                        vinta = (segno == esito)
                        q_chiusura = quote_chiusura[segno]
                        for s in SOGLIE_CONFRONTO:
                            if score < s: continue
                            stat[s]["n_bet"] += 1
                            if vinta: stat[s]["n_win"] += 1
                            if q_chiusura:
                                clv_pct = (quote_grezze[segno]/q_chiusura - 1)*100
                                stat[s]["somma_clv"] += clv_pct
                                stat[s]["n_clv"] += 1
                                if clv_pct > 0: stat[s]["n_clv_pos"] += 1

                righe = []
                for s in SOGLIE_CONFRONTO:
                    d = stat[s]
                    win_rate = (d["n_win"]/d["n_bet"]*100) if d["n_bet"] > 0 else None
                    clv_medio = (d["somma_clv"]/d["n_clv"]) if d["n_clv"] > 0 else None
                    righe.append({
                        "Soglia": s,
                        "Scommesse": d["n_bet"],
                        "Win rate": f"{win_rate:.1f}%" if win_rate is not None else "—",
                        "CLV medio": f"{clv_medio:+.2f}%" if clv_medio is not None else "—",
                    })
                st.table(pd.DataFrame(righe))
                st.caption("Nota: a soglie alte il numero di scommesse cala molto — con pochi casi "
                           "un win rate/CLV migliore può essere anche solo rumore statistico, non "
                           "necessariamente un segnale affidabile. Guarda anche quante scommesse restano.")
