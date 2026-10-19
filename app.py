from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
import streamlit as st

st.set_page_config(
    page_title="Global Football Value & Kelly Engine",
    page_icon="⚽",
    layout="wide",
)

LEAGUE_CODES = {
    "UEFA Champions League (Cross-League Pool)": "UCL",
    "UEFA Europa League & Conference (Elite Pool)": "UELE",
    "English Premier League (EPL)": "E0",
    "English Championship": "E1",
    "Spanish La Liga": "SP1",
    "Italian Serie A": "I1",
    "German Bundesliga": "D1",
    "French Ligue 1": "F1",
    "Greek Super League": "G1",
    "Dutch Eredivisie": "N1",
    "Portuguese Primeira Liga": "P1",
    "Custom / Regional Division (CSV Upload)": "CUSTOM",
}


@st.cache_data(ttl=3600)
def load_global_football_data(league_code, seasons, uploaded_file=None):
  if league_code == "CUSTOM":
    if uploaded_file is not None:
      try:
        df = pd.read_csv(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return (
            df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
      except Exception as e:
        st.error(f"Error parsing uploaded CSV: {e}")
        return pd.DataFrame()
    return pd.DataFrame()

  if league_code == "UCL":
    codes = ["E0", "SP1", "I1", "D1", "F1", "P1", "N1"]
  elif league_code == "UELE":
    codes = ["E1", "SP2", "I2", "D2", "G1", "P1"]
  else:
    codes = [league_code]

  frames = []
  for code in codes:
    for s in seasons:
      url = f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv"
      try:
        df = pd.read_csv(url, encoding="latin1")
        cols = [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "HTHG",
            "HTAG",
            "HST",
            "AST",
        ]
        available = [c for c in cols if c in df.columns]
        df = df[available].dropna(
            subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        )
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        frames.append(df.dropna(subset=["Date"]))
      except Exception:
        pass
  return (
      pd.concat(frames, ignore_index=True)
      .sort_values("Date")
      .reset_index(drop=True)
      if frames
      else pd.DataFrame()
  )


class GlobalDixonColesPredictor:

  def __init__(self, xi=0.0035):
    self.xi = xi

  def _fit_dc(self, df, h_col, a_col, teams, team_idx, use_rho=True):
    n = len(teams)
    h_i = df["HomeTeam"].map(team_idx).values
    a_i = df["AwayTeam"].map(team_idx).values
    x = df[h_col].values.astype(int)
    y = df[a_col].values.astype(int)
    w = df["weight"].values
    init = np.concatenate(
        [np.zeros(n), np.zeros(n), [0.25, -0.08 if use_rho else 0.0]]
    )

    def nll(params):
      alpha = params[:n]
      beta = params[n : 2 * n]
      gamma = params[-2]
      rho = params[-1] if use_rho else 0.0
      lam = np.exp(alpha[h_i] + beta[a_i] + gamma)
      mu = np.exp(alpha[a_i] + beta[h_i])
      tau = np.ones_like(lam)
      if use_rho:
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-6, None)
      log_lik = (
          np.log(tau)
          - lam
          + x * np.log(lam)
          - gammaln(x + 1)
          - mu
          + y * np.log(mu)
          - gammaln(y + 1)
      )
      return -np.sum(w * log_lik) + 100.0 * (np.sum(alpha) ** 2)

    bounds = [(None, None)] * (2 * n) + [
        (0.0, 1.0),
        (-0.25, 0.25) if use_rho else (0.0, 0.0),
    ]
    res = minimize(nll, init, method="L-BFGS-B", bounds=bounds)
    return {
        "alpha": res.x[:n],
        "beta": res.x[n : 2 * n],
        "gamma": res.x[-2],
        "rho": res.x[-1] if use_rho else 0.0,
    }

  def predict(self, df, home, away, kickoff_dt):
    hist = df[df["Date"] < kickoff_dt].copy()
    if hist.empty:
      raise ValueError("No historical matches prior to this date.")
    teams = sorted(list(set(hist["HomeTeam"]).union(set(hist["AwayTeam"]))))
    team_idx = {t: i for i, t in enumerate(teams)}
    if home not in team_idx or away not in team_idx:
      raise ValueError(f"Teams '{home}' or '{away}' not found in dataset.")

    hist["weight"] = np.exp(
        -self.xi * (kickoff_dt - hist["Date"]).dt.days.clip(lower=0)
    )
    if "HST" in hist.columns and "AST" in hist.columns:
      hist["Eff_HG"] = np.round(
          0.70 * hist["FTHG"] + 0.30 * (hist["HST"].fillna(hist["FTHG"] * 3) * 0.32)
      )
      hist["Eff_AG"] = np.round(
          0.70 * hist["FTAG"] + 0.30 * (hist["AST"].fillna(hist["FTAG"] * 3) * 0.32)
      )
    else:
      hist["Eff_HG"], hist["Eff_AG"] = hist["FTHG"], hist["FTAG"]

    ft_p = self._fit_dc(hist, "Eff_HG", "Eff_AG", teams, team_idx, use_rho=True)
    ht_hist = hist.dropna(subset=["HTHG", "HTAG"])
    ht_p = self._fit_dc(
        ht_hist, "HTHG", "HTAG", teams, team_idx, use_rho=False
    )

    h_i, a_i = team_idx[home], team_idx[away]
    lam = np.exp(ft_p["alpha"][h_i] + ft_p["beta"][a_i] + ft_p["gamma"])
    mu = np.exp(ft_p["alpha"][a_i] + ft_p["beta"][h_i])
    lam_ht = np.exp(ht_p["alpha"][h_i] + ht_p["beta"][a_i] + ht_p["gamma"])
    mu_ht = np.exp(ht_p["alpha"][a_i] + ht_p["beta"][h_i])

    max_g = 8
    goals = np.arange(max_g)
    mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    rho = ft_p["rho"]
    mat[0, 0] *= 1.0 - lam * mu * rho
    mat[0, 1] *= 1.0 + lam * rho
    mat[1, 0] *= 1.0 + mu * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 0, None) / mat.sum()

    p_home, p_draw, p_away = (
        np.sum(np.tril(mat, -1)),
        np.sum(np.diag(mat)),
        np.sum(np.triu(mat, 1)),
    )
    p_over25 = np.sum(mat[np.add.outer(goals, goals) > 2.5])
    p_btts = np.sum(mat[1:, 1:])
    p_ht_over05 = 1.0 - np.exp(-(lam_ht + mu_ht))

    top_scores = []
    flat_idx = np.argsort(mat.ravel())[::-1]
    for idx in flat_idx[:15]:  # Top 15 scorelines for comprehensive chart
      hg = idx // max_g
      ag = idx % max_g
      prob = mat.ravel()[idx]
      top_scores.append({
          "Scoreline": f"{hg} - {ag}",
          "Probability (%)": round(prob * 100, 2),
          "Fair Odds": round(1.0 / prob, 2) if prob > 0 else 99.9,
      })

    return {
        "home_xg": lam,
        "away_xg": mu,
        "markets": {
            "Home Win (1)": p_home,
            "Draw (X)": p_draw,
            "Away Win (2)": p_away,
            "Over 2.5 Goals": p_over25,
            "BTTS (Goal)": p_btts,
            "1H Over 0.5": p_ht_over05,
        },
        "score_matrix": top_scores,
    }


st.title("🌍 Global Football Value & Kelly Engine")
with st.sidebar:
  league_name = st.selectbox(
      "Select Competition / League", list(LEAGUE_CODES.keys())
  )
  league_code = LEAGUE_CODES[league_name]

  uploaded_file = None
  if league_code == "CUSTOM":
    st.markdown("### Regional / Custom CSV")
    uploaded_file = st.file_uploader(
        "Upload CSV (Date, HomeTeam, AwayTeam, FTHG, FTAG)", type=["csv"]
    )
    seasons = []
  else:
    seasons = st.multiselect(
        "Seasons", ["2526", "2627"], default=["2526", "2627"]
    )

  bankroll = st.number_input("Bankroll (€)", value=1000.0)
  kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.25)

df_data = load_global_football_data(league_code, tuple(seasons), uploaded_file)

if not df_data.empty:
  teams = sorted(
      list(set(df_data["HomeTeam"]).union(set(df_data["AwayTeam"])))
  )
  c1, c2, c3 = st.columns(3)
  home, away = c1.selectbox("Home Team", teams), c2.selectbox(
      "Away Team", teams, index=1 if len(teams) > 1 else 0
  )
  m_date = c3.date_input("Match Date", value=datetime.today())

  if st.button("Run Global Prediction", type="primary"):
    try:
      res = GlobalDixonColesPredictor().predict(
          df_data, home, away, datetime.combine(m_date, datetime.min.time())
      )
      st.success("Model executed successfully!")
      cols = st.columns(2)
      cols[0].metric("Home xG", f"{res['home_xg']:.2f}")
      cols[1].metric("Away xG", f"{res['away_xg']:.2f}")

      st.markdown("### 📊 Market Value & Kelly Staking Table")
      markets = [
          {
              "Market": k,
              "Prob (%)": round(v * 100, 2),
              "Fair Odds": round(1 / v, 2),
              "Bookie Odds": round((1 / v) * 1.08, 2),
          }
          for k, v in res["markets"].items()
      ]
      edited = st.data_editor(pd.DataFrame(markets), hide_index=True)

      recs = []
      for _, r in edited.iterrows():
        p, b = r["Prob (%)"] / 100.0, r["Bookie Odds"]
        ev = (p * b) - 1.0
        stake = (
            max(
                0.0,
                (((p * b - 1.0) / (b - 1.0)) * kelly_fraction * bankroll),
            )
            if b > 1
            else 0
        )
        recs.append({
            "Market": r["Market"],
            "EV": f"{ev*100:+.2f}%",
            "Stake": f"€{stake:.2f}",
            "Action": "✅ BET" if ev > 0 else "❌ SKIP",
        })
      st.table(pd.DataFrame(recs))

      st.markdown("### 🎯 Exact Score Probabilities Chart")
      st.markdown(
          "Calculated probabilities and fair decimal odds for the most likely"
          " match scorelines based on the Dixon-Coles model:"
      )

      score_df = pd.DataFrame(res["score_matrix"])
      st.dataframe(score_df, hide_index=True, use_container_width=True)

      # Visual bar chart representation inside streamlit
      chart_data = score_df.set_index("Scoreline")["Probability (%)"]
      st.bar_chart(chart_data)

    except Exception as e:
      st.error(f"Prediction Error: {e}")
elif league_code == "CUSTOM" and uploaded_file is None:
  st.info("Please upload a local division CSV file via the sidebar to begin.")
