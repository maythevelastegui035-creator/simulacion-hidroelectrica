from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
import streamlit as st

st.set_page_config(
    page_title="Simulador Hidroeléctrico — Alertas Tempranas",
    layout="wide",
)

st.title("Simulador de Central Hidroeléctrica Simplificada")
st.caption(
    "Continuidad, Bernoulli generalizada, Darcy-Weisbach (Swamee-Jain) e indicadores de alerta temprana"
)


@dataclass
class ParametrosCentral:
    Q: float
    Hb: float
    L: float
    D: float
    eps: float
    rho: float
    mu: float
    g: float
    sum_K: float
    eta_t: float
    eta_g: float


def factor_friccion_darcy(Re: float, eps: float, D: float) -> tuple[float, str]:
    if Re <= 0:
        return 0.0, "Sin flujo"
    if Re < 2300:
        return 64.0 / Re, "Laminar"
    f_calc = 0.25 / (np.log10(eps / (3.7 * D) + 5.74 / (Re ** 0.9)) ** 2)
    regimen = "Transición" if Re <= 4000 else "Turbulento"
    return float(f_calc), regimen


def simular_escenario(p: ParametrosCentral) -> dict:
    area = np.pi * (p.D ** 2) / 4.0
    vel = p.Q / area
    re = (p.rho * vel * p.D) / p.mu
    f, regimen = factor_friccion_darcy(re, p.eps, p.D)

    carga_cinetica = (vel ** 2) / (2.0 * p.g)
    hf = f * (p.L / p.D) * carga_cinetica
    hm = p.sum_K * carga_cinetica
    hL = hf + hm

    Hn = p.Hb - hL
    P2_man_pa = p.rho * p.g * (Hn - carga_cinetica)

    Ph_w = p.rho * p.g * p.Q * Hn
    Pe_w = p.eta_t * p.eta_g * Ph_w

    return {
        "Q_m3s": p.Q,
        "Hb_m": p.Hb,
        "D_m": p.D,
        "A_m2": area,
        "v_ms": vel,
        "Re": re,
        "regimen": regimen,
        "f": f,
        "hf_m": hf,
        "hm_m": hm,
        "hL_m": hL,
        "Hn_m": Hn,
        "P2_man_kPa": P2_man_pa / 1000.0,
        "Ph_MW": Ph_w / 1e6,
        "Pe_MW": Pe_w / 1e6,
    }


def clasificar_indicador(valor_pct: float, tipo: str) -> str:
    if tipo in ("Q", "H"):
        if valor_pct >= 90.0:
            return "Normal"
        elif valor_pct >= 80.0:
            return "Advertencia"
        return "Crítica"
    elif tipo == "P":
        if valor_pct >= 85.0:
            return "Normal"
        elif valor_pct >= 75.0:
            return "Advertencia"
        return "Crítica"
    elif tipo == "L":
        if valor_pct <= 120.0:
            return "Normal"
        elif valor_pct <= 150.0:
            return "Advertencia"
        return "Crítica"
    raise ValueError(f"Indicador desconocido: {tipo}")


def evaluar_alertas(res: dict, ref_nom: dict) -> dict:
    IQ = (res["Q_m3s"] / ref_nom["Q_m3s"]) * 100.0 if ref_nom["Q_m3s"] else 0.0
    IH = (res["Hn_m"] / ref_nom["Hn_m"]) * 100.0 if ref_nom["Hn_m"] else 0.0
    IP = (res["Ph_MW"] / ref_nom["Ph_MW"]) * 100.0 if ref_nom["Ph_MW"] else 0.0
    IL = (res["hL_m"] / ref_nom["hL_m"]) * 100.0 if ref_nom["hL_m"] else 0.0

    estados = {
        "IQ": clasificar_indicador(IQ, "Q"),
        "IH": clasificar_indicador(IH, "H"),
        "IP": clasificar_indicador(IP, "P"),
        "IL": clasificar_indicador(IL, "L"),
    }
    severidad = {"Normal": 0, "Advertencia": 1, "Crítica": 2}
    estado_global = max(estados.values(), key=lambda s: severidad[s])

    return {
        "IQ": (IQ, estados["IQ"]),
        "IH": (IH, estados["IH"]),
        "IP": (IP, estados["IP"]),
        "IL": (IL, estados["IL"]),
        "global": estado_global,
    }


def dibujar_esquema_central(res: dict, estado_global: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    nivel_agua = float(np.clip(1.0 + (res["Hb_m"] / 150.0) * 2.6, 1.0, 3.8))

    presa = Polygon(
        [[0.0, 0.0], [2.4, 0.0], [2.4, 4.2], [1.4, 4.2], [0.0, 4.2]],
        closed=True,
        facecolor="#E5E7EB",
        edgecolor="#6B7280",
        lw=1.5,
    )
    ax.add_patch(presa)

    agua = Rectangle(
        (0.05, 0.4),
        1.55,
        nivel_agua,
        facecolor="#93C5FD",
        edgecolor="#3B82F6",
        lw=1.2,
        alpha=0.85,
    )
    ax.add_patch(agua)

    grosor_tubo = float(np.clip(0.18 + (res["D_m"] - 0.5) * 0.14, 0.16, 0.45))
    tubo = Polygon(
        [
            [1.6, 0.8 + grosor_tubo],
            [6.2, 0.35 + grosor_tubo],
            [6.2, 0.35],
            [1.6, 0.8],
        ],
        closed=True,
        facecolor="#BFDBFE",
        edgecolor="#1D4ED8",
        lw=1.5,
    )
    ax.add_patch(tubo)

    casa = FancyBboxPatch(
        (6.2, 0.1),
        2.2,
        2.1,
        boxstyle="round,pad=0.05",
        facecolor="#F3F4F6",
        edgecolor="#4B5563",
        lw=1.5,
    )
    ax.add_patch(casa)

    turbina = plt.Circle((7.2, 0.85), 0.45, facecolor="#A7F3D0", edgecolor="#059669", lw=1.8)
    ax.add_patch(turbina)

    ax.text(0.82, 0.4 + nivel_agua + 0.15, f"Embalse\nHb = {res['Hb_m']:.1f} m", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(3.9, 1.35, f"Q = {res['Q_m3s']:.2f} m³/s\nv = {res['v_ms']:.2f} m/s\nhL = {res['hL_m']:.2f} m", ha="center", fontsize=8.5, bbox=dict(boxstyle="round", fc="#FFFBEB", ec="#F59E0B", alpha=0.9))
    ax.text(7.2, 1.55, f"Turbina\nHn = {res['Hn_m']:.2f} m\nP2 = {res['P2_man_kPa']:.1f} kPa\nPe = {res['Pe_MW']:.2f} MW", ha="center", va="bottom", fontsize=8, bbox=dict(boxstyle="round", fc="#ECFDF5", ec="#10B981", alpha=0.9))

    color_badge = {"Normal": "#10B981", "Advertencia": "#F59E0B", "Crítica": "#EF4444"}[estado_global]
    ax.text(8.3, 4.05, f"Estado: {estado_global}", ha="right", va="top", fontsize=9.5, fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.35", fc=color_badge, ec="none"))

    ax.set_xlim(-0.2, 8.8)
    ax.set_ylim(-0.2, 4.5)
    ax.axis("off")
    plt.tight_layout()
    return fig


with st.sidebar:
    st.header("Parámetros de operación")
    Q_in = st.number_input("Caudal de operación Q (m³/s)", min_value=0.1, max_value=100.0, value=6.0, step=0.5)
    Hb_in = st.number_input("Altura bruta Hb (m)", min_value=5.0, max_value=500.0, value=100.0, step=5.0)

    st.divider()
    st.subheader("Condición nominal de referencia")
    Q_nom = st.number_input("Caudal nominal Q0 (m³/s)", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
    Hb_nom = st.number_input("Altura bruta nominal Hb0 (m)", min_value=5.0, max_value=500.0, value=100.0, step=5.0)

    with st.expander("Geometría, fluido y eficiencias"):
        L_in = st.number_input("Longitud tubería L (m)", min_value=1.0, max_value=10000.0, value=500.0, step=10.0)
        D_in = st.number_input("Diámetro interno D (m)", min_value=0.1, max_value=15.0, value=2.0, step=0.1)
        eps_in = st.number_input("Rugosidad absoluta ε (m)", min_value=1e-6, max_value=0.05, value=0.00015, format="%.6f")
        sum_K_in = st.number_input("Pérdidas localizadas ΣK (-)", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        eta_t_in = st.slider("Eficiencia turbina ηt", min_value=0.10, max_value=1.00, value=0.90, step=0.01)
        eta_g_in = st.slider("Eficiencia generador ηg", min_value=0.10, max_value=1.00, value=0.95, step=0.01)
        rho_in = st.number_input("Densidad ρ (kg/m³)", min_value=500.0, max_value=1500.0, value=1000.0, step=10.0)
        mu_in = st.number_input("Viscosidad dinámica μ (Pa·s)", min_value=1e-5, max_value=0.1, value=0.001, format="%.5f")
        g_in = st.number_input("Gravedad g (m/s²)", min_value=1.0, max_value=20.0, value=9.81, step=0.01)


p_actual = ParametrosCentral(
    Q=Q_in, Hb=Hb_in, L=L_in, D=D_in, eps=eps_in,
    rho=rho_in, mu=mu_in, g=g_in, sum_K=sum_K_in,
    eta_t=eta_t_in, eta_g=eta_g_in
)
p_nominal = ParametrosCentral(
    Q=Q_nom, Hb=Hb_nom, L=L_in, D=D_in, eps=eps_in,
    rho=rho_in, mu=mu_in, g=g_in, sum_K=sum_K_in,
    eta_t=eta_t_in, eta_g=eta_g_in
)

res_act = simular_escenario(p_actual)
res_nom = simular_escenario(p_nominal)
alertas = evaluar_alertas(res_act, res_nom)

estado_glob = alertas["global"]
if estado_glob == "Normal":
    st.success(f"Estado global del sistema: **{estado_glob}**")
elif estado_glob == "Advertencia":
    st.warning(f"Estado global del sistema: **{estado_glob}**")
else:
    st.error(f"Estado global del sistema: **{estado_glob}**")


# ==========================================
# RESULTADOS COMPLETOS DEL SISTEMA
# ==========================================

st.subheader("Resultados hidráulicos y energéticos")

# Primera fila
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Velocidad media v",
    f"{res_act['v_ms']:.3f} m/s"
)

c2.metric(
    "Número de Reynolds",
    f"{res_act['Re']:.2e}"
)

c3.metric(
    "Factor de fricción f",
    f"{res_act['f']:.5f}"
)

c4.metric(
    "Régimen de flujo",
    res_act["regimen"]
)

# Segunda fila
c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Pérdidas por fricción hf",
    f"{res_act['hf_m']:.3f} m"
)

c6.metric(
    "Pérdidas localizadas hm",
    f"{res_act['hm_m']:.3f} m"
)

c7.metric(
    "Pérdidas totales hL",
    f"{res_act['hL_m']:.3f} m"
)

c8.metric(
    "Altura neta Hn",
    f"{res_act['Hn_m']:.3f} m"
)

# Tercera fila
c9, c10, c11 = st.columns(3)

c9.metric(
    "Presión manométrica P2",
    f"{res_act['P2_man_kPa']:.2f} kPa"
)

c10.metric(
    "Potencia hidráulica Ph",
    f"{res_act['Ph_MW']:.3f} MW"
)

c11.metric(
    "Potencia eléctrica Pe",
    f"{res_act['Pe_MW']:.3f} MW"
)



col_esq, col_ind = st.columns([1.35, 1.0])

with col_esq:

    st.subheader("Esquema del sistema")

    st.image(
        "hidro.jpeg",
        use_container_width=True
    )

    st.caption(
        f"Caudal de operación: {res_act['Q_m3s']:.2f} m³/s | "
        f"Altura bruta: {res_act['Hb_m']:.2f} m | "
        f"Altura neta: {res_act['Hn_m']:.2f} m"
    )


with col_ind:
    st.subheader("Indicadores de alerta temprana")
    df_ind = pd.DataFrame([
        {"Indicador": "Caudal (IQ)", "Valor (%)": round(alertas["IQ"][0], 2), "Estado": alertas["IQ"][1]},
        {"Indicador": "Altura neta (IH)", "Valor (%)": round(alertas["IH"][0], 2), "Estado": alertas["IH"][1]},
        {"Indicador": "Potencia hid. (IP)", "Valor (%)": round(alertas["IP"][0], 2), "Estado": alertas["IP"][1]},
        {"Indicador": "Pérdidas carga (IL)", "Valor (%)": round(alertas["IL"][0], 2), "Estado": alertas["IL"][1]},
    ])
    st.dataframe(df_ind, use_container_width=True, hide_index=True)


# ==========================================
# CURVAS DE SENSIBILIDAD
# ==========================================

st.divider()
st.subheader("Curvas de sensibilidad del sistema")

# ------------------------------------------
# 1. BARRIDO DE CAUDAL
# ------------------------------------------

q_max_plot = max(15.0, Q_nom * 1.4, Q_in * 1.2)

Q_grid = np.linspace(0.2, q_max_plot, 150)

Pe_grid = []
Hn_grid = []
hf_grid = []
hm_grid = []
hL_grid = []
P2_grid = []
f_grid = []
Re_grid = []

for q_val in Q_grid:

    r_tmp = simular_escenario(ParametrosCentral(
        Q=float(q_val), Hb=Hb_in, L=L_in, D=D_in,
        eps=eps_in, rho=rho_in, mu=mu_in, g=g_in,
        sum_K=sum_K_in, eta_t=eta_t_in, eta_g=eta_g_in
    ))

    Pe_grid.append(r_tmp["Pe_MW"])
    Hn_grid.append(r_tmp["Hn_m"])
    hf_grid.append(r_tmp["hf_m"])
    hm_grid.append(r_tmp["hm_m"])
    hL_grid.append(r_tmp["hL_m"])
    P2_grid.append(r_tmp["P2_man_kPa"])
    f_grid.append(r_tmp["f"])
    Re_grid.append(r_tmp["Re"])


# ------------------------------------------
# 2. BARRIDO DE ALTURA BRUTA
# ------------------------------------------

Hb_grid = np.linspace(5, 500, 150)

Pe_Hb_grid = []

for hb_val in Hb_grid:

    r_tmp = simular_escenario(ParametrosCentral(
        Q=Q_in, Hb=float(hb_val), L=L_in, D=D_in,
        eps=eps_in, rho=rho_in, mu=mu_in, g=g_in,
        sum_K=sum_K_in, eta_t=eta_t_in, eta_g=eta_g_in
    ))

    Pe_Hb_grid.append(r_tmp["Pe_MW"])


# ------------------------------------------
# 3. FIGURAS
# ------------------------------------------

fig_sens, axes = plt.subplots(2, 3, figsize=(16, 9))

fig_sens.patch.set_facecolor("white")

ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()


# POTENCIA ELECTRICA VS CAUDAL

ax1.plot(Q_grid, Pe_grid, lw=2, label="Pe")
ax1.scatter(Q_in, res_act["Pe_MW"], zorder=5, label="Punto actual")

ax1.set_title("Potencia eléctrica vs. Caudal")
ax1.set_xlabel("Q (m³/s)")
ax1.set_ylabel("Pe (MW)")
ax1.grid(True, alpha=0.3)
ax1.legend()


# ALTURA NETA VS CAUDAL

ax2.plot(Q_grid, Hn_grid, lw=2, label="Hn")
ax2.scatter(Q_in, res_act["Hn_m"], zorder=5, label="Punto actual")

ax2.set_title("Altura neta vs. Caudal")
ax2.set_xlabel("Q (m³/s)")
ax2.set_ylabel("Hn (m)")
ax2.grid(True, alpha=0.3)
ax2.legend()


# PERDIDAS DE CARGA VS CAUDAL

ax3.plot(Q_grid, hf_grid, lw=2, label="hf")
ax3.plot(Q_grid, hm_grid, lw=2, label="hm")
ax3.plot(Q_grid, hL_grid, lw=2, label="hL")

ax3.scatter(Q_in, res_act["hL_m"], zorder=5)

ax3.set_title("Pérdidas de carga vs. Caudal")
ax3.set_xlabel("Q (m³/s)")
ax3.set_ylabel("Pérdidas (m)")
ax3.grid(True, alpha=0.3)
ax3.legend()


# PRESION MANOMETRICA VS CAUDAL

ax4.plot(Q_grid, P2_grid, lw=2, label="P2")
ax4.scatter(Q_in, res_act["P2_man_kPa"], zorder=5,
            label="Punto actual")

ax4.set_title("Presión manométrica vs. Caudal")
ax4.set_xlabel("Q (m³/s)")
ax4.set_ylabel("P2 (kPa)")
ax4.grid(True, alpha=0.3)
ax4.legend()


# FACTOR DE FRICCION VS CAUDAL

ax5.plot(Q_grid, f_grid, lw=2, label="f")
ax5.scatter(Q_in, res_act["f"], zorder=5,
            label="Punto actual")

ax5.set_title("Factor de fricción vs. Caudal")
ax5.set_xlabel("Q (m³/s)")
ax5.set_ylabel("f")
ax5.grid(True, alpha=0.3)
ax5.legend()


# POTENCIA ELECTRICA VS ALTURA BRUTA

ax6.plot(Hb_grid, Pe_Hb_grid, lw=2, label="Pe")
ax6.scatter(Hb_in, res_act["Pe_MW"], zorder=5,
            label="Punto actual")

ax6.set_title("Potencia eléctrica vs. Altura bruta")
ax6.set_xlabel("Hb (m)")
ax6.set_ylabel("Pe (MW)")
ax6.grid(True, alpha=0.3)
ax6.legend()


# ------------------------------------------
# 4. FORMATO FINAL
# ------------------------------------------

plt.tight_layout()

st.pyplot(fig_sens, use_container_width=True)

plt.close(fig_sens)
