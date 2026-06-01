"""
=============================================================================
BFP450 — Diseño de amplificador @ 2.2 GHz
Tablas necesarias para los requerimientos del circuito
(adaptación a Rgen = RL = 50 Ω + síntesis de microtiras)
=============================================================================
Salida:
    Tabla 1 ▸ Parámetros S y figuras de mérito por polarización
    → El usuario elige VCE / IC interactivamente
    Tabla 2 ▸ Impedancias de adaptación en el punto elegido
    Tabla 3 ▸ Microtiras de la red de adaptación

Dependencias:  pip install scikit-rf pandas numpy
Uso:           colocar los .s2p en UPLOAD_DIR y ejecutar.
=============================================================================
"""

import cmath
import math
import os
import re
import pandas as pd
import numpy as np
import json
import skrf as rf

# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────────
UPLOAD_DIR = r"C:\Users\cirov\OneDrive\Documents\MATERIAS\EA3\TP2\s2p"
F_DESIGN   = 2.2      # GHz — frecuencia de diseño
Z0         = 50.0         # Ω   — impedancia característica (= Rgen = RL)
ER         = 4        # FR-4
H_MM       = 1.57         # mm
T_UM       = 27.5        # µm
IC_MIN_MA  = 10           # mA — descartar puntos por debajo

L_CHOKE_NH   = 100.0   # nH — inductancia de choque deseada
C_DECOUP_PF  =  10.0   # pF — capacitor de desacople deseado
Z0_CHOKE_MIN, Z0_CHOKE_MAX   = 100.0, 250.0   # Ω — rango Z0 microtira choke
Z0_DECOUP_MIN, Z0_DECOUP_MAX =  20.0,  60.0   # Ω — rango Z0 microtira desacople


# ─── FUNCIONES AUXILIARES ───────────────────────────────────────────────────
def load_s2p(path, f_ghz):
    ntwk = rf.Network(path)
    idx = np.argmin(np.abs(ntwk.f - f_ghz * 1e9))
    s = ntwk.s[idx]
    return s[0, 0], s[1, 0], s[0, 1], s[1, 1]

def analyze(S11, S21, S12, S22, Z0=50.0):
    """Estabilidad y adaptación simultánea conjugada (Gonzalez/Pozar)."""
    delta = S11 * S22 - S12 * S21
    K  = (1 - abs(S11)**2 - abs(S22)**2 + abs(delta)**2) / (2 * abs(S12 * S21))
    B1 = 1 + abs(S11)**2 - abs(S22)**2 - abs(delta)**2
    B2 = 1 + abs(S22)**2 - abs(S11)**2 - abs(delta)**2
    C1 = S11 - delta * S22.conjugate()
    C2 = S22 - delta * S11.conjugate()

    # ── Adaptación simultánea conjugada (signo CORRECTO):
    #   B>0 → restar √   |   B<0 → sumar √   (asegura |Γ|<1)
    sign_s = 1 if B1 > 0 else -1
    sign_l = 1 if B2 > 0 else -1
    disc1  = B1**2 - 4 * abs(C1)**2
    disc2  = B2**2 - 4 * abs(C2)**2

    Gs = (B1 - sign_s * cmath.sqrt(disc1)) / (2 * C1)   # Γs óptimo
    GL = (B2 - sign_l * cmath.sqrt(disc2)) / (2 * C2)   # ΓL óptimo

    # Impedancias que la red de adaptación debe presentar al transistor
    def g2Z(g): return Z0 * (1 + g) / (1 - g)
    Zs, ZL = g2Z(Gs), g2Z(GL)

    # Impedancias del transistor (con adaptación lograda → Zin = Zs*, Zout = ZL*)
    Gin  = S11 + S12 * S21 * GL / (1 - S22 * GL)
    Gout = S22 + S12 * S21 * Gs / (1 - S11 * Gs)
    Zin, Zout = g2Z(Gin), g2Z(Gout)

    # Ganancias
    GT_n = (1 - abs(Gs)**2) * abs(S21)**2 * (1 - abs(GL)**2)
    GT_d = abs((1 - S11*Gs)*(1 - S22*GL) - S12*S21*Gs*GL)**2
    GT   = GT_n / GT_d
    MAG  = abs(S21)/abs(S12) * (K - math.sqrt(max(K**2 - 1, 0))) if K > 1 else None
    MSG  = abs(S21) / abs(S12)

    estable = (K > 1) and (abs(delta) < 1) and (disc1 > 0) and (disc2 > 0)

    return dict(
        S11=S11, S21=S21, S12=S12, S22=S22,
        K=K, delta=abs(delta), estable=estable,
        Zs=Zs, ZL=ZL, Zin=Zin, Zout=Zout,
        GT_dB =10 * math.log10(GT) if GT > 0 else float('nan'),
        MAG_dB=10 * math.log10(MAG) if MAG else float('nan'),
        MSG_dB=10 * math.log10(MSG),
        S11_dB=20 * math.log10(abs(S11)),
        S21_dB=20 * math.log10(abs(S21)),
        S12_dB=20 * math.log10(abs(S12)),
        S22_dB=20 * math.log10(abs(S22)),
    )


def microstrip(Zc, er=4.4, h_mm=1.6, f_ghz=2.2):
    """Hammerstad: ancho W [mm] y largo λ/4 [mm] para impedancia Zc [Ω]."""
    h = h_mm * 1e-3
    f = f_ghz * 1e9
    c = 3e8
    A = (Zc / 60) * math.sqrt((er + 1) / 2) + ((er - 1) / (er + 1)) * (0.23 + 0.11 / er)
    B = 377 * math.pi / (2 * Zc * math.sqrt(er))
    Wh_A = 8 * math.exp(A) / (math.exp(2 * A) - 2)
    Wh_B = (2 / math.pi) * (B - 1 - math.log(2 * B - 1)
                            + (er - 1) / (2 * er) * (math.log(B - 1) + 0.39 - 0.61 / er))
    Wh   = Wh_A if Wh_A < 2 else Wh_B
    W    = Wh * h
    er_eff = ((er + 1)/2 + (er - 1)/2 * (1 / math.sqrt(1 + 12 / Wh))) if Wh >= 1 else \
             ((er + 1)/2 + (er - 1)/2 * (1 / math.sqrt(1 + 12 / Wh) + 0.04 * (1 - Wh)**2))
    lam = c / (f * math.sqrt(er_eff))
    return W * 1e3, lam / 4 * 1e3, er_eff


def stub_length_mm(Xc, Z0_stub, er_eff, f_ghz):
    """Largo físico [mm] del stub que sintetiza la reactancia Xc.
    l = arccot(Xc/Z0) · λg / (2π),  con λg = c/(f·√ε_eff).
    arccot(x) = atan2(1, x) → resultado ∈ (0, π).
    """
    lam_g = 3e8 / (f_ghz * 1e9 * math.sqrt(er_eff))
    theta  = math.atan2(1.0, Xc / Z0_stub)
    return theta * lam_g / (2 * math.pi) * 1e3


def bias_stub(value_SI, is_inductor, Z0_ohm, f_ghz, er, h_mm):
    """
    Microtira para red de polarización.
    Choke   (stub CC shunt): Z_in = jZ0·tan(θ) = jωL  → θ = arctan(ωL/Z0)
    Desacop (stub CA shunt): Z_in = −j/(ωC)           → θ = arctan(ωC·Z0)
    Retorna: (W_mm, l_mm, theta_deg, X_ohm)
    """
    omega = 2 * math.pi * f_ghz * 1e9
    W_mm, lam4_mm, _ = microstrip(Z0_ohm, er, h_mm, f_ghz)
    lam_g_mm = lam4_mm * 4
    if is_inductor:
        theta = math.atan(omega * value_SI / Z0_ohm)
        X_ohm = omega * value_SI
    else:
        theta = math.atan(omega * value_SI * Z0_ohm)
        X_ohm = 1.0 / (omega * value_SI)
    l_mm = theta * lam_g_mm / (2 * math.pi)
    return W_mm, l_mm, math.degrees(theta), X_ohm


def serie_a_paralelo(R, X):
    """Conversión serie → paralelo de Z = R + jX."""
    if X == 0:
        return R, float('inf')
    Q2 = (X / R) ** 2
    return R * (1 + Q2), X * (1 + 1 / Q2)


def pedir_punto(df):
    """Solicita VCE/IC al usuario; valida que existan en los datos cargados."""
    while True:
        print("\n  Ingresá la polarización a usar para el diseño")
        print("  (formato libre: '2,60'  o  '2 60'  o  'VCE=2 IC=60' …):")
        try:
            entrada = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nCancelado por el usuario.")

        nums = re.findall(r'[\d.]+', entrada.replace(',', '.'))
        if len(nums) < 2:
            print("  ⚠ No se reconocieron VCE e IC. Intentá de nuevo.")
            continue

        vce_in = float(nums[0])
        ic_in  = float(nums[1])

        match = df[(abs(df['VCE (V)'] - vce_in) < 1e-3) &
                   (abs(df['IC (mA)'] - ic_in) < 1e-2)]
        if match.empty:
            print(f"  ⚠ No existe el punto VCE = {vce_in} V, IC = {ic_in} mA "
                  f"en los .s2p cargados.")
            print("     Polarizaciones disponibles:")
            disp = df[['VCE (V)', 'IC (mA)']].drop_duplicates().sort_values(
                ['VCE (V)', 'IC (mA)'])
            for _, r in disp.iterrows():
                print(f"        VCE = {r['VCE (V)']:.2f} V    "
                      f"IC = {r['IC (mA)']:.2f} mA")
            continue
        return match.iloc[0]


# ─── LECTURA Y PROCESAMIENTO ────────────────────────────────────────────────
files = sorted(f for f in os.listdir(UPLOAD_DIR)
                if f.endswith('.s2p') and f.startswith('BFP450'))

rows, seen = [], set()
for fname in files:
    m = re.search(r'VCE_([\d.]+)V_IC_([\d.]+)(mA|A)\.s2p', fname)
    if not m:
        continue
    vce  = float(m.group(1))
    icv  = float(m.group(2))
    ic   = icv * 1000 if m.group(3) == 'A' else icv
    if ic <= IC_MIN_MA:
        continue
    key = (vce, round(ic, 4))
    if key in seen:
        continue
    seen.add(key)

    S11, S21, S12, S22 = load_s2p(os.path.join(UPLOAD_DIR, fname), F_DESIGN)
    r = analyze(S11, S21, S12, S22, Z0)
    rows.append({
        'VCE (V)':   vce,
        'IC (mA)':   ic,
        'S11 (dB)':  round(r['S11_dB'], 2),
        '∠S11 (°)':  round(math.degrees(cmath.phase(r['S11'])), 1),
        'S21 (dB)':  round(r['S21_dB'], 2),
        '∠S21 (°)':  round(math.degrees(cmath.phase(r['S21'])), 1),
        'S12 (dB)':  round(r['S12_dB'], 2),
        '∠S12 (°)':  round(math.degrees(cmath.phase(r['S12'])), 1),
        'S22 (dB)':  round(r['S22_dB'], 2),
        '∠S22 (°)':  round(math.degrees(cmath.phase(r['S22'])), 1),
        'K':         round(r['K'], 3),
        '|Δ|':       round(r['delta'], 3),
        'Estable':   'SÍ' if r['estable'] else 'NO',
        'GT (dB)':   round(r['GT_dB'], 2),
        'MAG (dB)':  round(r['MAG_dB'], 2),
        'Zs':        r['Zs'],
        'ZL':        r['ZL'],
        'Zin':       r['Zin'],
        'Zout':      r['Zout'],
    })

df = pd.DataFrame(rows).sort_values(['VCE (V)', 'IC (mA)']).reset_index(drop=True)


# ─── TABLA 1 ▸ Parámetros S y figuras de mérito ─────────────────────────────
print(f"\n{'='*116}")
print(f"  BFP450 — Diseño @ {F_DESIGN} GHz   (Rgen = RL = {Z0:g} Ω)")
print(f"  Sustrato FR-4: εr = {ER},  h = {H_MM} mm,  t = {T_UM} µm")
print(f"{'='*116}")

print("\n  TABLA 1 ▸ Parámetros S y figuras de mérito por polarización")
print('  ' + '─' * 112)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
t1 = df[['VCE (V)', 'IC (mA)',
         'S11 (dB)', '∠S11 (°)', 'S21 (dB)', '∠S21 (°)',
         'S12 (dB)', '∠S12 (°)', 'S22 (dB)', '∠S22 (°)',
         'K', '|Δ|', 'Estable', 'GT (dB)', 'MAG (dB)']]
print(t1.to_string(index=False))


# ─── SELECCIÓN INTERACTIVA DEL PUNTO ────────────────────────────────────────
elegido = pedir_punto(df)
Zin_e   = elegido['Zin']
Zout_e  = elegido['Zout']
Zs_e    = elegido['Zs']
ZL_e    = elegido['ZL']

print(f"\n{'='*92}")
print(f"  ✓ POLARIZACIÓN ELEGIDA  →  VCE = {elegido['VCE (V)']} V   "
      f"IC = {elegido['IC (mA)']} mA")
print(f"     K = {elegido['K']:.3f}   |Δ| = {elegido['|Δ|']:.3f}   "
      f"Estable: {elegido['Estable']}")
print(f"     GT = {elegido['GT (dB)']:.2f} dB   "
      f"MAG = {elegido['MAG (dB)']:.2f} dB   "
      f"S21 = {elegido['S21 (dB)']:.2f} dB")
if elegido['Estable'] == 'NO':
    print(f"     ⚠ Esta polarización NO es incondicionalmente estable.")
print(f"{'='*92}")


# ─── TABLA 2 ▸ Impedancias de adaptación ────────────────────────────────────
print("\n  TABLA 2 ▸ Impedancias de adaptación en el punto elegido")
print('  ' + '─' * 88)
t2 = pd.DataFrame([
    {'Punto': 'Zin   (transistor — entrada)',  'R (Ω)': f"{Zin_e.real:+8.3f}",
     'X (Ω)': f"{Zin_e.imag:+8.3f}"},
    {'Punto': 'Zout  (transistor — salida)',   'R (Ω)': f"{Zout_e.real:+8.3f}",
     'X (Ω)': f"{Zout_e.imag:+8.3f}"},
    {'Punto': 'Zs    (red debe sintetizar)',   'R (Ω)': f"{Zs_e.real:+8.3f}",
     'X (Ω)': f"{Zs_e.imag:+8.3f}"},
    {'Punto': 'ZL    (red debe sintetizar)',   'R (Ω)': f"{ZL_e.real:+8.3f}",
     'X (Ω)': f"{ZL_e.imag:+8.3f}"},
])
print(t2.to_string(index=False))


# ─── TABLA 3 ▸ Microtiras de la red de adaptación ───────────────────────────
# Topología (EA3 TP2):
#   Entrada: cancelar Xin_p con shunt  →  λ/4 con Z0 = √(Rgen·Rin_p)
#   Salida : λ/4 con Z0 = √(RL·Rout_s)  →  serie Xout_s → paralelo XL_p = −Z0²/Xout_s
Rin_s,  Xin_s  = Zin_e.real,  Zin_e.imag
Rout_s, Xout_s = Zout_e.real, Zout_e.imag

Rin_p, Xin_p = serie_a_paralelo(Rin_s, Xin_s)
Z_QW_in  = math.sqrt(Z0 * Rin_p)
Z_QW_out = math.sqrt(Z0 * Rout_s)

W_50,   l4_50,  er_50   = microstrip(50.0,      ER, H_MM, F_DESIGN)
W_QWi,  l4_QWi, er_QWi = microstrip(Z_QW_in,   ER, H_MM, F_DESIGN)
W_QWo,  l4_QWo, er_QWo = microstrip(Z_QW_out,  ER, H_MM, F_DESIGN)

print("\n  TABLA 3 ▸ Microtiras de la red de adaptación")
print('  ' + '─' * 88)
t3 = pd.DataFrame([
    {'Microtira':  'Línea de 50 Ω (entrada/salida)',
     'Z0 (Ω)':     f"{50.0:6.2f}",
     'W (mm)':     f"{W_50:7.3f}",
     'λ/4 (mm)':   f"{l4_50:7.3f}"},
    {'Microtira':  'Transformador λ/4 — entrada',
     'Z0 (Ω)':     f"{Z_QW_in:6.2f}",
     'W (mm)':     f"{W_QWi:7.3f}",
     'λ/4 (mm)':   f"{l4_QWi:7.3f}"},
    {'Microtira':  'Transformador λ/4 — salida',
     'Z0 (Ω)':     f"{Z_QW_out:6.2f}",
     'W (mm)':     f"{W_QWo:7.3f}",
     'λ/4 (mm)':   f"{l4_QWo:7.3f}"},
])
print(t3.to_string(index=False))

# Componentes de cancelación
f_hz = F_DESIGN * 1e9
# Salida: la reactancia serie Xout_s, tras el λ/4, aparece en PARALELO en el lado de 50 Ω:
#   XL_p = −Z0(QWout)² / Xout_s   (si Xout_s<0 → XL_p>0 → inductivo → cancelar con C shunt)
XL_p_out = -(Z_QW_out**2) / Xout_s if Xout_s != 0 else 0.0

print(f"\n  Componentes para cancelación de reactancia:")
print(f"   • Entrada  →  Rin_p = {Rin_p:7.3f} Ω,  Xin_p = {Xin_p:+7.3f} Ω")
if Xin_p > 0:
    C_in = 1 / (2 * math.pi * f_hz * Xin_p)
    print(f"                cancelar con C shunt de {C_in*1e12:6.3f} pF "
          f"(o stub equivalente)")
    l_stub_in = stub_length_mm(Xin_p, Z0, er_50, F_DESIGN)
    print(f"                stub abierto shunt (Z0={Z0:.0f} Ω):  "
          f"W = {W_50:.3f} mm,  l = {l_stub_in:.3f} mm")
elif Xin_p < 0:
    L_in = abs(Xin_p) / (2 * math.pi * f_hz)
    print(f"                cancelar con L shunt de {L_in*1e9:6.3f} nH "
          f"(o stub equivalente)")
    l_stub_in = stub_length_mm(Xin_p, Z0, er_50, F_DESIGN)
    print(f"                stub CC shunt (Z0={Z0:.0f} Ω):  "
          f"W = {W_50:.3f} mm,  l = {l_stub_in:.3f} mm")

print(f"   • Salida   →  Rout_s = {Rout_s:7.3f} Ω, Xout_s = {Xout_s:+7.3f} Ω")
print(f"                XL_p (paralelo tras λ/4) = {XL_p_out:+7.3f} Ω")
if XL_p_out > 0:
    C_out = 1 / (2 * math.pi * f_hz * XL_p_out)
    print(f"                cancelar con C shunt de {C_out*1e12:6.3f} pF "
          f"(o stub equivalente)")
    l_stub_out = stub_length_mm(XL_p_out, Z0, er_50, F_DESIGN)
    print(f"                stub abierto shunt (Z0={Z0:.0f} Ω):  "
          f"W = {W_50:.3f} mm,  l = {l_stub_out:.3f} mm")
elif XL_p_out < 0:
    L_out = abs(XL_p_out) / (2 * math.pi * f_hz)
    print(f"                cancelar con L shunt de {L_out*1e9:6.3f} nH "
          f"(o stub equivalente)")
    l_stub_out = stub_length_mm(XL_p_out, Z0, er_50, F_DESIGN)
    print(f"                stub CC shunt (Z0={Z0:.0f} Ω):  "
          f"W = {W_50:.3f} mm,  l = {l_stub_out:.3f} mm")

# ─── TABLA 4 ▸ Red de polarización (choke + desacople) ──────────────────────
omega_d   = 2 * math.pi * F_DESIGN * 1e9
XL_choke  = omega_d * L_CHOKE_NH  * 1e-9
XC_decoup = 1.0 / (omega_d * C_DECOUP_PF * 1e-12)

print(f"\n  TABLA 4 ▸ Red de polarización — inductor de choque y capacitor de desacople")
print(f"   L = {L_CHOKE_NH:.0f} nH  →  XL = {XL_choke:.1f} Ω  @ {F_DESIGN} GHz   "
      f"(stub CC shunt,  Z0 ∈ [{Z0_CHOKE_MIN:.0f}, {Z0_CHOKE_MAX:.0f}] Ω)")
print(f"   C = {C_DECOUP_PF:.0f} pF  →  XC = {XC_decoup:.2f} Ω  @ {F_DESIGN} GHz   "
      f"(stub CA shunt,  Z0 ∈ [{Z0_DECOUP_MIN:.0f}, {Z0_DECOUP_MAX:.0f}] Ω)")
print('  ' + '─' * 96)

rows_t4 = []
for Z0_c in [Z0_CHOKE_MIN, (Z0_CHOKE_MIN + Z0_CHOKE_MAX) / 2, Z0_CHOKE_MAX]:
    W_mm, l_mm, theta_deg, _ = bias_stub(
        L_CHOKE_NH * 1e-9, True, Z0_c, F_DESIGN, ER, H_MM)
    rows_t4.append({
        'Componente': f'Choke  {L_CHOKE_NH:.0f} nH',
        'Tipo stub':   'CC shunt',
        'Z0 (Ω)':     f"{Z0_c:6.1f}",
        'θ (°)':      f"{theta_deg:5.1f}",
        'W (mm)':     f"{W_mm:7.3f}",
        'l (mm)':     f"{l_mm:7.3f}",
    })

for Z0_d in [Z0_DECOUP_MIN, (Z0_DECOUP_MIN + Z0_DECOUP_MAX) / 2, Z0_DECOUP_MAX]:
    W_mm, l_mm, theta_deg, _ = bias_stub(
        C_DECOUP_PF * 1e-12, False, Z0_d, F_DESIGN, ER, H_MM)
    rows_t4.append({
        'Componente': f'Desacople  {C_DECOUP_PF:.0f} pF',
        'Tipo stub':   'CA shunt',
        'Z0 (Ω)':     f"{Z0_d:6.1f}",
        'θ (°)':      f"{theta_deg:5.1f}",
        'W (mm)':     f"{W_mm:7.3f}",
        'l (mm)':     f"{l_mm:7.3f}",
    })

t4 = pd.DataFrame(rows_t4)
print(t4.to_string(index=False))

OUT_DIR = "tablas_csv"
os.makedirs(OUT_DIR, exist_ok=True)

t1.to_csv(os.path.join(OUT_DIR, "tabla1_parametros_S.csv"), index=False)
t2.to_csv(os.path.join(OUT_DIR, "tabla2_impedancias.csv"), index=False)
t3.to_csv(os.path.join(OUT_DIR, "tabla3_microtiras.csv"), index=False)
t4.to_csv(os.path.join(OUT_DIR, "tabla4_polarizacion.csv"), index=False)

print("\nScript finalizado.\n")