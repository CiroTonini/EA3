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
import skrf as rf

# ─── CONFIGURACIÓN ──────────────────────────────────────────────────────────
UPLOAD_DIR = r"C:\Users\cirov\OneDrive\Documents\MATERIAS\EA3\TP2\s2p"
F_DESIGN   = 2.2          # GHz — frecuencia de diseño
Z0         = 50.0         # Ω   — impedancia característica (= Rgen = RL)
ER         = 4.4          # FR-4
H_MM       = 1.6          # mm
T_UM       = 35           # µm
IC_MIN_MA  = 10           # mA — descartar puntos por debajo


# ─── FUNCIONES AUXILIARES ───────────────────────────────────────────────────
def load_s2p(path, f_ghz):
    """Carga un .s2p e interpola a la frecuencia de diseño."""
    ntwk = rf.Network(path)
    f_target = rf.Frequency(f_ghz * 1e9, f_ghz * 1e9, 1, unit='hz')
    s = ntwk.interpolate(f_target).s[0]
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
        GT_dB =10 * math.log10(GT),
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
    return W * 1e3, lam / 4 * 1e3


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
#   Entrada: cancelar X de Zin con C en paralelo  →  λ/4 con Z0 = √(Rgen·Rin_p)
#   Salida : λ/4 con Z0 = √(RL·Rout_s)            →  cancelar X transformada con C
Rin_s,  Xin_s  = Zin_e.real,  Zin_e.imag
Rout_s, Xout_s = Zout_e.real, Zout_e.imag

Rin_p, Xin_p = serie_a_paralelo(Rin_s, Xin_s)
Z_QW_in  = math.sqrt(Z0 * Rin_p)
Z_QW_out = math.sqrt(Z0 * Rout_s)

W_50,   l4_50   = microstrip(50.0,      ER, H_MM, F_DESIGN)
W_QWi,  l4_QWi  = microstrip(Z_QW_in,   ER, H_MM, F_DESIGN)
W_QWo,  l4_QWo  = microstrip(Z_QW_out,  ER, H_MM, F_DESIGN)

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
X_out_p_after_QW = Z_QW_out**2 / Xout_s if Xout_s != 0 else 0.0

print(f"\n  Componentes para cancelación de reactancia:")
print(f"   • Entrada  →  Rin_p = {Rin_p:7.3f} Ω,  Xin_p = {Xin_p:+7.3f} Ω")
if Xin_p > 0:
    C_in = 1 / (2 * math.pi * f_hz * Xin_p)
    print(f"                cancelar con C en paralelo de {C_in*1e12:6.3f} pF "
          f"(o stub equivalente)")
elif Xin_p < 0:
    L_in = abs(Xin_p) / (2 * math.pi * f_hz)
    print(f"                cancelar con L en paralelo de {L_in*1e9:6.3f} nH "
          f"(o stub equivalente)")

print(f"   • Salida   →  Rout_s = {Rout_s:7.3f} Ω, Xout_s = {Xout_s:+7.3f} Ω")
print(f"                X transformada tras λ/4 = {X_out_p_after_QW:+7.3f} Ω")
if X_out_p_after_QW > 0:
    C_out = 1 / (2 * math.pi * f_hz * X_out_p_after_QW)
    print(f"                cancelar con C en paralelo de {C_out*1e12:6.3f} pF "
          f"(o stub equivalente)")
elif X_out_p_after_QW < 0:
    L_out = abs(X_out_p_after_QW) / (2 * math.pi * f_hz)
    print(f"                cancelar con L en paralelo de {L_out*1e9:6.3f} nH "
          f"(o stub equivalente)")

print(f"\n  Para la red de polarización (no incluida arriba):")
print(f"   • Choke  Lch  →  microtira de Z0 alto  (≈ 100 – 250 Ω), largo λ/4")
print(f"   • Decoup C    →  microtira de Z0 bajo  (≈  20 –  60 Ω), largo λ/4")

print("\nScript finalizado.\n")
