"""
=============================================================================
Análisis Comparativo de Amplificador RF - BFP450
Múltiples puntos de polarización (VCE / IC)
Frecuencia de diseño: 2.2 GHz
=============================================================================
Dependencias:
    pip install scikit-rf pandas numpy

Uso:
    Colocar todos los archivos .s2p en el mismo directorio que este script
    (o ajustar UPLOAD_DIR) y ejecutar:
        python BFP450_analisis_comparativo.py
=============================================================================
"""

import cmath
import math
import os
import re
import json
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
UPLOAD_DIR  = r"C:\Users\cirov\OneDrive\Documents\MATERIAS\EA3\TP2\s2p"      # directorio con los .s2p
F_DESIGN    = 2.2          # GHz
Z0          = 50.0         # Ω impedancia de referencia

# Parámetros del sustrato (cambiar según tu PCB real)
ER          = 4.4          # permitividad relativa (FR-4)
H_MM        = 1.6          # espesor dieléctrico [mm]
T_UM        = 35           # espesor conductor [µm]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────
def polar(mag, ang_deg):
    return mag * cmath.exp(1j * math.radians(ang_deg))


def parse_s2p(path):
    """Lee un .s2p en formato MA → dict {freq_GHz: (S11,S21,S12,S22)}"""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            cols = line.split()
            if len(cols) < 9:
                continue
            try:
                freq = float(cols[0])
                s11 = polar(float(cols[1]), float(cols[2]))
                s21 = polar(float(cols[3]), float(cols[4]))
                s12 = polar(float(cols[5]), float(cols[6]))
                s22 = polar(float(cols[7]), float(cols[8]))
                data[freq] = (s11, s21, s12, s22)
            except ValueError:
                continue
    return data


def find_nearest(data, target_ghz):
    freqs = list(data.keys())
    idx   = min(range(len(freqs)), key=lambda i: abs(freqs[i] - target_ghz))
    return freqs[idx], data[freqs[idx]]


def analyze(S11, S21, S12, S22, Z0=50.0):
    delta = S11 * S22 - S12 * S21
    K = (1 - abs(S11)**2 - abs(S22)**2 + abs(delta)**2) / (2 * abs(S12 * S21))
    B1 = 1 + abs(S11)**2 - abs(S22)**2 - abs(delta)**2
    B2 = 1 + abs(S22)**2 - abs(S11)**2 - abs(delta)**2
    C1 = S11 - delta * S22.conjugate()
    C2 = S22 - delta * S11.conjugate()

    sign_s = 1 if B1 > 0 else -1
    sign_l = 1 if B2 > 0 else -1
    Gs = (B1 - sign_s * cmath.sqrt(B1**2 - 4 * abs(C1)**2)) / (2 * C1)
    GL = (B2 - sign_l * cmath.sqrt(B2**2 - 4 * abs(C2)**2)) / (2 * C2)

    Gin  = S11 + S12 * S21 * GL  / (1 - S22 * GL)
    Gout = S22 + S12 * S21 * Gs  / (1 - S11 * Gs)

    def g2Z(g): return Z0 * (1 + g) / (1 - g)
    Zs   = g2Z(Gs);  ZL   = g2Z(GL)
    Zin  = g2Z(Gin); Zout = g2Z(Gout)

    GT_n = (1 - abs(Gs)**2) * abs(S21)**2 * (1 - abs(GL)**2)
    GT_d = abs((1 - S11*Gs)*(1 - S22*GL) - S12*S21*Gs*GL)**2
    GT   = GT_n / GT_d

    MAG  = abs(S21)/abs(S12) * (K - math.sqrt(max(K**2 - 1, 0))) if K > 1 else None
    MSG  = abs(S21) / abs(S12)

    return dict(
        K=K, delta_mag=abs(delta), B1=B1, stable=(K > 1 and abs(delta) < 1),
        Gs=Gs, GL=GL, Gin=Gin, Gout=Gout,
        Zs=Zs, ZL=ZL, Zin=Zin, Zout=Zout,
        GT_dB =10*math.log10(GT),
        MAG_dB=10*math.log10(MAG) if MAG else None,
        MSG_dB=10*math.log10(MSG),
        S21_dB=20*math.log10(abs(S21)),
        S11_dB=20*math.log10(abs(S11)),
        S12_dB=20*math.log10(abs(S12)),
        S22_dB=20*math.log10(abs(S22)),
    )


def microstrip(Zc, er=4.4, h_mm=1.6, f_ghz=2.2):
    """Calcula W y λ/4 de microtira (Hammerstad & Jensen)."""
    h = h_mm * 1e-3
    f = f_ghz * 1e9
    c = 3e8
    A = (Zc / 60) * math.sqrt((er+1)/2) + ((er-1)/(er+1)) * (0.23 + 0.11/er)
    B = 377 * math.pi / (2 * Zc * math.sqrt(er))
    Wh_A = 8 * math.exp(A) / (math.exp(2*A) - 2)
    Wh_B = (2/math.pi)*(B - 1 - math.log(2*B-1) + (er-1)/(2*er)*(math.log(B-1)+0.39-0.61/er))
    Wh   = Wh_A if Wh_A < 2 else Wh_B
    W    = Wh * h
    er_eff = ((er+1)/2 + (er-1)/2 * (1/math.sqrt(1 + 12/Wh))) if Wh >= 1 else \
             ((er+1)/2 + (er-1)/2 * (1/math.sqrt(1 + 12/Wh) + 0.04*(1 - Wh)**2))
    lam = c / (f * math.sqrt(er_eff))
    return W*1e3, lam/4*1e3, er_eff


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA Y PROCESAMIENTO DE ARCHIVOS
# Solo BFP450, IC > 10 mA
# ─────────────────────────────────────────────────────────────────────────────
files = sorted([f for f in os.listdir(UPLOAD_DIR)
                if f.endswith('.s2p') and f.startswith('BFP450')])
rows = []
seen = set()

for fname in files:
    m = re.search(r'VCE_([\d.]+)V_IC_([\d.]+)(mA|A)\.s2p', fname)
    if not m:
        continue
    vce    = float(m.group(1))
    ic_val = float(m.group(2))
    unit   = m.group(3)
    ic     = ic_val * 1000 if unit == 'A' else ic_val   # siempre en mA
    if ic <= 10:
        continue
    key = (vce, round(ic, 4))
    if key in seen:
        continue
    seen.add(key)

    data = parse_s2p(os.path.join(UPLOAD_DIR, fname))
    freq, (S11, S21, S12, S22) = find_nearest(data, F_DESIGN)
    r = analyze(S11, S21, S12, S22, Z0)

    W_Zs, l4_Zs, _ = microstrip(max(abs(r['Zs'].real), 1.0), ER, H_MM, F_DESIGN)
    W_ZL, l4_ZL, _ = microstrip(max(abs(r['ZL'].real), 1.0), ER, H_MM, F_DESIGN)
    W_50, l4_50, _ = microstrip(50.0,                         ER, H_MM, F_DESIGN)

    rows.append({
        'VCE (V)': vce, 'IC (mA)': ic,
        'S11 (dB)': round(r['S11_dB'], 2), 'S21 (dB)': round(r['S21_dB'], 2),
        'S12 (dB)': round(r['S12_dB'], 2), 'S22 (dB)': round(r['S22_dB'], 2),
        'K': round(r['K'], 4), '|Δ|': round(r['delta_mag'], 4),
        'Estable': 'SÍ' if r['stable'] else 'NO',
        'GT (dB)': round(r['GT_dB'], 2),
        'MAG/MSG': round(r['MAG_dB'], 2) if r['MAG_dB'] else round(r['MSG_dB'], 2),
        'Tipo':    'MAG' if r['MAG_dB'] else 'MSG',
        '|Γs_opt|': round(abs(r['Gs']), 4),
        'Γs∠(°)':  round(math.degrees(cmath.phase(r['Gs'])), 1),
        '|ΓL_opt|': round(abs(r['GL']), 4),
        'ΓL∠(°)':  round(math.degrees(cmath.phase(r['GL'])), 1),
        'Zs Re(Ω)': round(r['Zs'].real, 2), 'Zs Im(Ω)': round(r['Zs'].imag, 2),
        'ZL Re(Ω)': round(r['ZL'].real, 2), 'ZL Im(Ω)': round(r['ZL'].imag, 2),
        'Zin Re(Ω)':round(r['Zin'].real,2),  'Zin Im(Ω)':round(r['Zin'].imag,2),
        'Zout Re(Ω)':round(r['Zout'].real,2),'Zout Im(Ω)':round(r['Zout'].imag,2),
        'W_Zs (mm)': round(W_Zs, 3), 'λ/4_Zs (mm)': round(l4_Zs, 3),
        'W_ZL (mm)': round(W_ZL, 3), 'λ/4_ZL (mm)': round(l4_ZL, 3),
        'W_50Ω (mm)':round(W_50, 3), 'λ/4_50Ω (mm)':round(l4_50, 3),
    })

df = pd.DataFrame(rows).sort_values(['VCE (V)', 'IC (mA)']).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABLAS DE SALIDA
# ─────────────────────────────────────────────────────────────────────────────
sep = "─" * 100
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:,.3f}'.format)

print(f"\n{'='*100}")
print(f"  BFP450 – Análisis comparativo @ {F_DESIGN} GHz")
print(f"  Sustrato: FR-4  εr={ER}  h={H_MM}mm  t={T_UM}µm")
print(f"{'='*100}")

# ── TABLA 1: Parámetros S ─────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 1 ▸ Parámetros S")
print(f"{'─'*60}")
t1 = df[['VCE (V)','IC (mA)','S11 (dB)','S21 (dB)','S12 (dB)','S22 (dB)']].copy()
print(t1.to_string(index=False))

# ── TABLA 2: Estabilidad ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 2 ▸ Estabilidad")
print(f"{'─'*60}")
t2 = df[['VCE (V)','IC (mA)','K','|Δ|','Estable']].copy()
print(t2.to_string(index=False))

# ── TABLA 3: Ganancias ────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 3 ▸ Ganancias")
print(f"{'─'*60}")
t3 = df[['VCE (V)','IC (mA)','S21 (dB)','GT (dB)','MAG/MSG','Tipo']].copy()
print(t3.to_string(index=False))

# ── TABLA 4: Coeficientes de reflexión óptimos ───────────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 4 ▸ Coeficientes de reflexión óptimos (Γs_opt, ΓL_opt)")
print(f"{'─'*60}")
t4 = df[['VCE (V)','IC (mA)','|Γs_opt|','Γs∠(°)','|ΓL_opt|','ΓL∠(°)']].copy()
print(t4.to_string(index=False))

# ── TABLA 5: Impedancias ──────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 5 ▸ Impedancias de fuente/carga óptimas y E/S")
print(f"{'─'*60}")
t5 = df[['VCE (V)','IC (mA)',
          'Zs Re(Ω)','Zs Im(Ω)',
          'ZL Re(Ω)','ZL Im(Ω)',
          'Zin Re(Ω)','Zin Im(Ω)',
          'Zout Re(Ω)','Zout Im(Ω)']].copy()
print(t5.to_string(index=False))

# ── TABLA 6: Microtiras ───────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 6 ▸ Dimensiones de microtiras (FR-4)")
print(f"{'─'*60}")
t6 = df[['VCE (V)','IC (mA)',
          'W_Zs (mm)','λ/4_Zs (mm)',
          'W_ZL (mm)','λ/4_ZL (mm)',
          'W_50Ω (mm)','λ/4_50Ω (mm)']].copy()
print(t6.to_string(index=False))

# ── TABLA 7: Comparativa por VCE (promedio sobre IC) ─────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 7 ▸ Variación por VCE (promedio de IC disponibles)")
print(f"{'─'*60}")
t7 = df.groupby('VCE (V)').agg(
    IC_valores=('IC (mA)', lambda x: list(x)),
    S21_medio=('S21 (dB)', 'mean'),
    S21_max  =('S21 (dB)', 'max'),
    GT_medio =('GT (dB)',  'mean'),
    GT_max   =('GT (dB)',  'max'),
    K_medio  =('K',        'mean'),
    K_min    =('K',        'min'),
).reset_index()
t7['IC_valores'] = t7['IC_valores'].apply(lambda x: str(x).replace(' ',''))
print(t7.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── TABLA 8: Comparativa por IC (promedio sobre VCE) ─────────────────────────
print(f"\n{'─'*60}")
print("  TABLA 8 ▸ Variación por IC (promedio de VCE disponibles)")
print(f"{'─'*60}")
t8 = df.groupby('IC (mA)').agg(
    VCE_valores=('VCE (V)', lambda x: list(x)),
    S21_medio  =('S21 (dB)', 'mean'),
    S21_max    =('S21 (dB)', 'max'),
    GT_medio   =('GT (dB)',  'mean'),
    GT_max     =('GT (dB)',  'max'),
    K_medio    =('K',        'mean'),
).reset_index()
t8['VCE_valores'] = t8['VCE_valores'].apply(lambda x: str(x).replace(' ',''))
print(t8.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── PUNTO ÓPTIMO ─────────────────────────────────────────────────────────────
idx_best = df['GT (dB)'].idxmax()
best = df.loc[idx_best]
print(f"\n{'='*100}")
print(f"  ★  PUNTO DE POLARIZACIÓN ÓPTIMO (máxima GT)")
print(f"     VCE = {best['VCE (V)']} V   IC = {best['IC (mA)']} mA")
print(f"     GT  = {best['GT (dB)']} dB   S21 = {best['S21 (dB)']} dB")
print(f"     K   = {best['K']}   |Δ| = {best['|Δ|']}   Estable: {best['Estable']}")
print(f"     Zs_opt = {best['Zs Re(Ω)']:+.2f}{best['Zs Im(Ω)']:+.2f}j Ω")
print(f"     ZL_opt = {best['ZL Re(Ω)']:+.2f}{best['ZL Im(Ω)']:+.2f}j Ω")
print(f"{'='*100}")

print("\nScript finalizado. Ajustá ER / H_MM / T_UM para tu sustrato real.")

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN CSV
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = "tablas_csv"
os.makedirs(OUT_DIR, exist_ok=True)

t1.to_csv(os.path.join(OUT_DIR, "tabla1_parametros_S.csv"), index=False)
t2.to_csv(os.path.join(OUT_DIR, "tabla2_estabilidad.csv"), index=False)
t3.to_csv(os.path.join(OUT_DIR, "tabla3_ganancias.csv"), index=False)
t4.to_csv(os.path.join(OUT_DIR, "tabla4_reflexion_optima.csv"), index=False)
t5.to_csv(os.path.join(OUT_DIR, "tabla5_impedancias.csv"), index=False)
t6.to_csv(os.path.join(OUT_DIR, "tabla6_microtiras.csv"), index=False)
t7.to_csv(os.path.join(OUT_DIR, "tabla7_promedio_VCE.csv"), index=False)
t8.to_csv(os.path.join(OUT_DIR, "tabla8_promedio_IC.csv"), index=False)
df.to_csv(os.path.join(OUT_DIR, "tabla_completa.csv"), index=False)

print(f"\nCSV exportados en: {os.path.abspath(OUT_DIR)}")
