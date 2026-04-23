"""
=============================================================================
Análisis de Amplificador RF - BFP450
VCE = 2.0 V, IC = 60 mA
Frecuencia de diseño: 2.2 GHz
=============================================================================
Dependencias:
    pip install scikit-rf pandas numpy

Autor: Script generado para análisis de amplificador de microondas
=============================================================================
"""

import numpy as np
import skrf as rf
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DEL ARCHIVO S2P
# ─────────────────────────────────────────────────────────────────────────────
archivo = 'BFP450_w_noise_VCE_2.0V_IC_60mA.s2p'
red = rf.Network(archivo)

# Frecuencia de diseño
f_design_GHz = 2.2      # GHz
f_design_Hz  = f_design_GHz * 1e9

# Buscar el índice más cercano a 2.2 GHz
idx = np.argmin(np.abs(red.f - f_design_Hz))
f_real = red.f[idx] / 1e9
print(f"\n{'='*60}")
print(f"  BFP450 – Análisis a {f_design_GHz} GHz")
print(f"  Frecuencia más cercana en archivo: {f_real:.3f} GHz  (índice {idx})")
print(f"{'='*60}")

# Impedancia de referencia
Z0 = 50.0   # Ω

# ─────────────────────────────────────────────────────────────────────────────
# 2. PARÁMETROS S EN f_design
# ─────────────────────────────────────────────────────────────────────────────
S = red.s[idx]          # Matriz 2x2 compleja

S11 = S[0, 0]
S12 = S[0, 1]
S21 = S[1, 0]
S22 = S[1, 1]

def mag_dB(x):
    return 20 * np.log10(np.abs(x))

df_s = pd.DataFrame({
    "Parámetro": ["S11", "S12", "S21", "S22"],
    "|Sij| (lin)": [np.abs(S11), np.abs(S12), np.abs(S21), np.abs(S22)],
    "|Sij| (dB)":  [mag_dB(S11), mag_dB(S12), mag_dB(S21), mag_dB(S22)],
    "Ángulo (°)":  [np.degrees(np.angle(S11)),
                    np.degrees(np.angle(S12)),
                    np.degrees(np.angle(S21)),
                    np.degrees(np.angle(S22))],
})
print("\n── Parámetros S ────────────────────────────────────────────────")
print(df_s.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# ─────────────────────────────────────────────────────────────────────────────
# 3. FACTOR DE ESTABILIDAD DE ROLLETT (K) y Δ
# ─────────────────────────────────────────────────────────────────────────────
delta = S11 * S22 - S12 * S21         # Determinante de [S]

K = (1 - np.abs(S11)**2 - np.abs(S22)**2 + np.abs(delta)**2) / \
    (2 * np.abs(S12 * S21))

B1 = 1 + np.abs(S11)**2 - np.abs(S22)**2 - np.abs(delta)**2

estable = "INCONDICIONALMENTE ESTABLE" if (K > 1 and np.abs(delta) < 1) \
          else "POTENCIALMENTE INESTABLE"

df_estab = pd.DataFrame({
    "Parámetro":  ["K (Rollett)", "|Δ|", "B1", "Condición"],
    "Valor":      [f"{K:.4f}", f"{np.abs(delta):.4f}", f"{B1:.4f}", estable],
})
print("\n── Estabilidad ─────────────────────────────────────────────────")
print(df_estab.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. COEFICIENTES DE REFLEXIÓN DE ENTRADA Y SALIDA (Γin / Γout)
#    con carga adaptada (ΓS = ΓL = 0) – caso sin red de adaptación
# ─────────────────────────────────────────────────────────────────────────────
# Para máxima ganancia disponible se usarán Γs_opt y ΓL_opt en §5.
# Aquí se calcula Γin/Γout con source/load conjugadas (caso MAG/MSG).

# Con ΓL = 0 y ΓS = 0 (terminaciones en Z0):
Gamma_in_Z0  = S11                     # Γin = S11 si ΓL = 0
Gamma_out_Z0 = S22                     # Γout = S22 si ΓS = 0

# ─────────────────────────────────────────────────────────────────────────────
# 5. COEFICIENTES ÓPTIMOS PARA MÁXIMA GANANCIA DISPONIBLE
#    (Γs_opt y ΓL_opt  ↔  conjugate match)
# ─────────────────────────────────────────────────────────────────────────────
B1_val = 1 + np.abs(S11)**2 - np.abs(S22)**2 - np.abs(delta)**2
B2_val = 1 + np.abs(S22)**2 - np.abs(S11)**2 - np.abs(delta)**2
C1     = S11 - delta * np.conj(S22)
C2     = S22 - delta * np.conj(S11)

# Γs_opt (fuente para MAG)
sign_s = +1 if B1_val > 0 else -1
Gamma_s_opt = (B1_val - sign_s * np.sqrt(B1_val**2 - 4 * np.abs(C1)**2)) / \
              (2 * C1)

# ΓL_opt (carga para MAG)
sign_l = +1 if B2_val > 0 else -1
Gamma_l_opt = (B2_val - sign_l * np.sqrt(B2_val**2 - 4 * np.abs(C2)**2)) / \
              (2 * C2)

# Γin y Γout con las terminaciones óptimas
Gamma_in  = S11 + S12 * S21 * Gamma_l_opt / (1 - S22 * Gamma_l_opt)
Gamma_out = S22 + S12 * S21 * Gamma_s_opt / (1 - S11 * Gamma_s_opt)

df_gamma = pd.DataFrame({
    "Coeficiente":   ["Γin (Z0 match)", "Γout (Z0 match)",
                      "Γs_opt (MAG)",  "ΓL_opt (MAG)",
                      "Γin (Γs_opt,ΓL_opt)", "Γout (Γs_opt,ΓL_opt)"],
    "|Γ|":           [np.abs(Gamma_in_Z0),  np.abs(Gamma_out_Z0),
                      np.abs(Gamma_s_opt),  np.abs(Gamma_l_opt),
                      np.abs(Gamma_in),     np.abs(Gamma_out)],
    "Ángulo (°)":    [np.degrees(np.angle(Gamma_in_Z0)),
                      np.degrees(np.angle(Gamma_out_Z0)),
                      np.degrees(np.angle(Gamma_s_opt)),
                      np.degrees(np.angle(Gamma_l_opt)),
                      np.degrees(np.angle(Gamma_in)),
                      np.degrees(np.angle(Gamma_out))],
    "RL (dB)":       [mag_dB(Gamma_in_Z0),  mag_dB(Gamma_out_Z0),
                      mag_dB(Gamma_s_opt),  mag_dB(Gamma_l_opt),
                      mag_dB(Gamma_in),     mag_dB(Gamma_out)],
})
print("\n── Coeficientes de reflexión ───────────────────────────────────")
print(df_gamma.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# ─────────────────────────────────────────────────────────────────────────────
# 6. IMPEDANCIAS DE ENTRADA Y SALIDA
# ─────────────────────────────────────────────────────────────────────────────
def gamma_to_Z(gamma, Z0=50.0):
    return Z0 * (1 + gamma) / (1 - gamma)

Zin_Z0  = gamma_to_Z(Gamma_in_Z0)
Zout_Z0 = gamma_to_Z(Gamma_out_Z0)
Zin_opt = gamma_to_Z(Gamma_in)
Zout_opt= gamma_to_Z(Gamma_out)
Zs_opt  = gamma_to_Z(Gamma_s_opt)
ZL_opt  = gamma_to_Z(Gamma_l_opt)

df_Z = pd.DataFrame({
    "Impedancia":   ["Zin  (ΓL=0)",   "Zout (ΓS=0)",
                     "Zin  (MAG)",    "Zout (MAG)",
                     "Zs_opt (fuente)", "ZL_opt (carga)"],
    "Re [Ω]":       [Zin_Z0.real,  Zout_Z0.real,
                     Zin_opt.real, Zout_opt.real,
                     Zs_opt.real,  ZL_opt.real],
    "Im [Ω]":       [Zin_Z0.imag,  Zout_Z0.imag,
                     Zin_opt.imag, Zout_opt.imag,
                     Zs_opt.imag,  ZL_opt.imag],
    "|Z| [Ω]":      [np.abs(Zin_Z0),  np.abs(Zout_Z0),
                     np.abs(Zin_opt), np.abs(Zout_opt),
                     np.abs(Zs_opt),  np.abs(ZL_opt)],
})
print("\n── Impedancias ─────────────────────────────────────────────────")
print(df_Z.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# ─────────────────────────────────────────────────────────────────────────────
# 7. GANANCIAS
# ─────────────────────────────────────────────────────────────────────────────
# Ganancia máxima disponible (MAG) – solo si K > 1
if K > 1:
    MAG = np.abs(S21) / np.abs(S12) * (K - np.sqrt(K**2 - 1))
    MSG = None
else:
    MAG = None
    MSG = np.abs(S21) / np.abs(S12)   # Máxima ganancia estable

# Ganancia de transducción (GT) con Γs y ΓL óptimas
GT_num = (1 - np.abs(Gamma_s_opt)**2) * np.abs(S21)**2 * (1 - np.abs(Gamma_l_opt)**2)
GT_den = np.abs((1 - S11 * Gamma_s_opt) * (1 - S22 * Gamma_l_opt) - S12 * S21 * Gamma_s_opt * Gamma_l_opt)**2
GT = GT_num / GT_den

# Ganancia de potencia disponible (GA) con Γs_opt
GA_num = (1 - np.abs(Gamma_s_opt)**2) * np.abs(S21)**2
GA_den = np.abs(1 - S11 * Gamma_s_opt)**2 * (1 - np.abs(Gamma_out)**2)
GA = GA_num / GA_den

# Ganancia de potencia operativa (GP) con ΓL_opt
GP_num = np.abs(S21)**2 * (1 - np.abs(Gamma_l_opt)**2)
GP_den = (1 - np.abs(Gamma_in)**2) * np.abs(1 - S22 * Gamma_l_opt)**2
GP = GP_num / GP_den

rows = [
    ("GT  (transducción, Γs_opt/ΓL_opt)", 10 * np.log10(GT)),
    ("GA  (disponible, Γs_opt)",           10 * np.log10(GA)),
    ("GP  (operativa, ΓL_opt)",            10 * np.log10(GP)),
]
if MAG is not None:
    rows.append(("MAG (máxima disponible, K>1)", 10 * np.log10(MAG)))
else:
    rows.append(("MSG (máx. ganancia estable, K<1)", 10 * np.log10(MSG)))

rows.append(("S21 (ganancia inserción)",  mag_dB(S21)))

df_G = pd.DataFrame(rows, columns=["Ganancia", "Valor (dB)"])
print("\n── Ganancias ───────────────────────────────────────────────────")
print(df_G.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# ─────────────────────────────────────────────────────────────────────────────
# 8. DISEÑO DE MICROTIRAS (Microstrip)
#    Sustrato: FR-4  (εr = 4.4, h = 1.6 mm, t = 35 µm)
#    Se calcula ancho (W) y longitud λ/4 para cada impedancia
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Dimensiones de Microtiras ───────────────────────────────────")
print("   Sustrato por defecto: FR-4  εr=4.4  h=1.6 mm  t=35 µm")
print("   Modifica εr, h, t según tu sustrato real.\n")

er  = 4.4       # permitividad relativa del sustrato
h   = 1.6e-3    # espesor del dieléctrico (m)
t   = 35e-6     # espesor del conductor (m)
f   = f_design_Hz
c   = 3e8       # velocidad de la luz (m/s)


def microstrip_dimensions(Zc, er, h, t, f):
    """
    Calcula W y λ_eff/4 de una microtira dada su impedancia característica Zc.
    Usa las fórmulas de Hammerstad & Jensen (IPC-2141A).

    Retorna: (W_mm, lambda_eff_4_mm, er_eff, lambda_full_mm)
    """
    # ── ancho W ──────────────────────────────────────────────────────────────
    A = (Zc / 60) * np.sqrt((er + 1) / 2) + ((er - 1) / (er + 1)) * (0.23 + 0.11 / er)
    B = 377 * np.pi / (2 * Zc * np.sqrt(er))

    W_over_h_A = 8 * np.exp(A) / (np.exp(2 * A) - 2)           # W/h para W/h < 2
    W_over_h_B = (2 / np.pi) * (B - 1 - np.log(2 * B - 1) +   # W/h para W/h > 2
                  (er - 1) / (2 * er) * (np.log(B - 1) + 0.39 - 0.61 / er))

    # Seleccionar la solución correcta
    if W_over_h_A < 2:
        W_over_h = W_over_h_A
    else:
        W_over_h = W_over_h_B

    W = W_over_h * h

    # ── εr efectiva ──────────────────────────────────────────────────────────
    if W_over_h < 1:
        er_eff = ((er + 1) / 2 + (er - 1) / 2 *
                  (1 / np.sqrt(1 + 12 * h / W) + 0.04 * (1 - W / h)**2))
    else:
        er_eff = (er + 1) / 2 + (er - 1) / 2 * 1 / np.sqrt(1 + 12 * h / W)

    # ── longitudes ───────────────────────────────────────────────────────────
    lambda_full = c / (f * np.sqrt(er_eff))
    lambda_4    = lambda_full / 4

    return W * 1e3, lambda_4 * 1e3, er_eff, lambda_full * 1e3  # mm


# Línea de 50 Ω (referencia)
W50, l4_50, eff50, lam50 = microstrip_dimensions(50, er, h, t, f)

# Líneas para las impedancias de diseño
impedancias = {
    "Z0 = 50 Ω (referencia)":       50.0,
    "Zs_opt Re(Ω)  (fuente)":       max(abs(Zs_opt.real), 1.0),
    "|Zs_opt| (Ω) (fuente)":        abs(Zs_opt),
    "ZL_opt Re(Ω)  (carga)":        max(abs(ZL_opt.real), 1.0),
    "|ZL_opt| (Ω) (carga)":         abs(ZL_opt),
    "Zin Re(Ω)  (entrada MAG)":     max(abs(Zin_opt.real), 1.0),
    "Zout Re(Ω) (salida MAG)":      max(abs(Zout_opt.real), 1.0),
}

filas = []
for nombre, Zc in impedancias.items():
    try:
        W_mm, l4_mm, er_eff, lam_mm = microstrip_dimensions(Zc, er, h, t, f)
        filas.append({
            "Descripción":    nombre,
            "Zc (Ω)":         f"{Zc:.2f}",
            "W (mm)":         f"{W_mm:.4f}",
            "λ/4 (mm)":       f"{l4_mm:.4f}",
            "λ (mm)":         f"{lam_mm:.4f}",
            "εr_eff":         f"{er_eff:.4f}",
        })
    except Exception as e:
        filas.append({
            "Descripción": nombre, "Zc (Ω)": f"{Zc:.2f}",
            "W (mm)": "N/A", "λ/4 (mm)": "N/A",
            "λ (mm)": "N/A", "εr_eff": str(e),
        })

df_ms = pd.DataFrame(filas)
print(df_ms.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 9. RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║              RESUMEN – BFP450 @ {f_design_GHz} GHz                    ║
╠══════════════════════════════════════════════════════════════╣
║  Condición de polarización : VCE=2 V, IC=60 mA             ║
║  Frecuencia de diseño      : {f_real:.3f} GHz                      ║
║                                                              ║
║  S11 = {np.abs(S11):.4f} ∠ {np.degrees(np.angle(S11)):+.1f}°    S21 = {np.abs(S21):.4f} ∠ {np.degrees(np.angle(S21)):+.1f}°  ║
║  S12 = {np.abs(S12):.4f} ∠ {np.degrees(np.angle(S12)):+.1f}°     S22 = {np.abs(S22):.4f} ∠ {np.degrees(np.angle(S22)):+.1f}° ║
║                                                              ║
║  K (Rollett) = {K:.4f}   |Δ| = {np.abs(delta):.4f}                     ║
║  Estabilidad : {estable:<44}║
║                                                              ║
║  GT (adapt. conjugada) = {10*np.log10(GT):+.2f} dB                    ║
║  S21 (ganancia inserción) = {mag_dB(S21):+.2f} dB                   ║
║                                                              ║
║  Zs_opt = {Zs_opt.real:+.2f}{Zs_opt.imag:+.2f}j Ω   W={abs(Zs_opt):.1f}Ω    ║
║  ZL_opt = {ZL_opt.real:+.2f}{ZL_opt.imag:+.2f}j Ω   W={abs(ZL_opt):.1f}Ω    ║
║                                                              ║
║  Microtira Z0=50Ω : W={W50:.3f} mm  λ/4={l4_50:.3f} mm            ║
║  Sustrato FR-4    : εr={er}  h={h*1e3:.1f}mm  t={t*1e6:.0f}µm           ║
╚══════════════════════════════════════════════════════════════╝
""")

print("Script finalizado. Modifica εr / h / t en §8 para tu sustrato real.")
