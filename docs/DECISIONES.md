# Bitácora de decisiones

Las métricas de cada corrida están en `experiments/bitacora.csv`; aquí se
registra el *porqué* de cada paso. Todas las métricas son de **validación**
(partición 70/15/15, semilla 42). El conjunto de prueba no se ha usado.

## E0 — Modelo base (2026-11-22)
- Configuración: `configs/baseline.json`. MLP 16-64-32-32-5, ReLU, Adam
  lr=1e-3, batch 1024, máx. 50 épocas, early stopping (patience 10) por F1-macro.
- La arquitectura sigue a Duarte et al. (2018); lr y batch son elección propia
  (pendiente: verificar los del artículo).
- Resultado: **F1-macro 0.7725 · accuracy 0.7692 · 4,389 parámetros · 87 s**.
- Diagnóstico (`scripts/curvas.py`): mejor época 41; train_loss 0.6415,
  val_loss 0.6408, brecha −0.0007 → **sin sobreajuste; indicio de subajuste**.

## E1 — Dinámica de entrenamiento (8 corridas, semilla 0)
Arquitectura fija; se varían lr ∈ {3e-4, 1e-3, 3e-3, 1e-2} y batch ∈ {256, 1024, 4096}.

| Corrida | F1m | acc | s | mejor época |
|---|---:|---:|---:|---:|
| e1_lr1e-3_bs256 | 0.7745 | 0.7716 | 421.0 | 48 |
| e1_lr3e-3_bs256 | 0.7738 | 0.7713 | 420.8 | 46 |
| e1_lr3e-3_bs4096 | 0.7737 | 0.7707 | 26.6 | 48 |
| e1_lr1e-2_bs4096 | 0.7737 | 0.7708 | 26.3 | 49 |
| e1_lr3e-3_bs1024 | 0.7736 | 0.7713 | 80.3 | 36 |
| e1_lr3e-4_bs256 | 0.7730 | 0.7701 | 408.5 | 48 |
| e1_lr1e-3_bs4096 | 0.7712 | 0.7680 | 27.9 | 49 |
| e1_lr3e-4_bs1024 | 0.7703 | 0.7670 | 93.8 | 48 |

Conclusiones:
1. Rango total de 0.004 en F1m; la mejor mejora sobre la base es +0.002,
   probablemente dentro de la variación entre semillas (por medir). **El
   optimizador no es el cuello de botella** de la red de 4,389 parámetros.
2. 6 de 8 corridas alcanzan su mejor época en 46–49 de 50: **el tope de épocas
   estaba limitando** el entrenamiento.

Decisiones para E2:
- **lr = 3e-3, batch = 4096**: F1m 0.7737 en 26.6 s, frente a 0.7745 en 421 s
  con batch 256 (16× más rápido por −0.0008). La velocidad permite explorar
  arquitecturas más grandes. Se descarta lr = 1e-2 (mismo F1m) por riesgo de
  inestabilidad en redes más profundas.
- **Máx. 120 épocas, patience 15** (por la conclusión 2).
- Se incluye una corrida de **control** (red base con 120 épocas) para separar
  el efecto de entrenar más tiempo del efecto de aumentar la capacidad.
- No se añade regularización todavía: no hay sobreajuste que corregir.
