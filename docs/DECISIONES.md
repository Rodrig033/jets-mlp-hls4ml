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

## E2 — Capacidad (8 corridas, semilla 0, lr 3e-3, batch 4096, ≤120 épocas)

| Corrida | Parámetros | F1m | acc | s | mejor época |
|---|---:|---:|---:|---:|---:|
| e2_256-128-64 | 45,829 | 0.7761 | 0.7731 | 47.1 | 46 |
| e2_128-128-128 | 35,845 | 0.7757 | 0.7734 | 59.6 | 63 |
| e2_128-64-32 | 12,677 | 0.7756 | 0.7728 | 68.1 | 84 |
| e2_512-512-512 | 536,581 | 0.7755 | 0.7730 | 137.6 | 30 |
| e2_256-256-256 | 137,221 | 0.7753 | 0.7727 | 75.4 | 42 |
| e2_512-256-128 | 173,573 | 0.7752 | 0.7726 | 60.1 | 23 |
| e2_128x5 | 68,869 | 0.7749 | 0.7721 | 70.1 | 53 |
| e2_64-32-32_control | 4,389 | 0.7739 | 0.7712 | 42.9 | 58 |

Conclusiones:
1. **Más épocas casi no aportan:** control 0.7739 vs. la misma configuración
   con 50 épocas en E1 (0.7737), +0.0002.
2. **La capacidad ayuda poco y se satura pronto:** mejor arquitectura +0.0022
   sobre el control; desde ~12.7k parámetros no hay mejora pese a multiplicar
   los parámetros hasta ×42. Rango total de E2: 0.0022.
3. Junto con E1, la evidencia indica que el límite (~0.776) está en la
   **información de las 16 características** (traslape físico q/g y W/Z), no en
   el optimizador ni en la arquitectura.
4. Las redes más grandes alcanzan su mejor época antes (23–30 vs. 46–84):
   posible inicio de sobreajuste (por confirmar con curvas).
5. Las diferencias entre variantes son de milésimas: **hay que medir la
   variación entre semillas antes de elegir** (E3).

Decisiones:
- E3: repetir con semillas 0, 1, 2 el control y las 3 mejores arquitecturas.
- E4 (ablación de un factor a la vez) sobre 256-128-64, provisional hasta E3:
  cosine, BatchNorm, dropout 0.1, AdamW + weight decay, SiLU, transformación
  Yeo-Johnson (por la asimetría del EDA) y quitar la columna D₂ duplicada.

Diagnóstico de curvas (mejor época de cada red):

| Red | Parámetros | train_loss | val_loss | brecha |
|---|---:|---:|---:|---:|
| 64-32-32 | 4,389 | 0.6415 | 0.6408 | −0.0007 |
| 256-128-64 | 45,829 | 0.6283 | 0.6322 | +0.0039 |
| 512-512-512 | 536,581 | 0.6230 | 0.6329 | +0.0099 |

La brecha crece con la capacidad. De 256-128-64 a 512-512-512 la pérdida de
entrenamiento baja pero la de validación no: la capacidad extra se usa para
memorizar (varianza), no para generalizar. 64-32-32 = sesgo; 256-128-64 ≈
punto óptimo; 512-512-512 = sobreajuste leve sin ganancia.

## E3 — Variación entre semillas (semillas 0, 1, 2; misma partición)

| Variante | F1m (media ± d.e.) | acc (media ± d.e.) | valores F1m |
|---|---|---|---|
| e2_256-128-64 | 0.7759 ± 0.0004 | 0.7727 ± 0.0004 | 0.7761 0.7755 0.7762 |
| e2_128-128-128 | 0.7756 ± 0.0003 | 0.7729 ± 0.0006 | 0.7757 0.7752 0.7759 |
| e2_128-64-32 | 0.7755 ± 0.0003 | 0.7725 ± 0.0005 | 0.7756 0.7752 0.7757 |
| e2_64-32-32_control | 0.7744 ± 0.0007 | 0.7717 ± 0.0004 | 0.7739 0.7753 0.7742 |

Conclusiones:
1. Ruido entre semillas: d.e. ≈ 0.0003–0.0007. Diferencias menores a ~0.001
   no se consideran mejoras.
2. El aumento de capacidad es real: la peor semilla de 256-128-64 (0.7755)
   supera a la mejor del control (0.7753).
3. Las tres mejores arquitecturas están empatadas (≈1 d.e.).

## E4 — Ablación sobre 256-128-64 (una semilla, un factor a la vez)

| Variante | F1m | Δ vs ref | acc | s | mejor época |
|---|---:|---:|---:|---:|---:|
| e4_cosine | 0.7768 | +0.0007 | 0.7742 | 52.1 | 49 |
| e4_silu | 0.7765 | +0.0004 | 0.7740 | 67.6 | 70 |
| e4_yeojohnson | 0.7764 | +0.0003 | 0.7739 | 77.1 | 81 |
| e4_ref | 0.7761 | — | 0.7731 | 51.9 | 46 |
| e4_sin_d2dup | 0.7761 | 0.0000 | 0.7734 | 72.2 | 66 |
| e4_adamw_wd1e-4 | 0.7757 | −0.0004 | 0.7734 | 55.0 | 53 |
| e4_batchnorm | 0.7756 | −0.0005 | 0.7727 | 43.5 | 26 |
| e4_dropout0.1 | 0.7753 | −0.0008 | 0.7730 | 63.0 | 53 |

Conclusiones:
1. Cosine, SiLU y Yeo-Johnson mejoran levemente, pero cada una por separado
   está dentro de 1–2 d.e.: no concluyente.
2. Quitar la D₂ duplicada no cambia el F1m: confirma empíricamente que la
   columna no aporta información (256 parámetros menos).
3. Dropout, BatchNorm y weight decay empeoran: la regularización reduce la
   brecha, pero la validación no mejoraba con capacidad, así que no hay
   generalización que recuperar.

Decisiones:
- Se mantiene 256-128-64 (empatada con 128-128-128 y 128-64-32, pero es donde
  se midió la ablación; trasladar las mejoras a otra red sin medirlas no tendría
  fundamento).
- E5: combinar cosine + SiLU + Yeo-Johnson + sin D₂ duplicada, y elegir la
  final con 3 semillas entre e4_ref, e4_cosine y e5_combo. Se corren también
  las semillas 1 y 2 del modelo base para compararlo en prueba con media ± d.e.
