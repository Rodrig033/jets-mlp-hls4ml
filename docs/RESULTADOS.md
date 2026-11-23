# Resultados en prueba (evaluación única tras `v1.0-congelado`)

Conjunto de prueba completo: 124,500 jets (15 %, estratificado, semilla 42).

## Métricas globales (3 semillas, media ± d.e. con ddof=1)

| Modelo | Parámetros | F1-macro | Accuracy | Inferencia (µs/jet)* |
|---|---:|---|---|---|
| Base hls4ml (16-64-32-32-5, ReLU) | 4,389 | 0.7705 ± 0.0006 | 0.7674 ± 0.0006 | 0.21–0.27 |
| Final e5_combo (15-256-128-64-5, SiLU) | 45,573 | **0.7746 ± 0.0002** | **0.7722 ± 0.0002** | 0.59–0.71 |
| Diferencia | ×10.4 | **+0.0041** | +0.0048 | ≈ ×2.7 |

\* Inferencia por lotes de 8,192 en MacBook Air (Apple Silicon); ver
`experiments/prueba_final/resultados.json` para el hardware exacto.

Validación → prueba: ambos modelos bajan ~0.002 en F1-macro y en la misma
magnitud (base −0.0022, final −0.0025), consistente con variación de muestreo
del conjunto de prueba (error estándar del accuracy ≈ 0.0012) y no con un
sobreajuste a validación del modelo final. La mejora sobre la base se conserva
(+0.0044 en validación, +0.0041 en prueba).

## F1 por clase (media de 3 semillas)

| Clase | Base | Final | Δ |
|---|---:|---:|---:|
| g | 0.741 | 0.749 | +0.008 |
| q | 0.663 | 0.667 | +0.004 |
| t | 0.838 | 0.842 | +0.004 |
| w | 0.798 | 0.800 | +0.002 |
| z | 0.813 | 0.815 | +0.002 |

Orden de dificultad: **q < g < W < Z < t**.

## Análisis de la matriz de confusión (semilla 0)

Precisión y recall por clase:

| Clase | Recall base | Recall final | Precisión base | Precisión final |
|---|---:|---:|---:|---:|
| g | 0.747 | 0.775 | 0.733 | 0.725 |
| q | 0.754 | 0.736 | **0.592** | **0.611** |
| t | 0.844 | 0.843 | 0.830 | 0.842 |
| w | 0.766 | 0.771 | 0.834 | 0.833 |
| z | 0.725 | 0.736 | 0.927 | 0.913 |

Confusiones principales del modelo final (fracción de la clase real):

| Real → predicho | Jets | % |
|---|---:|---:|
| W → q | 3,913 | 15.6 |
| Z → q | 3,588 | 14.3 |
| q → g | 3,536 | 14.6 |
| g → q | 3,107 | 12.4 |
| g → t | 1,802 | 7.2 |
| Z → W | 1,544 | 6.2 |
| W → Z | 453 | 1.8 |

Hallazgos:
1. **La clase q funciona como "sumidero".** El modelo final predice q en el
   23.4 % de los jets (base: 24.7 %), aunque q es solo el 19.4 % real. El 40 %
   de todos los errores son predicciones de q (base: 43 %), lo que explica su
   baja precisión (0.61).
2. **El error dominante no es W↔Z, sino W/Z → q.** W y Z decaen en dos quarks;
   una interpretación plausible es que, cuando los dos subchorros no se
   resuelven o el *grooming* elimina uno, el jet se parece a uno de un solo
   quark. W↔Z es menor y **asimétrico**: Z → W 6.2 % frente a W → Z 1.8 %.
3. **q ↔ g es la confusión simétrica esperada** (~12–15 % en ambos sentidos):
   ambas tienen un solo núcleo.
4. **t es la clase más fácil** (F1 0.84) y no cambia entre modelos: tres
   subchorros y la masa más alta (~173 GeV) la hacen distintiva.

## Qué cambió del modelo base al final (diferencia de matrices, semilla 0)

- 655 aciertos más (de 124,500).
- El final predice q con menos frecuencia: −559 g → q, −303 W → q, −281 Z → q.
- Costo: −435 aciertos de q (recall de q 0.754 → 0.736) y +336 q → g.
- Balance: la precisión de q sube (0.592 → 0.611) y el F1 de q mejora
  ligeramente; g gana más (+702 aciertos, recall 0.747 → 0.775).

Es decir, la mejora no viene de resolver la confusión de fondo, sino de que el
modelo final reparte mejor la frontera alrededor de la clase q.

## Curvas ROC y AUC (uno contra el resto; prueba; 3 semillas)

Análisis posterior a la evaluación final sobre los mismos modelos congelados
(`scripts/roc_auc.py`); no interviene en ninguna selección.

| Modelo | AUC macro | g | q | t | W | Z |
|---|---|---:|---:|---:|---:|---:|
| Base | 0.9432 ± 0.0002 | 0.9409 | 0.9067 | 0.9646 | 0.9543 | 0.9495 |
| Final | **0.9459 ± 0.0000** | 0.9443 | 0.9101 | 0.9668 | 0.9564 | 0.9519 |
| Δ | +0.0027 | +0.0034 | +0.0034 | +0.0022 | +0.0021 | +0.0024 |

(d.e. del modelo final < 0.00005: las tres semillas dan 0.9459.)

Eficiencia de señal a tasa fija de fondo aceptado (media de 3 semillas):

| Fondo aceptado | Modelo | g | q | t | W | Z |
|---|---|---:|---:|---:|---:|---:|
| 10 % | Base | 0.820 | 0.698 | 0.915 | 0.846 | 0.834 |
| 10 % | Final | 0.831 | 0.710 | 0.919 | 0.850 | 0.840 |
| 1 % | Base | 0.414 | 0.163 | 0.647 | 0.646 | 0.708 |
| 1 % | Final | 0.424 | 0.169 | 0.665 | 0.651 | 0.710 |

Hallazgos:
1. **Validación externa del protocolo.** El AUC macro del modelo base
   (0.9432) coincide con el reportado de forma independiente para el mismo
   dataset y la misma arquitectura (0.9421, contribución en Indico CERN, 2026),
   igual que su accuracy (76.74 % vs. 76.64 %).
2. **El modelo final mejora el AUC en las cinco clases** (+0.002 a +0.003),
   sin excepción y de forma idéntica en las tres semillas.
3. **q no tiene una región "pura":** con 1 % de fondo solo se conserva el 17 %
   de los quarks. Su espacio de características se traslapa con g y con los
   bosones degradados en todos los umbrales.
4. **Z es la clase más pura a cortes estrictos** (71 % de eficiencia con 1 % de
   fondo, por encima de t), pero su curva se dobla con fuerza alrededor de
   TPR ≈ 0.75–0.8, donde cruza a W y t. Es coherente con la matriz de
   confusión: un núcleo de Z es muy separable (precisión 0.91), mientras que
   ~25 % de los Z (recall 0.74) se confunde con q y W. W muestra el mismo
   quiebre. *Interpretación (hipótesis):* esa fracción corresponde a bosones
   cuyos dos subchorros no se resuelven o pierden masa.
5. Por AUC el orden de dificultad es q < g < Z < W < t; por F1 era
   q < g < W < Z < t. W y Z intercambian lugar porque el AUC resume todos los
   umbrales y el F1 solo el de máxima probabilidad.