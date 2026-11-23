# Ficha del dataset — hls4ml LHC Jets (high-level features)

## Identificación
| Campo | Valor |
|---|---|
| Nombre | `hls4ml_lhc_jets_hlf` |
| Repositorio / enlace oficial | OpenML, id 42468 — https://www.openml.org/d/42468 |
| Fuente original | HLS4ML LHC Jet dataset, Zenodo — https://doi.org/10.5281/zenodo.3602260 |
| Creadores | Colaboración hls4ml / FastML (CERN, Fermilab, UC San Diego, Rutgers) |
| Subido a OpenML por | Sioni Summers (02-06-2020) |
| Licencia | CC BY 4.0 (Attribution) |

## Objetivo predictivo
Clasificación supervisada multiclase (5 clases): identificar la partícula que
originó un jet en colisiones protón-protón simuladas del LHC
(√s = 13 TeV, jets anti-kT con R = 0.8 y pT ≈ 1 TeV), a partir de 16
observables físicos de alto nivel de la subestructura del jet.

| Clase | Partícula que origina el jet | Estructura esperada |
|---|---|---|
| g | gluon | un núcleo, radiación más ancha |
| q | quark ligero | un núcleo, más estrecho que g |
| w | bosón W (≈ 80 GeV) | dos subchorros |
| z | bosón Z (≈ 91 GeV) | dos subchorros |
| t | quark top (≈ 173 GeV) | tres subchorros |

## Tamaño y clases
- Muestras: **830,000** · Características: **16** · Clases: **5**
- Valores faltantes: **0** · Filas duplicadas: **0**

| Clase | Jets | % |
|---|---:|---:|
| g | 167,125 | 20.14 |
| q | 161,066 | 19.41 |
| t | 167,851 | 20.22 |
| w | 167,083 | 20.13 |
| z | 166,875 | 20.11 |

Dataset prácticamente balanceado (razón máx./mín. = 1.04).

## Tipos de variables
Las 16 características son numéricas continuas (float), excepto `multiplicity`,
que es un conteo entero (188 valores distintos, rango 6–212). La etiqueta
`class` es nominal.

| Variable | Significado físico |
|---|---|
| `zlogz` | Σ z·log z sobre los constituyentes (z = fracción de pT) |
| `c1_b0/b1/b2_mmdt` | funciones de correlación de energía C₁ con β = 0, 1, 2 |
| `c2_b1/b2_mmdt` | razón de correlación de energía C₂ (β = 1, 2) |
| `d2_b1/b2_mmdt` | razón D₂ (β = 1, 2), sensible a estructura de dos subchorros |
| `d2_a1_b1/b2_mmdt` | razón generalizada D₂ con (α, β) = (1, 1) y (1, 2) |
| `m2_b1/b2_mmdt`, `n2_b1/b2_mmdt` | razones generalizadas M₂ y N₂ |
| `mass_mmdt` | masa del jet tras *grooming* (modified mass-drop tagger), GeV |
| `multiplicity` | número de constituyentes del jet |

El sufijo `_mmdt` indica que el observable se calcula sobre el jet depurado con
el *modified mass-drop tagger*.
<!-- TODO: verificar definiciones exactas en Duarte et al. (2018), Sec. 2 / Moult, Necib y Thaler (2016). -->

## Partición
OpenML no prescribe una partición oficial, y los jets son eventos simulados
independientes (no hay grupos ni orden temporal). Por lo tanto, según la
consigna, se usó **70/15/15 estratificado con semilla 42**:

| Conjunto | Jets | g | q | t | w | z |
|---|---:|---:|---:|---:|---:|---:|
| Entrenamiento | 581,000 | 0.201 | 0.194 | 0.202 | 0.201 | 0.201 |
| Validación | 124,500 | 0.201 | 0.194 | 0.202 | 0.201 | 0.201 |
| Prueba | 124,500 | 0.201 | 0.194 | 0.202 | 0.201 | 0.201 |

Índices guardados en `splits/split_seed42.npz`.

## Desafíos identificados
1. **Clases físicamente parecidas.** q/g tienen un solo núcleo; W y Z tienen la
   misma topología de dos subchorros con masas separadas por ~11 GeV. Antes de
   entrenar se esperaba que los errores se concentraran en q↔g y W↔Z. Los
   resultados confirmaron q↔g, pero mostraron que la confusión dominante es
   W/Z → q, y que W↔Z es menor y asimétrica (ver `docs/RESULTADOS.md`).
2. **Redundancia entre características.** `d2_b1_mmdt` y `d2_a1_b1_mmdt` son
   idénticas (|r| = 1.000, mismos estadísticos): con (α, β) = (1, 1) el
   observable generalizado se reduce al D₂ con β = 1. Otros 9 pares tienen
   |r| > 0.92, así que la información efectiva es menor que 16 dimensiones.
3. **Distribuciones asimétricas y escalas muy distintas.** Asimetría entre −2.2
   (`c1_b0_mmdt`) y 2.4 (`c2_b2_mmdt`); `mass_mmdt` llega a 573 GeV mientras
   otros observables viven en [0, 0.1]. Requiere estandarización.
4. **Inconsistencia en los metadatos del repositorio.** El md5 registrado en
   OpenML no coincide con el archivo servido (dos descargas idénticas con el
   mismo hash). Se resolvió descargando el ARFF directamente, validando su
   estructura y registrando sus hashes en `data/DATA_INFO.json`.
5. **Escala.** 830 mil muestras: el entrenamiento debe ser eficiente (minilotes
   en el dispositivo, sin DataLoader por muestra).
6. **Límite físico del accuracy.** Con solo observables de alto nivel, las
   clases se traslapan de forma irreducible; no se espera accuracy cercano a 100 %.

## Condiciones de uso y citación
Licencia CC BY 4.0: uso libre con atribución. Citar:

- Duarte, J., et al. (2018). Fast inference of deep neural networks in FPGAs for
  particle physics. *Journal of Instrumentation*, 13(07), P07027.
  https://doi.org/10.1088/1748-0221/13/07/P07027
- Pierini, M., Duarte, J. M., Tran, N., & Freytsis, M. (2020). *HLS4ML LHC Jet
  dataset (150 particles)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3602260
  <!-- TODO: confirmar lista de autores en la página de Zenodo -->
- Vanschoren, J., van Rijn, J. N., Bischl, B., & Torgo, L. (2013). OpenML:
  networked science in machine learning. *SIGKDD Explorations*, 15(2), 49–60.
