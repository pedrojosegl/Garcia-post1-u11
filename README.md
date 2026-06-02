# Post-Contenido 1 — CUDA Benchmark CPU vs GPU
Arquitectura de Computadores — Unidad 11
Ingeniería de Sistemas — UFPS 2026

## Entorno de trabajo
| Parámetro      | Valor                         |
|----------------|-------------------------------|
| GPU            | NVIDIA Tesla T4 (Google Colab)|
| CUDA version   | 12.8 (nvcc) / 13.0 (driver)  |
| OS             | Ubuntu 22.04 (Colab)          |
| Compilador     | nvcc / gcc 11                 |

## Compilación
```bash
nvcc -O2 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -o matMul    src/matMul.cu
```

## Resultados — vectorAdd (N = 16M elementos)
| Medición               | Tiempo    |
|------------------------|-----------|
| CPU                    | 40.94 ms  |
| GPU kernel             | 91.48 ms  |
| GPU total (con memcpy) | 73.75 ms  |
| Errores                | 0         |

## Resultados — matMul (C = A × B)
| N    | Naïve GPU (ms) | Tiled GPU (ms) | Speedup  | Errores |
|------|---------------|---------------|----------|---------|
| 512  | 29.96         | 0.46          | 65.51×   | 0       |
| 1024 | 9.20          | 5.81          | 1.58×    | 0       |

## Análisis

**¿Por qué el kernel tiled es más rápido que el naïve para N=512?**
El kernel con tiling carga bloques de 16×16 elementos en shared memory antes
de operar sobre ellos. La shared memory tiene latencia ~100x menor que la
memoria global de la GPU. En el kernel naïve, cada multiplicación accede
directamente a memoria global, generando miles de accesos lentos redundantes.
Con tiling, cada dato se carga una sola vez a shared memory y se reutiliza
TILE=16 veces, reduciendo los accesos a memoria global por un factor de 16
y logrando un speedup de 65×.

**¿Por qué el tiempo total GPU (con memcpy) puede superar al CPU?**
Las transferencias PCIe entre la RAM del host y la memoria de la GPU tienen
una latencia fija considerable. Para N pequeño, el cálculo es tan rápido
que ese overhead de transferencia domina el tiempo total, haciendo que la
GPU sea más lenta en conjunto que la CPU. Solo cuando N es suficientemente
grande el speedup del kernel compensa el costo de las transferencias de datos.

## Capturas de checkpoints
- capturas/checkpoint 1.png
- capturas/checkpoint 2.png
- capturas/checkpoint 3.png
