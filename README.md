# Tensor++

Una librería de tensores implementada en C++ inspirada en NumPy y PyTorch, capaz de manejar tensores de hasta 3 dimensiones, realizar operaciones matemáticas avanzadas y construir redes neuronales simples.

> **Programación III — Tarea #1**  
> Universidad de Ingeniería y Tecnología (UTEC) — 2026-1  
> Profesor: José A. Chávez Álvarez

---

## Contenido del repositorio

```
.
├── tensor.cpp      # Código fuente completo de la librería y main de prueba
└── README.md       # Este archivo
```

---

## Requisitos

- Compilador C++ con soporte para **C++11 o superior** (g++, clang++)
- Sistema operativo: Linux, macOS o Windows (con MinGW / WSL)

---

## Compilación

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
g++ -std=c++11 -o tensor tensor.cpp
```

O si quieres activar optimizaciones:

```bash
g++ -std=c++11 -O2 -o tensor tensor.cpp
```

---

## Ejecución

```bash
./tensor
```

En Windows (MinGW):

```bash
tensor.exe
```

---

## Estructura del código

### Clase principal: `Tensor`

Almacena los datos en un **array dinámico contiguo** (`double*`) y gestiona su ciclo de vida completo mediante:

| Método | Descripción |
|---|---|
| Constructor principal | Recibe `shape` y `values`, valida tamaños |
| Constructor de copia | Deep copy de datos |
| Constructor de movimiento | Transfiere propiedad del puntero |
| `operator=` (copia) | Libera memoria actual y copia los nuevos datos |
| `operator=` (movimiento) | Libera recursos actuales y toma los del temporal |
| Destructor | Libera toda la memoria dinámica con `delete[]` |

### Métodos estáticos de creación

```cpp
Tensor::zeros({2, 3})           // Tensor de ceros
Tensor::ones({3, 3})            // Tensor de unos
Tensor::random({2, 2}, 0.0, 1.0) // Valores aleatorios en [min, max)
Tensor::arange(0, 6)            // Secuencia 0, 1, 2, 3, 4, 5
```

### Modificación de dimensiones

```cpp
Tensor B = A.view({2, 3});   // Reinterpreta la forma sin copiar datos
Tensor C = A.unsqueeze(0);   // Inserta dimensión de tamaño 1 en posición 0
```

### Operadores sobrecargados

```cpp
Tensor C = A + B;    // Suma elemento a elemento
Tensor D = A - B;    // Resta elemento a elemento
Tensor E = A * B;    // Multiplicación elemento a elemento
Tensor F = A * 2.0;  // Multiplicación por escalar
```

> Si las dimensiones son incompatibles, se lanza una excepción `runtime_error`.

### Concatenación

```cpp
Tensor C = Tensor::concat({A, B}, 0);  // Une tensores a lo largo del eje 0
```

### Transformaciones (Polimorfismo)

La interfaz abstracta `TensorTransform` define el método `apply`. Las clases derivadas son:

| Clase | Fórmula |
|---|---|
| `ReLU` | `y = max(0, x)` |
| `Sigmoid` | `y = 1 / (1 + e^-x)` |

```cpp
ReLU relu;
Tensor B = A.apply(relu);

Sigmoid sigmoid;
Tensor C = A.apply(sigmoid);
```

### Funciones amigas

```cpp
dot(A, B);      // Producto punto entre tensores de igual forma
matmul(A, B);   // Multiplicación matricial 2D (m×k) * (k×n) = (m×n)
```

---

## Ejemplo: Red Neuronal

El `main` incluye la construcción y ejecución de una red neuronal de dos capas:

```
Entrada:   1000 × 20 × 20
  → view   1000 × 400
  → matmul W1 (400×100) + bias b1 (1×100)   → 1000 × 100
  → ReLU                                      → 1000 × 100
  → matmul W2 (100×10)  + bias b2 (1×10)    → 1000 × 10
  → Sigmoid                                   → 1000 × 10
```

La salida imprime las primeras 3 filas del tensor de salida `(1000 × 10)`.

---

## Salida esperada (fragmento)

```
--- Gestion de memoria ---
Gestion de memoria OK.

--- Metodos estaticos ---
zeros:
Tensor(shape={2, 2}):
  0.00   0.00
  0.00   0.00
...

--- Red Neuronal ---
Primeras 3 filas de la salida (1000 x 10):
Tensor(shape={1000, 10}) - primeras 3 filas:
...
Red neuronal ejecutada correctamente.
```

---

## Autores

| Nombre | Código |
|---|---|
| [Nombre del integrante 1] | [Código] |
| [Nombre del integrante 2] | [Código] |
