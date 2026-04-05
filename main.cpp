#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept> //Para lanzar errores
#include <algorithm> //Para usar max
#include <cstdlib> //Para usar numeros aleatorios rand()
using namespace std;

class Tensor; //Avisa que va existir una clase Tensor

//INTERFAZ DE TRANSFORMACIONES
//CLASE TensorTransform
class TensorTransform {
public:
    virtual ~TensorTransform() = default; //Aegura que al borrar un hijo se llama su destructor
    virtual Tensor apply(const Tensor& t) const = 0; //Cada clase hija debe implementar esto
};

//CLASE TENSOR
class Tensor {
    //Funciones amigas para realizar operaciones algebraicas que no pertenecen a un unico objeto Tensor
    friend class ReLU; //Acceder a los atributos privados de Tensor
    friend class Sigmoid; //Lo mismo acceder a atributos privados
    friend Tensor matmul(const Tensor& a, const Tensor& b); //La funcion matmul tambien
    friend Tensor dot(const Tensor& a, const Tensor& b); //La funcion dot tambien

private:
    vector<size_t> shape; //Guarda las dimensiones del Tensor , {2 , 3} = matriz 2 x 3
    double* data; //Puntero donde estan los valores del Tensor
    size_t total_size; //Cantidad total de elementos (va ser el producto de todas las dimensiones)

    // Multiplica todas las dimensiones para obtener el total de elementos
    size_t computeSize(const vector<size_t>& s) const {
        size_t n = 1;
        for (size_t d : s) n *= d;
        return n; // Devuelve el total, ej: {2,3} → 6
    }

public:
    //CONSTRUCTOR PRINCIPAL
    Tensor(const vector<size_t>& shape, const vector<double>& values) : shape(shape) {//Inicializa el atributo shape con el parametro recibido
        total_size = computeSize(shape); //Calcula cuantos elementos debe haber en total osea el producto
        if (total_size != values.size()) //Verifica que la cantidad de values conincida con las dimensiones
            throw runtime_error("El numero de valores no coincide con las dimensiones.");
        data = new double[total_size]; //Reserva memoria dinamica para guardar los values
        for (size_t i = 0; i < total_size; ++i) //Recorre cada posicion  {2 , 3} = matriz 2 x 3 = 6 = (0 1 2 3 4 5 )
            data[i] = values[i]; //Copia cada valor a la memoria
    }

    //CONSTRUCTOR DE COPIA (deep copy)
    Tensor(const Tensor& other) //llama al Tensor cuando hago B = A
        : shape(other.shape), total_size(other.total_size) { //Copia las dimensiones y el tamaño de otro Tensor
        data = new double[total_size]; //Reserva memoria
        for (size_t i = 0; i < total_size; ++i)
            data[i] = other.data[i]; //Copia cada valor a la memoria (deep copy)
    }

    //CONSTRUCTOR DE MOVIMIENTO (move constructor):
    Tensor(Tensor&& other) noexcept //Se llama cuando Tensor C = move(A)
        : shape(move(other.shape)), data(other.data), total_size(other.total_size) { //No copia el vector shape , toma al puntero al bloque de memoria del otro , copia el tamaño total
        other.data = nullptr; //Deja el puntero del otro en null para que no libere memoria
        other.total_size = 0; //Deja el tamaño del otro en 0
    }
    //DESTRUCTOR
    ~Tensor() {
        delete[] data; //Liberar memoria - evita fuga de memoria
    }

    // ASIGNACION DE COPIA  (operator=)
    Tensor& operator=(const Tensor& other) { //llama al Tensor cuando B = A
        if (this != &other) { //Verivica que no sean iguales
            double* tmp = new double[other.total_size]; //Reserva memoria
            for (size_t i = 0; i < other.total_size; ++i) //Recorre cada elemento
                tmp[i] = other.data[i]; //Copia los valores
            delete[] data; //Libera memoria
            data = tmp; //Apunta al nuevo bloque con los datos copiados
            shape = other.shape; //Copia las dimensiones
            total_size = other.total_size; //Conpia el tamaño total
        }
        return *this; //retorna la referencia al propio objeto
    }

    //ASIGNADOR DE MOVIMIENTO (move operator=)
    Tensor& operator=(Tensor&& other) noexcept { //Se llama A = move(B)
        if (this != &other) { //Verifica que no se autosigne
            delete[] data; //Libera memoria actual del tensor
            data = other.data; //Toma el puntero del otro
            shape = move(other.shape); //Roba el vector shape del otro
            total_size = other.total_size; //Copia el tamaño
            other.data = nullptr;  //Deja el otro sin puntero válido
            other.total_size = 0; // Deja el tamaño del otro en 0
        }
        return *this; // Devuelve referencia al objeto actual
    }

    //CREACION DE TENSORES PRE-DEFINIDOS
    //METODOS ESTATICOS

    // Tensor lleno de ceros
    static Tensor zeros(const vector<size_t>& shape) {
        size_t n = 1; //Inicializa el conteo en 1
        for (size_t d : shape) n *= d; //Apartir de ahi multiplica por 1
        return Tensor(shape, vector<double>(n, 0.0)); //Crea el Tensor con n ceros
    }

    // Tensor lleno de unos
    static Tensor ones(const vector<size_t>& shape) {
        size_t n = 1;
        for (size_t d : shape) n *= d; //Realiza lo mismo que con zeros
        return Tensor(shape, vector<double>(n, 1.0)); //Crea el Tensor con n unos
    }

    // Valores aleatorios uniformes en [min, max)
    static Tensor random(const vector<size_t>& shape, double min, double max) {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        vector<double> vals(n); //Crea un vector con n posiciones
        for (size_t i = 0; i < n; ++i) //Recorre cada posicion
            vals[i] = min + ((double)rand() / RAND_MAX) * (max - min); //Genera un numero aleatorio en [min , max)
        return Tensor(shape, vals); //Crea un tensor con valores aleatorios distribuidos uniformemente en el rango [min , max)
    }

    // Secuencia de enteros desde start hasta end (sin incluir end)
    static Tensor arange(int start, int end) {
        vector<double> vals; //Crea un vector vacio
        for (int i = start; i < end; ++i) //Recorre desde start hasta end
            vals.push_back((double)i); //Agrega el numero convertido a double
        return Tensor({(size_t)(end - start)}, vals); //Crea un Tensor de 1 dimeniones con esos valores
    }

    //METODO VIEW
    // Cambia la forma del tensor sin mover los datos subyacentes
    Tensor view(vector<size_t> new_shape) const {
        if (computeSize(new_shape) != total_size) // Verifica que el total de elementos no cambie
            throw runtime_error("La nueva forma no es compatible."); //Error si no coincide
        if (new_shape.size() > 3)  // Verifica que no supere 3 dimensiones
            throw runtime_error("Maximo 3 dimensiones.");
        Tensor t = *this;   // ← aquí hace deep copy (crea una copia completa del objeto , incluyendo datos y forma)
        t.shape = new_shape; // Reemplaza solo la forma con la nueva
        return t; //Retorna t con su nueva forma
    }

    //METODO UNSQUEEZE
    // Inserta una dimension de tamano 1 en la posicion indicada
    Tensor unsqueeze(size_t dim) const {
        vector<size_t> ns = shape; //Copia el vector con sus dimensiones
        if (dim > ns.size()) dim = ns.size(); //Si la posicion pedida es mayor al tamaño, la ajusta al final
        ns.insert(ns.begin() + dim, 1); //Inserta un 1 en la posicion indicada
        if (ns.size() > 3) //Garantizar que el numero de dimensiones no exceda tres
            throw runtime_error("Maximo 3 dimensiones.");
        Tensor t = *this; //Copia el tensor acutal
        t.shape = ns; //Asigna la nueva forma con la dimension extra
        return t; //retorna Tensor modificado
    }
    //METODO CONCATENACION
    // Une varios tensores a lo largo de un eje
    static Tensor concat(const vector<Tensor>& tensors, size_t axis) {
        if (tensors.empty()) //Verifica que haya al menos un tensor
            throw runtime_error("No hay tensores para concatenar.");

        const vector<size_t>& ref = tensors[0].shape; //Toma la forma del primer Tensor como referencia
        size_t ndim = ref.size(); //Va ser el numero de dimensiones de referencia

        if (axis >= ndim) //Verifica que en el eje exista dimensiones
            throw runtime_error("Eje fuera de rango.");

        for (size_t t = 1; t < tensors.size(); ++t) { //Recorre los demas Tensores
            const vector<size_t>& s = tensors[t].shape; //Obtiene la forma del Tensor actual
            if (s.size() != ndim)  // Verifica que tenga el mismo número de dimensiones
                throw runtime_error("Los tensores deben tener el mismo numero de dimensiones.");
            for (size_t d = 0; d < ndim; ++d) // Recorre cada dimensión
                if (d != axis && s[d] != ref[d]) // Si no es el eje de concatenación y los tamaños difieren
                    throw runtime_error("Dimensiones incompatibles para concat.");
        }

        vector<size_t> new_shape = ref; // La nueva forma comienza igual a la de referencia
        for (size_t t = 1; t < tensors.size(); ++t)   // Recorre los demás tensores
            new_shape[axis] += tensors[t].shape[axis];  // Suma el tamaño de cada tensor respectivamente a su eje

        size_t total = 1;   //Empieza el conteo para el total de elementos
        for (size_t d : new_shape) total *= d;    // Calcula total multiplicando todas las dimensiones nuevas

        vector<double> result(total); // Crea el vector resultado con el tamaño total
        size_t offset = 0;   // Posición actual donde copiar (empieza en 0)
        for (const Tensor& t : tensors) { //Recorre el Tensor
            for (size_t i = 0; i < t.total_size; ++i) // Recorre cada elemento del tensor
                result[offset + i] = t.data[i];  //Copia el elemento en la posición
            offset += t.total_size;  //Avanza para el siguiente Tensor
        }

        return Tensor(move(new_shape), result); // Devuelve el tensor concatenado (mueve el shape)
    }

    //SOBRECARGA DE OPERADORES
    //En cada una verifica si ambos Tensores tengan la misma forma si no envia un runtime_error (excepcion)
    // Suma elemento a elemento
    Tensor operator+(const Tensor& other) const {
        if (shape != other.shape) //Verifica que ambos Tensores tenga la misma forma
            throw runtime_error("Dimensiones incompatibles para suma."); //excepcion
        vector<double> res(total_size); //Crea un vector guarda los resultados
        for (size_t i = 0; i < total_size; ++i) //Recorre cada elemento
            res[i] = data[i] + other.data[i]; //Suma los elementos en la misma posicion
        return Tensor(shape, res); //Retorna el resultado  (X + Y).print();
    }

    // Resta elemento a elemento
    Tensor operator-(const Tensor& other) const {
        if (shape != other.shape)
            throw runtime_error("Dimensiones incompatibles para resta.");
        vector<double> res(total_size);
        for (size_t i = 0; i < total_size; ++i)
            res[i] = data[i] - other.data[i]; //La ressta de los elementos en la misma posicion
        return Tensor(shape, res); //Retorna el resultado  (X - Y).print();
    }

    // Multiplicacion elemento a elemento
    Tensor operator*(const Tensor& other) const {
        if (shape != other.shape)
            throw runtime_error("Dimensiones incompatibles para multiplicacion.");
        vector<double> res(total_size);
        for (size_t i = 0; i < total_size; ++i)
            res[i] = data[i] * other.data[i]; // Multiplica los elementos en la misma posición
        return Tensor(shape, res); //Retorna el resultado  (X * Y).print();
    }

    // Multiplicacion por escalar
    Tensor operator*(double scalar) const {
        vector<double> res(total_size);
        for (size_t i = 0; i < total_size; ++i)
            res[i] = data[i] * scalar;
        return Tensor(shape, res); //Retorna el resultado (X * 2.0).print();
    }

    // Suma con broadcast de bias (N x cols) + (1 x cols)
    Tensor addBias(const Tensor& bias) const {
        if (shape.size() != 2 || bias.shape.size() != 2) //Verifica que ambos sean tensores 2D
            throw runtime_error("addBias solo soporta tensores 2D.");
        size_t rows = shape[0]; //Numero de filas del tensor principal
        size_t cols = shape[1]; //Numero de columnas del tensor principal
        if (bias.shape[0] != 1 || bias.shape[1] != cols) //Verifica que el bias sea de forma {1, cols}
            throw runtime_error("El bias debe tener forma {1, cols}.");
        vector<double> res(total_size); //Crea vector para el resultado
        for (size_t i = 0; i < rows; ++i) //Recorre fil o rows
            for (size_t j = 0; j < cols; ++j) //Recorre cols
                res[i * cols + j] = data[i * cols + j] + bias.data[j]; //Suma el bias a cada elemento de la fila
        return Tensor(shape, res); //Retorna el tensor con el bias sumado
    }

    // METODO DE APLICACION EN CLASE TENSOR (POLIMORFISMO)
    Tensor apply(const TensorTransform& transform) const {
        return transform.apply(*this); //Llama al apply de la transformacion (ReLU o Sigmoid) pasando este tensor
    }

    void print() const {
        cout << "Tensor(shape={"; //Imprime la etiqueta inicial
        for (size_t i = 0; i < shape.size(); ++i) //Recorre cada dimension
            cout << shape[i] << (i + 1 < shape.size() ? ", " : ""); //Imprime la dimension
        cout << "}):" << endl;
        for (size_t i = 0; i < total_size; ++i) { //Recorre cada elemento
            cout << setw(6) << fixed << setprecision(2) << data[i] << " ";
            if (shape.size() > 1 && (i + 1) % shape.back() == 0) //Termina en una fila (Tensores 2D Y 3D)
                cout << endl; //salto de linea
        }
        cout << endl;
    }

    // Imprime solo las primeras n filas
    void printRows(size_t n) const {
        if (shape.size() != 2) { print(); return; } //Si no es 2D usa print() normal y sale
        size_t cols = shape[1];  //Numero de columnas
        size_t rows_to_print = min(n, shape[0]); //Cuantas filas va imprimir
        cout << "Tensor(shape={" << shape[0] << ", " << cols << "}) - primeras " << rows_to_print << " filas:" << endl;
        for (size_t i = 0; i < rows_to_print; ++i) { //Recorre las filas a imprimir
            for (size_t j = 0; j < cols; ++j) //Recorre cada columna
                cout << setw(6) << fixed << setprecision(4) << data[i * cols + j] << " ";
            cout << endl;
        }
        cout << endl;
    }

    const vector<size_t>& getShape() const { return shape; } //Devuelve la forma del tensor
    size_t getSize() const { return total_size; } //Devuelve la cantidad total de elementos
};

//Implelentacion de clases derivadas (clases hijas)
//CLASE: ReLU  →  y = max(0, x)
class ReLU : public TensorTransform { //Herda de la clase TensorTransform
public:
    Tensor apply(const Tensor& t) const override { //Esto simpre se llama cuando usas TensorTransform
        vector<double> vals(t.total_size); //Crea un vector para resultados
        for (size_t i = 0; i < t.total_size; ++i) //Recorre cada elemento
            vals[i] = max(0.0, t.data[i]); //Si un valor el negativo lo pone en 0 , si no lo deja igual
        return Tensor(t.shape, vals); //Retorna el Tensor Transformado
    }
};

// CLASE: Sigmoid  →  y = 1 / (1 + e^-x)
class Sigmoid : public TensorTransform { //Hereda de la clase TensorTransform
public:
    Tensor apply(const Tensor& t) const override { //Lo mismo llamo pro que es clase hija
        vector<double> vals(t.total_size);
        for (size_t i = 0; i < t.total_size; ++i)
            vals[i] = 1.0 / (1.0 + exp(-t.data[i])); //Aplico la formula a cada valor
        return Tensor(t.shape, vals); //Retorna el Tensor Transformado
    }
};

// Producto punto entre dos tensores del mismo tamaño
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) //Verifica que ambos tengan exactamente la misma forma
        throw runtime_error("Las formas deben coincidir para dot.");
    double suma = 0.0; //Acumulador para la suma de productos
    for (size_t i = 0; i < a.total_size; ++i)  //Recorre cada elemento
        suma += a.data[i] * b.data[i]; //Multiplica los elementos en la misma posicion y los acumula en suma
    return Tensor({1}, {suma});  //Retorna un tensor de 1 elemento con el resultado
}

// Multiplicacion de matrices 2D: A (m x k) * B (k x n) = C (m x n)
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) //Que sea 2D
        throw runtime_error("matmul solo acepta tensores 2D.");
    if (a.shape[1] != b.shape[0]) //verifica que las columnas de A == filas de B
        throw runtime_error("Columnas de A deben coincidir con filas de B.");

    size_t rows  = a.shape[0]; //Filas del resultado
    size_t inner = a.shape[1]; //Dimension compartida (cols A = rows B)
    size_t cols  = b.shape[1]; //Columnas del resultado

    vector<double> res(rows * cols, 0.0); //Crea vector resultado inicializado en ceros
    for (size_t i = 0; i < rows; ++i) //Recorre cada fila
        for (size_t j = 0; j < cols; ++j)  //Recorre cada columna
            for (size_t k = 0; k < inner; ++k) //Recorre la dimension
                res[i * cols + j] += a.data[i * inner + k] * b.data[k * cols + j]; //Acumula el producto para la celda (i,j)

    return Tensor({rows, cols}, res); //Retorna la matriz resultado de dimensiones (rows x cols)
}

int main() {
    try {
        // ---- Gestion de memoria ----
        cout << "--- Gestion de memoria ---" << endl;
        Tensor A = Tensor::zeros({2, 3});
        Tensor B = A;           // copia
        Tensor C = move(A);     // movimiento
        Tensor D = Tensor::arange(0, 6);
        D = B;                  // asignacion de copia
        Tensor E({1, 2}, {5.0, 6.0});
        E = move(D);            // asignacion de movimiento
        cout << "Gestion de memoria OK." << endl << endl;

        // ---- Metodos estaticos ----
        cout << "--- Metodos estaticos ---" << endl;

        Tensor Z = Tensor::zeros({2, 2});
        cout << "zeros:" << endl;
        Z.print();

        Tensor O = Tensor::ones({2, 2});
        cout << "ones:" << endl;
        O.print();

        Tensor R = Tensor::random({2, 2}, 0.0, 1.0);
        cout << "random:" << endl;
        R.print();

        Tensor AR = Tensor::arange(-3, 3);
        cout << "arange(-3, 3):" << endl;
        AR.print();

        // ---- View y Unsqueeze ----
        cout << "--- View y Unsqueeze ---" << endl;

        Tensor V = Tensor::arange(0, 6).view({2, 3});
        cout << "arange(0,6) -> view({2,3}):" << endl;
        V.print();

        Tensor U = V.unsqueeze(0);
        cout << "unsqueeze(0):" << endl;
        U.print();

        // ---- Operadores ----
        cout << "--- Sobrecarga de operadores ---" << endl;

        Tensor X({2, 2}, {1.0, 2.0, 3.0, 4.0});
        Tensor Y({2, 2}, {5.0, 6.0, 7.0, 8.0});

        cout << "X + Y:" << endl;
        (X + Y).print();

        cout << "X - Y:" << endl;
        (X - Y).print();

        cout << "X * Y (elem a elem):" << endl;
        (X * Y).print();

        cout << "X * 2.0 (escalar):" << endl;
        (X * 2.0).print();

        // Excepcion esperada por dimensiones incompatibles
        try {
            Tensor P = Tensor::ones({2, 3});
            Tensor Q = Tensor::ones({3, 2});
            Tensor bad = P + Q;
        } catch (const exception& e) {
            cout << "Excepcion capturada: " << e.what() << endl << endl;
        }

        // ---- Concat ----
        cout << "--- Concat ---" << endl;
        Tensor CA = Tensor::ones({2, 3});
        Tensor CB = Tensor::zeros({2, 3});
        Tensor CC = Tensor::concat({CA, CB}, 0);
        cout << "concat(ones, zeros) axis=0:" << endl;
        CC.print();

        // ---- ReLU y Sigmoid ----
        cout << "--- ReLU y Sigmoid ---" << endl;

        Tensor F = Tensor::arange(-5, 5);

        ReLU relu;
        cout << "ReLU sobre arange(-5, 5):" << endl;
        F.apply(relu).print();

        Sigmoid sigmoid;
        cout << "Sigmoid sobre arange(-5, 5):" << endl;
        F.apply(sigmoid).print();

        // ---- Dot ----
        cout << "--- Dot ---" << endl;
        Tensor DA({4}, {1.0, 2.0, 3.0, 4.0});
        Tensor DB({4}, {4.0, 3.0, 2.0, 1.0});
        cout << "dot([1,2,3,4], [4,3,2,1]):" << endl;
        dot(DA, DB).print();

        // ---- Matmul ----
        cout << "--- Matmul ---" << endl;
        Tensor M1({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        Tensor M2({3, 2}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
        cout << "M1 x M2:" << endl;
        matmul(M1, M2).print();

        // ---- Red neuronal ----
        cout << "--- Red Neuronal ---" << endl;

        // Paso 1: tensor de entrada 1000 x 20 x 20
        Tensor input = Tensor::random({1000, 20, 20}, 0.0, 1.0);

        // Paso 2: aplanar a 1000 x 400
        Tensor flat = input.view({1000, 400});

        // Pesos y bias primera capa: 400->100
        Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
        Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);

        // Paso 3 y 4: multiplicacion + bias
        Tensor h1 = matmul(flat, W1).addBias(b1);

        // Paso 5: activacion ReLU
        Tensor h1_act = h1.apply(relu);

        // Pesos y bias segunda capa: 100->10
        Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
        Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);

        // Paso 6 y 7: segunda capa lineal
        Tensor h2 = matmul(h1_act, W2).addBias(b2);

        // Paso 8: activacion Sigmoid
        Tensor output = h2.apply(sigmoid);

        cout << "Primeras 3 filas de la salida (1000 x 10):" << endl;
        output.printRows(3);
        cout << "Red neuronal ejecutada correctamente." << endl;

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
    }

    return 0;
}
