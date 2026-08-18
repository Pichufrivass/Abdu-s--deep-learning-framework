#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <algorithm>
#include "derv/nasty.h"
#include "derv/neuron.h"
#include "derv/CNN.h"
#include "derv/cpas.h"
using namespace std;

vector<string> plantilla_0 = {
    "00111100",
    "01000010",
    "10000001",
    "10000001",
    "10000001",
    "10000001",
    "01000010",
    "00111100"
};
vector<string> plantilla_1 = {
    "00011000",
    "00111000",
    "01011000",
    "00011000",
    "00011000",
    "00011000",
    "00011000",
    "01111100"
};
vector<string> plantilla_2 = {
    "01111100",
    "10000010",
    "00000010",
    "00000100",
    "00001000",
    "00010000",
    "00100000",
    "11111110"
};
vector<string> plantilla_3 = {
    "11111100",
    "00000010",
    "00000100",
    "00111000",
    "00000100",
    "00000010",
    "10000010",
    "01111100"
};

vector<vector<string>> plantillas = { plantilla_0, plantilla_1, plantilla_2, plantilla_3 };
const int NUM_CLASES = 4;
const int TAM = 8; 

vector<vector<vector<shared_ptr<value>>>> plantilla_a_imagen(const vector<string>& plantilla, int pixeles_ruido, mt19937& gen) {
    vector<vector<double>> pix(TAM, vector<double>(TAM));
    for (int y = 0; y < TAM; y++)
        for (int x = 0; x < TAM; x++)
            pix[y][x] = (plantilla[y][x] == '1') ? 1.0 : 0.0;

    uniform_int_distribution<int> pos(0, TAM - 1);
    for (int i = 0; i < pixeles_ruido; i++) {
        int y = pos(gen), x = pos(gen);
        pix[y][x] = 1.0 - pix[y][x];
    }

    vector<vector<shared_ptr<value>>> canal(TAM, vector<shared_ptr<value>>(TAM));
    for (int y = 0; y < TAM; y++)
        for (int x = 0; x < TAM; x++)
            canal[y][x] = make_shared<value>(pix[y][x]);

    return { canal }; 
}

void imprimir_imagen(const vector<string>& plantilla) {
    for (auto& fila : plantilla) cout << "  " << fila << endl;
}

int main() {
    static random_device rd;
    static mt19937 gen(rd());

    int fi = TAM, ci = TAM;
    int fk = 3, ck = 3;
    int nf = 4;
    int nc = 1;
    int mfk = 2, mck = 2;

    CNN mi_cnn(fi, ci, fk, ck, nf, nc, mfk, mck);

    int alto = ci, ancho = fi;
    for (int i = 0; i < nc; i++) {
        ancho = ancho - fk + 1; alto = alto - ck + 1;
        ancho = ancho - mfk + 1; alto = alto - mck + 1;
    }
    int tam_entrada_mlp = nf * alto * ancho;
    cout << "Feature map final: " << alto << "x" << ancho << " x " << nf << " filtros = " << tam_entrada_mlp << " entradas a la MLP" << endl;

    MLP mi_mlp(tam_entrada_mlp, {16, NUM_CLASES});

    vector<shared_ptr<value>> params = mi_cnn.parameter();
    vector<shared_ptr<value>> params_mlp = mi_mlp.parameter();
    params.insert(params.end(), params_mlp.begin(), params_mlp.end());
    cout << "Total de parametros entrenables: " << params.size() << endl << endl;

    struct Ejemplo { vector<vector<vector<shared_ptr<value>>>> imagen; int clase; };
    int copias_por_clase = 25;
    int ruido_max = 4; 

    vector<Ejemplo> dataset;
    uniform_int_distribution<int> dis_ruido(0, ruido_max);
    for (int c = 0; c < NUM_CLASES; c++) {
        for (int k = 0; k < copias_por_clase; k++) {
            int ruido = dis_ruido(gen);
            auto img = plantilla_a_imagen(plantillas[c], ruido, gen);
            dataset.push_back({ img, c });
        }
    }

    int epocas = 40;
    double rate = 0.03;

    for (int epoca = 0; epoca < epocas; epoca++) {
        shuffle(dataset.begin(), dataset.end(), gen);
        double loss_total = 0.0;

        for (auto& ej : dataset) {
            for (auto& p : params) p->grad = 0.0;

            auto salida_cnn = mi_cnn(ej.imagen, fi, ci, fk, ck, mfk, mck);
            auto vector_plano = flatten(salida_cnn);
            auto y_pred = mi_mlp(vector_plano);

            shared_ptr<value> loss = make_shared<value>(0.0);
            for (int i = 0; i < NUM_CLASES; i++) {
                double etiqueta = (i == ej.clase) ? 1.0 : 0.0;
                auto y_real_i = make_shared<value>(etiqueta);
                shared_ptr<value> diff;
                auto termino = perdida(diff, y_real_i, y_pred[i]);
                loss = suma(loss, termino);
            }

            backward(loss);
            for (auto& p : params) actualizar(p, rate);

            loss_total += loss->data;
        }

        if (epoca % 5 == 0 || epoca == epocas - 1) {
            cout << "Epoca " << epoca << " - loss promedio: " << (loss_total / dataset.size()) << endl;
        }
    }

    cout << "\n=== Predicciones sobre plantillas limpias ===" << endl;
    int aciertos = 0;
    for (int c = 0; c < NUM_CLASES; c++) {
        auto img = plantilla_a_imagen(plantillas[c], 0, gen); // sin ruido
        auto salida_cnn = mi_cnn(img, fi, ci, fk, ck, mfk, mck);
        auto vector_plano = flatten(salida_cnn);
        auto y_pred = mi_mlp(vector_plano);

        int mejor_clase = 0;
        double mejor_valor = y_pred[0]->data;
        for (int i = 1; i < NUM_CLASES; i++) {
            if (y_pred[i]->data > mejor_valor) { mejor_valor = y_pred[i]->data; mejor_clase = i; }
        }

        cout << "\nDigito real: " << c << endl;
        imprimir_imagen(plantillas[c]);
        cout << "  Prediccion: " << mejor_clase << (mejor_clase == c ? "  <-- correcto" : "  <-- INCORRECTO") << endl;
        cout << "  Activaciones: ";
        for (int i = 0; i < NUM_CLASES; i++) cout << "[" << i << "]=" << y_pred[i]->data << " ";
        cout << endl;

        if (mejor_clase == c) aciertos++;
    }

    cout << "\nAccuracy en plantillas limpias: " << aciertos << "/" << NUM_CLASES << endl;

    cout << "\n=== Predicciones con ruido (2-3 pixeles volteados, 5 pruebas por clase) ===" << endl;
    int aciertos_ruido = 0, total_ruido = 0;
    uniform_int_distribution<int> ruido_test(2, 3);
    for (int c = 0; c < NUM_CLASES; c++) {
        for (int t = 0; t < 5; t++) {
            auto img = plantilla_a_imagen(plantillas[c], ruido_test(gen), gen);
            auto salida_cnn = mi_cnn(img, fi, ci, fk, ck, mfk, mck);
            auto vector_plano = flatten(salida_cnn);
            auto y_pred = mi_mlp(vector_plano);

            int mejor_clase = 0;
            double mejor_valor = y_pred[0]->data;
            for (int i = 1; i < NUM_CLASES; i++) {
                if (y_pred[i]->data > mejor_valor) { mejor_valor = y_pred[i]->data; mejor_clase = i; }
            }
            total_ruido++;
            if (mejor_clase == c) aciertos_ruido++;
        }
    }
    cout << "Accuracy con ruido: " << aciertos_ruido << "/" << total_ruido << endl;

    return 0;
}