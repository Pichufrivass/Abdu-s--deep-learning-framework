#include "CNN.h"

kernel::kernel(int fk, int ck, int canales)
    : filtro(canales, vector<vector<shared_ptr<neurona>>>(fk, vector<shared_ptr<neurona>>(ck))){
    for (int c = 0; c < canales; c++) {
        for (int i = 0; i < fk; i++) {
            for (int j = 0; j < ck; j++) {
                filtro[c][i][j] = make_shared<neurona>(1);
            }
        }
    }
}

vector<vector<shared_ptr<value>>> kernel::operator()(vector<vector<vector<shared_ptr<value>>>>& imagenes, int fi, int ci, int fk, int ck) {
    int xtopmax = (fi - fk) + 1;
    int ytopmax = (ci - ck) + 1;
    int canales = imagenes.size();

    vector<vector<shared_ptr<value>>> mapa_salida(ytopmax, vector<shared_ptr<value>>(xtopmax));

    for (int y = 0; y < ytopmax; y++) {
        for (int x = 0; x < xtopmax; x++) {
            shared_ptr<value> suma1 = make_shared<value>(0.0);
            for (int c = 0; c < canales; c++) {
                for (int m = 0; m < fk; m++) {
                    for (int n = 0; n < ck; n++) {
                        shared_ptr<value> producto = multi(imagenes[c][y + m][x + n], filtro[c][m][n]->w[0]);
                        suma1 = suma(suma1, producto);
                    }
                }
            }
            mapa_salida[y][x] = suma1;
        }
    }

    return mapa_salida;
}

vector<shared_ptr<value>> kernel::parameter() {
    vector<shared_ptr<value>> p;
    for (size_t c = 0; c < filtro.size(); c++) {
        for (size_t i = 0; i < filtro[c].size(); i++) {
            for (size_t j = 0; j < filtro[c][i].size(); j++) {
                p.push_back(filtro[c][i][j]->w[0]); 
            }
        }
    }
    return p;
}

vector<vector<shared_ptr<value>>> maxpooling(vector<vector<shared_ptr<value>>>& feature_map, int fi, int ci, int fk, int ck) {
    int xtopmax = (fi - fk) + 1;
    int ytopmax = (ci - ck) + 1;
    vector<vector<shared_ptr<value>>> mapa_salida(ytopmax, vector<shared_ptr<value>>(xtopmax));

    for (int y = 0; y < ytopmax; y++) {
        for (int x = 0; x < xtopmax; x++) {
            shared_ptr<value> mayor_que = feature_map[y][x];
            for (int m = 0; m < fk; m++) {
                for (int n = 0; n < ck; n++) {
                    shared_ptr<value> producto = feature_map[y + m][x + n];
                    if (mayor_que->data < producto->data) {
                        mayor_que = producto;
                    }
                }
            }
            mapa_salida[y][x] = mayor_que;
        }
    }

    return mapa_salida;
}

vector<shared_ptr<value>> flatten(vector<vector<vector<shared_ptr<value>>>>& mapas) {
    vector<shared_ptr<value>> resultado;
    for (size_t c = 0; c < mapas.size(); c++) {
        for (size_t y = 0; y < mapas[c].size(); y++) {
            for (size_t x = 0; x < mapas[c][y].size(); x++) {
                resultado.push_back(mapas[c][y][x]);
            }
        }
    }
    return resultado;
}