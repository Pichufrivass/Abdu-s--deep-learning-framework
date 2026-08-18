#include <memory>
#include "nasty.h"
#include "cnn.h"

class capa_convulsion {
public:
    vector<shared_ptr<kernel>> filtros;

    capa_convulsion(int fk, int ck, int nf, int canales_entrada) {
        for (int i = 0; i < nf; i++) {
            filtros.push_back(make_shared<kernel>(fk, ck, canales_entrada));
        }
    }

    vector<vector<vector<shared_ptr<value>>>> operator()(vector<vector<vector<shared_ptr<value>>>>& imagenes, int fi, int ci, int fk, int ck, int mfk, int mck) {
        vector<vector<vector<shared_ptr<value>>>> salidasF;

        for (size_t i = 0; i < filtros.size(); i++) {
            vector<vector<shared_ptr<value>>> mapa = (*filtros[i])(imagenes, fi, ci, fk, ck);
            
            for (size_t y = 0; y < mapa.size(); y++) {
            	for (size_t x = 0; x < mapa[y].size(); x++) {
                	mapa[y][x] = relu(mapa[y][x]);
            	}
        	}

            int mapa_h = mapa.size();
            int mapa_w = mapa[0].size();

            salidasF.push_back(maxpooling(mapa, mapa_h, mapa_w, mfk, mck));
        }

        return salidasF;
    }
};
class CNN {
public:
    vector<shared_ptr<capa_convulsion>> capas;

    CNN(int fi, int ci, int fk, int ck, int nf, int nc, int mfk, int mck) {
        int alto = ci;
        int ancho = fi;
        for (int i = 0; i < nc; i++) {
            ancho = ancho - fk + 1;
            alto  = alto  - ck + 1;
            if (ancho < 1 || alto < 1) {
                throw std::runtime_error("No se puede crear esta CNN: demasiadas capas");
            }
            ancho = ancho - mfk + 1;
            alto  = alto  - mck + 1;
            if (ancho < 1 || alto < 1) {
                throw std::runtime_error("No se puede crear esta CNN: demasiadas capas");
            }
        }
        for (int i = 0; i < nc; i++) {
            int canales_entrada = (i == 0) ? 1 : nf;
            capas.push_back(make_shared<capa_convulsion>(fk, ck, nf, canales_entrada));
        }
    }

    vector<vector<vector<shared_ptr<value>>>> operator()(vector<vector<vector<shared_ptr<value>>>>& imagen, int fi, int ci, int fk, int ck, int mfk, int mck) {
        vector<vector<vector<shared_ptr<value>>>> salida_actual = imagen;
        int alto = ci;
        int ancho = fi;

        for (size_t i = 0; i < capas.size(); i++) {
            salida_actual = (*capas[i])(salida_actual, ancho, alto, fk, ck, mfk, mck);

            ancho = ancho - fk + 1;
            alto  = alto  - ck + 1;
            ancho = ancho - mfk + 1;
            alto  = alto  - mck + 1;
        }

        return salida_actual;
    }
};