#ifndef CPAS_H
#define CPAS_H

#include <vector>
#include <memory>
#include "nasty.h"
#include "CNN.h"

using namespace std;

class capa_convulsion {
public:
    vector<shared_ptr<kernel>> filtros;

    capa_convulsion(int fk, int ck, int nf, int canales_entrada);

    vector<vector<vector<shared_ptr<value>>>> operator()(vector<vector<vector<shared_ptr<value>>>>& imagenes, int fi, int ci, int fk, int ck, int mfk, int mck);

    vector<shared_ptr<value>> parameter();
};

class CNN {
public:
    vector<shared_ptr<capa_convulsion>> capas;

    CNN(int fi, int ci, int fk, int ck, int nf, int nc, int mfk, int mck);

    vector<vector<vector<shared_ptr<value>>>> operator()(vector<vector<vector<shared_ptr<value>>>>& imagen, int fi, int ci, int fk, int ck, int mfk, int mck);

    vector<shared_ptr<value>> parameter();
};

#endif