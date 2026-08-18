#include <vector>
#ifndef CNN_H
#define CNN_H

#include <vector>
#include <memory>
#include "nasty.h"
#include "neuron.h"

using namespace std;

class kernel {
public:
    vector<vector<vector<shared_ptr<neurona>>>> filtro;

    kernel(int fk, int ck, int canales);

    vector<vector<shared_ptr<value>>> operator()(vector<vector<vector<shared_ptr<value>>>>& imagenes, int fi, int ci, int fk, int ck);
};

vector<vector<shared_ptr<value>>> maxpooling(vector<vector<shared_ptr<value>>>& feature_map, int fi, int ci, int fk, int ck);

#endif