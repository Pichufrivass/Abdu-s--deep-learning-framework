#ifndef NEURON_H
#define NEURON_H
#include<vector>
#include<memory>
#include"nasty.h"
using namespace std;

class neurona{
public:
	vector<shared_ptr<value>> w;
	shared_ptr<value> b;
	bool nonlin;
	
	neurona(int entradas, bool nonlin_param = true);
	
	shared_ptr<value> operator()(const vector<shared_ptr<value>>& x);
	vector<shared_ptr<value>> parameter();
};

class capas{
public:
	vector<neurona> neurons;
	
	capas(int entradas, int neu, bool nonlin_param = true);
	
	vector<shared_ptr<value>> operator()(const vector<shared_ptr<value>>& x);
	vector<shared_ptr<value>> parameter();
};

class MLP {
public:
	vector<capas> capass;
	
	MLP(int entradas, const vector<int>& ncapas);
	
	vector<shared_ptr<value>> operator()(vector<shared_ptr<value>>& x);
	vector<shared_ptr<value>> parameter();
};

#endif