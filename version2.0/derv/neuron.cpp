#include "neuron.h"
#include <random>
#include <cmath>

neurona::neurona(int nin, bool nonlin) : nonlin(nonlin){
	static random_device rd;
	static mt19937 gen(rd());
	double escala = 1.0 / sqrt((double)nin); 
	uniform_real_distribution<double> dis_w(-escala, escala);
	for (int i = 0; i < nin; i++){
		w.push_back(make_shared<value>(dis_w(gen)));
	}
	b = make_shared<value>(0.0);
}

shared_ptr<value> neurona::operator()(const vector<shared_ptr<value>>& x){
	shared_ptr<value> y = b;
	for(size_t i = 0; i < w.size(); i++){
		auto multiplicacion = multi(x[i], w[i]);
		y = suma(y, multiplicacion);
	}
	if(nonlin){
		return tanh_v(y);
	}
	return y;
}

vector<shared_ptr<value>> neurona::parameter(){
	vector<shared_ptr<value>> p = w;
	p.push_back(b);
	return p;
}

capas::capas(int entradas, int neu, bool nonlin_param){
	for(int i = 0; i < neu; i++){
		neurons.push_back(neurona(entradas, nonlin_param));
	}
}

vector<shared_ptr<value>> capas::operator()(const vector<shared_ptr<value>>& x){
	vector<shared_ptr<value>> salidas;
	for(size_t i = 0; i < neurons.size(); i++){
		salidas.push_back(neurons[i](x));
	}
	return salidas;
}

vector<shared_ptr<value>> capas::parameter(){
	vector<shared_ptr<value>> p;
	for(size_t i = 0; i < neurons.size(); i++){
		vector<shared_ptr<value>> neuron_p = neurons[i].parameter();
		p.insert(p.end(), neuron_p.begin(), neuron_p.end());
	}
	return p;
}

MLP::MLP(int entradas, const vector<int>& ncapas){
	vector<int> tamano = {entradas};
	tamano.insert(tamano.end(), ncapas.begin(), ncapas.end());
	for(size_t i = 0; i < ncapas.size(); i++){
		bool usar_tanh = (i < ncapas.size() - 1);
		capass.push_back(capas(tamano[i], tamano[i+1], usar_tanh));
	}
}

vector<shared_ptr<value>> MLP::operator()(vector<shared_ptr<value>>& x){
	for(size_t i = 0; i < capass.size(); i++){
		x = capass[i](x);
	}
	return x;
}

vector<shared_ptr<value>> MLP::parameter(){
	vector<shared_ptr<value>> p;
	for(size_t i = 0; i < capass.size(); i++){
		vector<shared_ptr<value>> capa_p = capass[i].parameter();
		p.insert(p.end(), capa_p.begin(), capa_p.end());
	}
	return p;
}