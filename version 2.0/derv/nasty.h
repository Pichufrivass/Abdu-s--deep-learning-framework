#ifndef NASTY_H
#define NASTY_H
#include<iostream>
#include<vector>
#include<memory>
#include<set>
using namespace std;

class value {
public:
	double data;
	double grad;
	string soy;
	
	vector<shared_ptr<value>> prev;
	
	value(double value, string soy_param);
	value(double value);
	
	void derivada();
};
shared_ptr<value> suma(const shared_ptr<value>& a, const shared_ptr<value>& b);
shared_ptr<value> multi(const shared_ptr<value>& a, const shared_ptr<value>& b);
shared_ptr<value> resta(const shared_ptr<value>& a, const shared_ptr<value>& b);
shared_ptr<value> relu(const shared_ptr<value>& a);
shared_ptr<value> sigmoid(const shared_ptr<value>& a);
shared_ptr<value> tanh_v(const shared_ptr<value>& a);
shared_ptr<value> cuadrado(const shared_ptr<value>& a);
void ftopo(const shared_ptr<value>& v, vector<shared_ptr<value>>& topo, set<shared_ptr<value>>& visited);
void backward(const shared_ptr<value>& nodo_raiz);
shared_ptr<value> perdida(shared_ptr<value>& diff, const shared_ptr<value>& yreal, const shared_ptr<value>& ycausa);
void actualizar(const shared_ptr<value>& w, double rate);

#endif