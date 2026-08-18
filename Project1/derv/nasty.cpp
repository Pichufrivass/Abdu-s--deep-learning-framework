#include "nasty.h"
#include <cmath>

value::value(double value, string soy_param){
	data = value;
	grad = 0;
	soy = soy_param;
}
value::value(double value){
	data = value;
	grad = 0;
	soy = "";
}
void value::derivada(){
	if (soy == "*"){
		prev[0]->grad += prev[1]->data * grad;
		prev[1]->grad += prev[0]->data * grad;
	}
	if(soy == "+"){
		prev[0]->grad += grad;
		prev[1]->grad += grad;
	}
	if(soy == "-"){
		prev[0]->grad += grad;
		prev[1]->grad += -1 * grad;
	}
	if(soy == "^2"){
		prev[0]->grad += 2 * prev[0]->data * grad;
	}
	if(soy == "relu"){
		prev[0]->grad += (data > 0.0) ? grad : 0.0;
	}
	if(soy == "sigmoid"){
		prev[0]->grad += data * (1.0 - data) * grad;
	}
	if(soy == "tanh"){
		prev[0]->grad += (1.0 - data * data) * grad;
	}
}

shared_ptr<value> suma(const shared_ptr<value>& a, const shared_ptr<value>& b){
	auto out = make_shared<value>((a->data + b->data), "+");
	out->prev.push_back(a);
	out->prev.push_back(b);
	return out;
}
shared_ptr<value> multi(const shared_ptr<value>& a, const shared_ptr<value>& b){
	auto out = make_shared<value>((a->data * b->data), "*");
	out->prev.push_back(a);
	out->prev.push_back(b);
	return out;
}
shared_ptr<value> cuadrado(const shared_ptr<value>& a){
	auto out = make_shared<value>((a->data * a->data), "^2");
	out->prev.push_back(a);
	return out;
}
shared_ptr<value> resta(const shared_ptr<value>& a, const shared_ptr<value>& b){
	auto out = make_shared<value>((a->data - b->data), "-");
	out->prev.push_back(a);
	out->prev.push_back(b); 
	return out;
}
shared_ptr<value> relu(const shared_ptr<value>& a){
	double valor_salida = (a->data > 0.0) ? a->data : 0.0;
	auto out = make_shared<value>(valor_salida, "relu");
	out->prev.push_back(a);
	return out;
}
shared_ptr<value> sigmoid(const shared_ptr<value>& a){
	double valor_salida = 1.0 / (1.0 + exp(-a->data));
	auto out = make_shared<value>(valor_salida, "sigmoid");
	out->prev.push_back(a);
	return out;
}
shared_ptr<value> tanh_v(const shared_ptr<value>& a){
	double valor_salida = tanh(a->data);
	auto out = make_shared<value>(valor_salida, "tanh");
	out->prev.push_back(a);
	return out;
}
shared_ptr<value> perdida(shared_ptr<value>& diff, const shared_ptr<value>& yreal, const shared_ptr<value>& ycausa){
	diff = resta(yreal, ycausa);
	return cuadrado(diff);
}
void ftopo(const shared_ptr<value>& v, vector<shared_ptr<value>>& topo, set<shared_ptr<value>>& visited){
	if(visited.find(v) == visited.end()){
		visited.insert(v);
		for(const auto& p : v->prev){
			ftopo(p, topo, visited);
		}
		topo.push_back(v);
	}
}
void backward(const shared_ptr<value>& nodo_raiz){
	vector<shared_ptr<value>> topo;
	set<shared_ptr<value>> visited;
	ftopo(nodo_raiz, topo, visited);
	nodo_raiz->grad = 1.0;
	for(auto it = topo.rbegin(); it != topo.rend(); ++it){
		(*it)->derivada();
	}
}
void actualizar(const shared_ptr<value>& w, double rate){
	w->data = w->data - (rate * w->grad);
}
