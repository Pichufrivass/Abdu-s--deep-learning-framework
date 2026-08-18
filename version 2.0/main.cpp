#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include "derv/nasty.h"
#include "derv/neuron.h"
using namespace std;

struct ejemplo{
	double edad;
	double resultado_real;
};

int main() {
	vector<ejemplo> dataset;
	
	static random_device rd;
	static mt19937 gen(rd());
	uniform_real_distribution<double> dis_edad(0.0, 1.0);
	
	for(int i = 0; i < 100; i++){
		double edad_normalizada = dis_edad(gen);
		double puede_entrar = 0.0;
		
		if (edad_normalizada >= 0.25 && edad_normalizada <= 0.70){  
			puede_entrar = 1.0;
		}
		
		dataset.push_back({edad_normalizada, puede_entrar});
	}
	
	MLP red(1, {4, 4, 1});
	auto params = red.parameter();
	
	for(int epoca = 0; epoca < 5000; epoca++){
		for (const auto& ej : dataset){
			for (auto p : params) p->grad = 0.0;
			
			vector<shared_ptr<value>> input = {make_shared<value>(ej.edad)};
			auto yreal = make_shared<value>(ej.resultado_real);
			
			vector<shared_ptr<value>> outputs = red(input);
			shared_ptr<value> ypred = sigmoid(outputs[0]);
			
			shared_ptr<value> diff;
			shared_ptr<value> L = perdida(diff, yreal, ypred);
			
			backward(L);
			
			for (auto p : params){
				actualizar(p, 0.01);
			}
		}
	}
	
	vector<double> edades_test = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80};
	
	for (double e : edades_test){
		vector<shared_ptr<value>> input_test = {make_shared<value>(e)};
		auto prediccion = sigmoid(red(input_test)[0]);
		
		int edad_real = static_cast<int>(e * 100);
		string esperado = (edad_real >= 25 && edad_real <= 70) ? "SI (1.0)" : "NO (0.0)";
		
		cout << "edad: " << edad_real << " anos | esperado: " << esperado
		     << " | probabilidad IA: " << prediccion->data << endl;
	}
	
	return 0;
}
