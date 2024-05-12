/*
 * rheology.cc
 *
 *  Created on: Feb 16, 2021
 *      Author: eai
 */

#include <cmath>

class Rheology
{
public:
	Rheology();
	double A;
	double Ea;
	double R;
	double Tref;

	double get_eta(unsigned int m_id, double T_local);
};



Rheology::Rheology(void)
{
	Ea   = 60.0;
	Tref = 273.0;
	R    = 8.31446261815324;
	A    = 1.0e13;
};

double Rheology::get_eta(unsigned int m_id, double T_local)
{
	// this is giving a Newtonian viscosity
	return A*exp(Ea/(R*Tref)*(Tref/T_local-1.0));
};
