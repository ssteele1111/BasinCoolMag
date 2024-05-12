/*
 * heat_transfer.cc
 *
 *  Created on: Feb 16, 2021
 *      Author: eai
 */


// Heat transfer class
class HeatTransfer
{
public:
	HeatTransfer(vector<double> in_rho, vector<double> in_k, vector<double> in_c_p);
	vector<double> rho;
	vector<double> k;
	vector<double> c_p;
	double get_alpha(unsigned int m_id, double T_local);
	double get_kappa(unsigned int m_id, double T_local, double depth);
	double get_cp(unsigned int m_id, double T_local);

};


HeatTransfer::HeatTransfer(vector<double> in_rho, vector<double> in_k, vector<double> in_c_p) :
		  rho(in_rho), k(in_k), c_p(in_c_p)
{

};

double HeatTransfer::get_alpha(unsigned int m_id, double T_local)
{
	double alpha = k[m_id]/(c_p[m_id]*rho[m_id]);

	// water ice
	double k_ice  = 0.4685 + 488.12/T_local;
	double cp_ice = 185 + 7.037*T_local;
	double L_ice  = 3.34e5;

	// water
	double k_water  = 0.56;
	double cp_water = 4200.0;
	double L_water  = 0.0;

	// hydrate
	double k_hydrate  = 0.64;
	double cp_hydrate = 494 + 6.1*T_local;
	double L_hydrate  = 0.0;

	// salt (hydrohalite)
	double k_hydrohalite  = 0.6;
	double cp_hydrohalite = 920;
	double L_hydrohalite  = 0.0;

	// if between liquidus and solidus
	double T_min = 245.0;
	double T_max = 273.0;

	//	  if ((T_local < T_max) & (T_local > T_min))
	//	  {
	//		  // figure out melt fraction
	//
	//		  // compute diffusivity
	//
	//		  alpha /= 1000.0;
	//	  }


	return alpha;
};




double HeatTransfer::get_kappa(unsigned int m_id, double T_local, double depth)
{
//	double alpha = k[m_id]/(c_p[m_id]*rho[m_id]);

	// water ice
	//double k_ice  = 0.4685 + 488.12/T_local;
//	double cp_ice = 185 + 7.037*T_local;
//	double L_ice  = 3.34e5;

	// water
//	double k_water  = 0.56;
//	double cp_water = 4200.0;
//	double L_water  = 0.0;

	// hydrate
//	double k_hydrate  = 0.64;
//	double cp_hydrate = 494 + 6.1*T_local;
//	double L_hydrate  = 0.0;

	// salt (hydrohalite)
//	double k_hydrohalite  = 0.6;
//	double cp_hydrohalite = 920;
//	double L_hydrohalite  = 0.0;

	// if between liquidus and solidus
	//	  double T_min = 245.0;
	//	  double T_max = 273.0;

	// implement computation of conductivity for a mixture of materials
	//double kappa = k_ice;

	double kappa = k[m_id];

//	if (T_local > 1175+273 && (abs(depth) < 71000 ))
//	{
//		double kappa = kappa * 1000;
//	}

	return kappa;
};

double HeatTransfer::get_cp(unsigned int m_id, double T_local)
{
	//double cp_ice = 185 + 7.037*T_local;

	double cp = c_p[m_id];
	return cp;
};

