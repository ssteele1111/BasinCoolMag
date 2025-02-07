#ifndef CONFIG_IN_
#define CONFIG_IN_


#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <libconfig.h++>

#include "local_math.h"

using namespace std;
using namespace libconfig;

class config_in3D
{
public:
	config_in3D(char*);

	// I/O information
	string config_filename;
    string mesh_filename;
    string kappa_filename;
    unsigned int kappa_type;
	string output_folder;
	string x_file;
	string y_file;
	string z_file;
	string temp_file;
	string eq_temp_file;

	// Body parameters
	double r_mean;
	double period;
	double omegasquared;
	double beta;
	double intercept;

	// solution domain parameters
	double x_min;
	double x_max;
	double y_min;
	double y_max;
	double z_bottom;

	// Rheology parameters
	vector<double> depths_eta;
	vector<double> eta_kinks;
	vector<double> depths_rho;
	vector<double> rho;
	vector<int>    material_id;
	vector<double> G;

	double eta_ceiling;
	double eta_floor;
	double eta_Ea;
	bool   lat_dependence;

	unsigned int sizeof_depths_eta;
	unsigned int sizeof_depths_rho;
	unsigned int sizeof_rho;
	unsigned int sizeof_eta_kinks;
	unsigned int sizeof_material_id;
	unsigned int sizeof_G;

	double pressure_scale;
	double q;
	bool cylindrical;
	bool continue_plastic_iterations;

	// plasticity variables
	bool plasticity_on;
	unsigned int failure_criterion;
	unsigned int max_plastic_iterations;
	double smoothing_radius;

	// viscoelasticity variables
	unsigned int initial_elastic_iterations;
	double elastic_time;
	double viscous_time;
	double initial_disp_target;
	double final_disp_target;
	double current_time_interval;

	//mesh refinement variables
	unsigned int global_refinement;
	unsigned int adaptive_refinement;
	unsigned int small_r_refinement;
	unsigned int crustal_refinement;
	double       crust_refine_region;
	unsigned int surface_refinement;
	unsigned int impact_refinement;
	double       impact_refine_region;

	//Stokes solver variables
	int    stokes_iteration_coefficient;
	double stokes_tolerance_coefficient;

	// Heat conduction solver variables
	unsigned int heat_iteration_coefficient;
	unsigned int max_iters;
	double       heat_tolerance_coefficient;

	// heat parameters
	vector<double> k;
	vector<double> cp;
	vector<double> alpha; // diffusivity (computed from inputs)

	double T_surf; // surface temperature

	//time step variables
	double       present_time;
	double       time_step;
	double       final_time;

	// run type
	unsigned int run_type;

//private:
	void write_config();
};

config_in3D::config_in3D(char* filename)
{
	// This example reads the configuration file 'example.cfg' and displays
	// some of its contents.

	  Config cfg;

	  // Read the file. If there is an error, report it and exit.
	  try
	  {
	    cfg.readFile(filename);
	  }
	  catch(const FileIOException &fioex)
	  {
	    cerr << "I/O error while reading file:" << filename << std::endl;
	  }
	  catch(const ParseException &pex)
	  {
	    cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
	              << " - " << pex.getError() << std::endl;
	  }

	  //********* IO ****************
	  cout << "******** I/O parameters **********" << endl;
	  // get mesh name

	  try
	  {
		string msh = cfg.lookup("mesh_filename");
	    mesh_filename = msh;
	    cout << "mesh file: " << mesh_filename << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "No 'mesh_filename' setting in configuration file." << endl;
	  }

	  // get mesh name

	//   try
	//   {
	// 	string kappa = cfg.lookup("kappa_file");
	// 	kappa_filename = kappa;
	// 	cout << "kappa file: " << kappa_filename << endl;
	// 	kappa_type = 1; // load z-kappa data from file

	//   }
	//   catch(const SettingNotFoundException &nfex)
	//   {
	// 	  kappa_type = 0; // use material id to get kappa
	// 	  cerr << "No 'kappa_filename' setting in configuration file." << endl;
	//   }

	  // get output folder
	  try
	  {
		  string output = cfg.lookup("output_folder");
		  output_folder = output;
		  cout << "output folder: " << output << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "No 'output_folder' setting in configuration file." << endl;
	  }

	  // get x file
	  try
	  {
		  string x_file_string = cfg.lookup("x_file");
		  x_file = x_file_string;
		  cout << "x file: " << x_file_string << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "No 'x_file' setting in configuration file." << endl;
	  }

		// get y file
	  try
	  {
		  string y_file_string = cfg.lookup("y_file");
		  y_file = y_file_string;
		  cout << "y file: " << y_file_string << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "No 'y_file' setting in configuration file." << endl;
	  }

	  // get z file
	  try
	  {
		  string z_file_string = cfg.lookup("z_file");
		  z_file = z_file_string;
		  cout << "z file: " << z_file_string << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "No 'z_file' setting in configuration file." << endl;
	  }

	  // get initial temperature field file
	  try
	  {
		  string temp_file_string = cfg.lookup("temp_file");
		  temp_file = temp_file_string;

		  cout << "temp file: " << temp_file_string << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "No 'temp_file' setting in configuration file." << endl;
	  }

      // get equilibrium temperature field file
	  try
	  {
		  string eq_temp_file_string = cfg.lookup("eq_temp_file");
		  eq_temp_file = eq_temp_file_string;

		  cout << "eq_temp file: " << eq_temp_file_string << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "No 'eq_temp_file' setting in configuration file." << endl;
	  }


	  // ROOT
	  const Setting& root = cfg.getRoot();

	  // Mesh refinement parameters
	  try
	  {
		  const Setting& mesh_refinement_parameters = root["mesh_refinement_parameters"];
		  mesh_refinement_parameters.lookupValue("global_refinement", global_refinement);
		  mesh_refinement_parameters.lookupValue("adaptive_refinement", adaptive_refinement);
		  mesh_refinement_parameters.lookupValue("crustal_refinement", crustal_refinement);
		  mesh_refinement_parameters.lookupValue("crust_refine_region", crust_refine_region);
		  mesh_refinement_parameters.lookupValue("surface_refinement", surface_refinement);
		  mesh_refinement_parameters.lookupValue("impact_refinement", impact_refinement);
		  mesh_refinement_parameters.lookupValue("impact_refine_region", impact_refine_region);

		  mesh_refinement_parameters.lookupValue("x_min",   x_min);
		  mesh_refinement_parameters.lookupValue("x_max",   x_max);
		  mesh_refinement_parameters.lookupValue("y_min",   y_min);
		  mesh_refinement_parameters.lookupValue("y_max",   y_max);
		  mesh_refinement_parameters.lookupValue("z_bottom", z_bottom);

		  cout << "******** Mesh refinement parameters **********" << endl;
		  cout << "global refinement: " << global_refinement << endl;
		  cout << "crustal_refinement: " << crustal_refinement << endl;
		  cout << "crust_refine_region: " << crust_refine_region << endl;
		  cout << "surface_refinement: " << surface_refinement << endl;
		  cout << "impact_refinement: " << impact_refinement << endl;
		  cout << "impact_refine_region: " << impact_refine_region << endl;

		  cout << "x min: " << x_min << endl;
		  cout << "x max: " << x_max << endl;
		  cout << "y min: " << y_min << endl;
		  cout << "y max: " << y_max << endl;
		  cout << "z bottom: " << z_bottom << endl;

	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the mesh refinement parameters block" << endl;
	  }

	  // Rheology parameters
	  try
	  {
		  cout << "******** Rheology parameters **********" << endl;
		  // get depths_eta ---------------------
		  
		  const Setting& set_depths_eta = cfg.lookup("rheology_parameters.depths_eta");
		  
		  unsigned int ndepths_eta = set_depths_eta.getLength();
		  sizeof_depths_eta = ndepths_eta;

		  for(unsigned int i=0; i<ndepths_eta; i++)
		  {
			  depths_eta.push_back(set_depths_eta[i]);
			  cout << "depth_eta[" << i << "] = " << depths_eta[i] << endl;
		  }

		  // get eta_kinks -------------------------
		  const Setting& set_eta_kinks = cfg.lookup("rheology_parameters.eta_kinks");

		  unsigned int neta_kinks = set_eta_kinks.getLength();
		  sizeof_eta_kinks = neta_kinks;

		  for(unsigned int i=0; i<neta_kinks; i++)
		  {
			  eta_kinks.push_back(set_eta_kinks[i]);
			  cout << "eta_kinks[" << i << "] = " << eta_kinks[i] << endl;
		  }

		  // get depths_rho -------------------------
		  const Setting& set_depths_rho = cfg.lookup("rheology_parameters.depths_rho");

		  unsigned int ndepths_rho = set_depths_rho.getLength();
		  sizeof_depths_rho = ndepths_rho;

		  for(unsigned int i=0; i<ndepths_rho; i++)
		  {
			  depths_rho.push_back(set_depths_rho[i]);
			  cout << "depths_rho[" << i << "] = " << depths_rho[i] << endl;
		  }

		  // get rho -------------------------
		  const Setting& set_rho = cfg.lookup("rheology_parameters.rho");

		  unsigned int nrho = set_rho.getLength();
		  sizeof_rho = nrho;

		  for(unsigned int i=0; i<nrho; i++)
		  {
			  rho.push_back(set_rho[i]);
			  cout << "rho[" << i << "] = " << rho[i] << endl;
		  }

		  // get material_id -------------------------
		  const Setting& set_material_id = cfg.lookup("rheology_parameters.material_id");

		  unsigned int nmaterial_id = set_material_id.getLength();
		  sizeof_material_id = nmaterial_id;

		  for(unsigned int i=0; i<nmaterial_id; i++)
		  {
			  material_id.push_back(set_material_id[i]);
			  cout << "material_id[" << i << "] = " << material_id[i] << endl;
		  }

		  // get G -------------------------
		  const Setting& set_G = cfg.lookup("rheology_parameters.G");

		  unsigned int nG = set_G.getLength();
		  sizeof_G = nG;

		  for(unsigned int i=0; i<nG; i++)
		  {
			  G.push_back(set_G[i]);
			  cout << "G[" << i << "] = " << G[i] << endl;
		  }

		  const Setting& rheology_parameters = root["rheology_parameters"];
		  rheology_parameters.lookupValue("eta_ceiling", eta_ceiling);
		  rheology_parameters.lookupValue("eta_floor", eta_floor);
		  rheology_parameters.lookupValue("eta_Ea", eta_Ea);
		  rheology_parameters.lookupValue("lat_dependence", lat_dependence);
		  rheology_parameters.lookupValue("pressure_scale", pressure_scale);
		  rheology_parameters.lookupValue("q", q);
		  rheology_parameters.lookupValue("cylindrical", cylindrical);
		  rheology_parameters.lookupValue("continue_plastic_iterations", continue_plastic_iterations);

		  cout << "eta ceiling: " << eta_ceiling << endl;
		  cout << "eta floor: " << eta_floor << endl;
		  cout << "eta Ea: " << eta_Ea << endl;
		  cout << "lat dependence: " << lat_dependence << endl;
		  cout << "pressure scale: " << pressure_scale << endl;
		  cout << "q: " << q << endl;
		  cout << "cylindrical: " << cylindrical << endl;
		  cout << "continue_plastic_iterations: " << continue_plastic_iterations << endl;
	  }

	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the rheology parameters block" << endl;
	  }

	  // Plasticity parameters
	  try
	  {
		  const Setting& plasticity_parameters = root["plasticity_parameters"];
		  plasticity_parameters.lookupValue("plasticity_on", plasticity_on);
		  plasticity_parameters.lookupValue("failure_criterion", failure_criterion);
		  plasticity_parameters.lookupValue("max_plastic_iterations", max_plastic_iterations);
		  plasticity_parameters.lookupValue("smoothing_radius", smoothing_radius);

		  cout << "******** Plasticity parameters **********" << endl;
		  cout << "plasticity on?: " << plasticity_on << endl;
		  cout << "failure_criterion: " << failure_criterion << endl;
		  cout << "max plastic iterations: " << max_plastic_iterations << endl;
		  cout << "smoothing_radius: " << smoothing_radius << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the plasticity parameters block" << endl;
	  }

	  // Viscoelasticity parameters
	  try
	  {

		  const Setting& viscoelasticity_parameters = root["viscoelasticity_parameters"];
		  viscoelasticity_parameters.lookupValue("initial_elastic_iterations", initial_elastic_iterations);
		  viscoelasticity_parameters.lookupValue("elastic_time", elastic_time);
		  viscoelasticity_parameters.lookupValue("viscous_time", viscous_time);
		  viscoelasticity_parameters.lookupValue("initial_disp_target", initial_disp_target);
		  viscoelasticity_parameters.lookupValue("final_disp_target", final_disp_target);
		  viscoelasticity_parameters.lookupValue("current_time_interval", current_time_interval);

		  viscous_time *= SECSINYEAR;

		  cout << "******** Viscoelasticity parameters **********" << endl;
		  cout << "initial elastic iterations? " << initial_elastic_iterations << endl;
		  cout << "elastic_time: " << elastic_time << endl;
		  cout << "viscous_time: " << viscous_time << " (converted to seconds)" << endl;
		  cout << "initial_disp_target: " << initial_disp_target << endl;
		  cout << "initial final_disp_target: " << final_disp_target << endl;
		  cout << "current_time_interval: " << initial_elastic_iterations << endl;
		  cout << "initial elastic iterations?: " << current_time_interval << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the viscoelasticity parameters block" << endl;
	  }

	  // Heat conduction parameters
	  try
	  {
		  const Setting& heat_parameters = root["heat_parameters"];
		  cout << "******** Heat conduction parameters **********" << endl;


		  // get k
		  const Setting& set_k = cfg.lookup("heat_parameters.k");
		  unsigned int numel_k = set_k.getLength();
		  
		  cout << "total number of k = " << numel_k << endl;
		  
		  for(unsigned int i=0; i<numel_k; i++)
		  {
			  k.push_back(set_k[i]);
			  cout << "k[" << i << "] = " << k[i] << endl;
		  }

		  // get cp
		  const Setting& set_cp = cfg.lookup("heat_parameters.cp");
		  unsigned int numel_cp = set_cp.getLength();
		  
		  cout << "total number of cp  = " << numel_cp << endl;
		  
		  for(unsigned int i=0; i<numel_cp; i++)
		  {
			  cp.push_back(set_cp[i]);
			  cout << "cp[" << i << "] = " << cp[i] << endl;
		  }
		  // set diffusivity
		  for(unsigned int i=0; i<numel_k; i++)
		  {
			  alpha.push_back(k[i]/(cp[i]*rho[i]));
			  cout << "alpha[" << i << "] = " << alpha[i] << endl;
		  }

		  // get T_surf
		  heat_parameters.lookupValue("T_surf", T_surf);
		  cout << "T_surf = " << T_surf << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the heat parameters block" << endl;
	  }

	  // Stokes flow solver parameters
	  try
	  {
		  const Setting& solve_parameters = root["stokes_solve_parameters"];
		  solve_parameters.lookupValue("iteration_coefficient", stokes_iteration_coefficient);
		  solve_parameters.lookupValue("tolerance_coefficient", stokes_tolerance_coefficient);

		  cout << "******** Stokes flow solver parameters **********" << endl;
		  cout << "Stokes flow iteration coefficient = " << stokes_iteration_coefficient << endl;
		  cout << "Stokes flow tolerance coefficient = " << stokes_tolerance_coefficient << endl;

	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the Stokes flow solver parameters block" << endl;
	  }

	  // Heat conduction solver parameters
	  try
	  {
		  const Setting& solve_parameters = root["heat_solve_parameters"];
		  solve_parameters.lookupValue("iteration_coefficient", heat_iteration_coefficient);
		  solve_parameters.lookupValue("max_iters", max_iters);
		  solve_parameters.lookupValue("tolerance_coefficient", heat_tolerance_coefficient);

		  cout << "******** Heat conduction solver parameters **********" << endl;
		  cout << "heat conduction iteration coefficient = " << heat_iteration_coefficient << endl;
		  cout << "max. Newton iterations = " << max_iters << endl;
		  cout << "heat conduction tolerance coefficient = " << heat_tolerance_coefficient << endl;
	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the heat conduction solver parameters block" << endl;
	  }

	  // Time step parameters
	  try
	  {
	    const Setting& time_step_parameters = root["time_step_parameters"];
	    time_step_parameters.lookupValue("present_time", present_time);
	    time_step_parameters.lookupValue("time_step", time_step);
	    time_step_parameters.lookupValue("final_time", final_time);

	    present_time *= SECSINYEAR;
	    time_step    *= SECSINYEAR;
	    final_time   *= SECSINYEAR;

	    cout << "******** time step parameters **********" << endl;
	    cout << "present_time = " << present_time << " (converted to secs)" << endl;
	    cout << "time_step = " << time_step << " (converted to secs)" << endl;
	    cout << "final_time = " << final_time << endl;

	  }
	  catch(const SettingNotFoundException &nfex)
	  {
		  cerr << "We've got a problem in the time step parameters block" << endl;
	  }

	  // Run type
	  const Setting& run_parameters = root["run_parameters"];
	  run_parameters.lookupValue("run_type", run_type);
	  cout << "run_type = " << run_type << endl;
}

void config_in3D::write_config()
{
//		std::ostringstream config_parameters;
//		config_parameters << output_folder << "/run_parameters.txt";
//		std::ofstream fout_config(config_parameters.str().c_str());
//
//		// mesh filename
//		fout_config << "mesh filename: " << mesh_filename << endl;
	cout << "mesh filename: " << mesh_filename << endl;
}



#endif
