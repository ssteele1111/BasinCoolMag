
 
//  Author: Sarah Steele, Harvard 2024 (based on code written by Anton Ermakov (Stanford), which was in turn based on a deal.ii tutorial 
//  code written by Wolfgang Bangerth (Texas A&M University))



#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/grid/tria_iterator.h>

#include <cstring>
#include <exception>
#include <iostream>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <cmath>

#include "support_code/config_in.cc"
#include "support_code/timer.h"

#include "heat_transfer.cc"
#include "rheology.cc"
#include "heat_right_hand_side.cc"
#include "initial_temperature.cc"
#include "heat_boundary_values.cc"

#include <armadillo>

// import deal.II + arma namespaces

using namespace dealii;
using namespace arma;
namespace fs = std::filesystem;


template <int dim>
class HeatEquation
{
public:
	HeatEquation(config_in&);
	void clear_output_directory();
	void run_newton();
	void run_simple();
	config_in& cfg;

private:
	void load_initial_temperature();
	void heat_setup_system();
	bool check_equil();
	double get_alpha();
	void set_system_BCs();
	void set_boundary_indicators();
	void postrefine_setup();
	void postunrefine_setup();
	void heat_compute_mass_and_laplace_matrices();
	void heat_compute_mass_and_laplace_matrices_simple();
	void heat_setup_crank_nicolson_nonlinear();
	void heat_setup_crank_nicolson_linear();
	void heat_setup_crank_nicolson_simple();
	void heat_solve_system_nonlinear();
	void heat_solve_system_linear();
	void heat_solve_system_simple();
	double compute_residual(const double alpha) const;
	void do_newton_iter();
	void do_linear();
	void do_heat_step_simple();
	void refine_mesh(const unsigned int min_grid_level,
			 const unsigned int max_grid_level);
	void refine_mesh_simple(const unsigned int min_grid_level,
			 const unsigned int max_grid_level);
	void unrefine_mesh();
	void graphical_output_results(unsigned int newton_iter) const;
	void graphical_output_results_simple() const;
	void textual_output_results() const;
	void textual_output_results_blocks() const;
	void set_initial_temperature();

	Triangulation<dim> triangulation;
	Triangulation<dim> triangulation_preref;
	FE_Q<dim>          fe;
	DoFHandler<dim>    dof_handler;
	DoFHandler<dim>    dof_handler_preref;

	AffineConstraints<double> constraints;

	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> heat_mass_matrix;
	SparseMatrix<double> heat_bc_matrix;
	SparseMatrix<double> heat_bc_matrix_2;
	SparseMatrix<double> heat_laplace_matrix;
	SparseMatrix<double> heat_system_matrix;

	Vector<double> heat_tmp;
	Vector<double> heat_MU_tmp;
	Vector<double> heat_LU_tmp;
	Vector<double> heat_bc_tmp;
	Vector<double> heat_bc_tmp_2;
	Vector<double> heat_forcing_terms;

	Vector<double> heat_solution;
	Vector<double> old_heat_solution;
	Vector<double> kappa_vals;
	Vector<double> heat_update;
	Vector<double> heat_system_rhs;
	Vector<double> heat_bc_rhs;

	HeatBoundaryValuesRight<dim> heat_boundary_values_function_right;
	HeatBoundaryValuesBottom<dim> heat_boundary_values_function_bottom;
	HeatBoundaryValuesTop<dim> heat_boundary_values_function_top;
	HeatBoundaryValuesTop<dim> heat_boundary_values_function_top2;

	double       time;
	double       time_step;
	unsigned int timestep_number;
	double theta;
	double alpha;

	vec x_vec;
	vec z_vec;
	mat kappa_mat;
	Kappa<dim> k_init;
	mat initial_temperature_mat;
	mat eq_temperature_mat;
	double		 T_eq;
	double		 T_bottom;
	const Point<dim> bottom_corner;
};


// Implementation of the main class
template <int dim>
HeatEquation<dim>::HeatEquation(config_in& cfgi)
: fe(1)
  , dof_handler(triangulation)
  , dof_handler_preref(triangulation_preref)
  , cfg(cfgi)
  , time(cfgi.present_time)
  , time_step(cfgi.time_step)
  , timestep_number(0)
//  , alpha(0.05)
  {};


template <int dim>
void HeatEquation<dim>::clear_output_directory()
{
	const fs::path& dir_path = cfg.output_folder;
	for (auto& path: fs::directory_iterator(dir_path))
	{
		fs::remove_all(path);
	}
}

template <int dim>
void HeatEquation<dim>::load_initial_temperature()
{
	x_vec.load(cfg.x_file, raw_ascii);
	z_vec.load(cfg.z_file, raw_ascii);
	initial_temperature_mat.load(cfg.temp_file, raw_ascii);
	eq_temperature_mat.load(cfg.eq_temp_file, raw_ascii);

	std::cout << "Finite>?" << initial_temperature_mat.has_nan() << std::endl;
	if (cfg.kappa_type == 1) {
		kappa_mat.load(cfg.kappa_filename, raw_ascii);
	}

	T_eq = cfg.T_surf;
}

template <int dim>
void HeatEquation<dim>::set_boundary_indicators()
{
	double zero_tolerance = 1.0e-4;
	double distance_left;
	double distance_right;
	double distance_bottom;
	double distance_top;

	for (const auto &cell : dof_handler.active_cell_iterators()) // loop over all cells
	{
		// loop over faces to set boundary ids
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) // loop over all faces
		{
			if (cell->face(f)->at_boundary())
			{
				// find distance to closest flat boundary
				distance_left   = fabs(cell->face(f)->center()[0] - cfg.x_left);
				distance_right  = fabs(cell->face(f)->center()[0] - cfg.x_right);
				distance_bottom = fabs(cell->face(f)->center()[1] - cfg.z_bottom);
				distance_top = fabs(cell->face(f)->center()[1]);

				// ID = 0 : left boundary
				if (distance_left < zero_tolerance)
				{
					cell->face(f)->set_all_boundary_ids(0);

				}
				// ID = 1 : bottom boundary
				else if (distance_bottom <  zero_tolerance)
				{
					cell->face(f)->set_all_boundary_ids(1);
				}
				// ID = 2 : right boundary
				else if (distance_right <  zero_tolerance)
				{
					cell->face(f)->set_all_boundary_ids(2);
				}
				// ID = 3 : top boundary
				else if (distance_top <  zero_tolerance)
				{
					cell->face(f)->set_all_boundary_ids(3);
				}
			}
		}
	}
};

template <int dim>
void HeatEquation<dim>::set_system_BCs()
{
	std::map<types::global_dof_index, double> heat_boundary_values_right;
	std::map<types::global_dof_index, double> heat_boundary_values_bottom;
	std::map<types::global_dof_index, double> heat_boundary_values_top;

	VectorTools::interpolate_boundary_values(dof_handler,
			1,
			heat_boundary_values_function_bottom,
			heat_boundary_values_bottom);

	VectorTools::interpolate_boundary_values(dof_handler,
			2,
			heat_boundary_values_function_right,
			heat_boundary_values_right);
			
	// VectorTools::interpolate_boundary_values(dof_handler,
			// 5,
			// heat_boundary_values_function_top,
			// heat_boundary_values_top);

	MatrixTools::apply_boundary_values(heat_boundary_values_bottom,
			heat_system_matrix,
			heat_solution,
			heat_system_rhs);

	MatrixTools::apply_boundary_values(heat_boundary_values_right,
			heat_system_matrix,
			heat_solution,
			heat_system_rhs);

	// MatrixTools::apply_boundary_values(heat_boundary_values_top,
	// 		heat_system_matrix,
	// 		heat_solution,
	// 		heat_system_rhs);
}


template <int dim>
bool HeatEquation<dim>::check_equil()
{
//	const QGauss<dim-1> top_quadrature_formula(fe.degree + 1);
	const QTrapezoid<dim-1> top_quadrature_formula;
	double T_avg = 0;
	double n = 0;

	// make FEFaceValues (for top boundary condition)
	FEFaceValues<dim> fe_top_values(fe,
				top_quadrature_formula,
				update_values |
				update_quadrature_points | update_JxW_values);

	const unsigned int n_top_q_points	= top_quadrature_formula.size();
	double T_local;
	bool is_equil = false;

	std::vector<double> temperature_at_top_q_points(n_top_q_points);

	for (const auto &cell : dof_handler.active_cell_iterators())
	{

		// compute term from radiative boundary condition
		// iterate over faces
		for (unsigned int f = 0; f<GeometryInfo<dim>::faces_per_cell; ++f)
		{

			// check if face is at top boundary
			if (cell->face(f)->at_boundary() && (cell->face(f)->boundary_id() == 3))
			{
				fe_top_values.reinit(cell, f);

				// get values of heat_solution at top quadrature points
				fe_top_values.get_function_values(heat_solution,temperature_at_top_q_points);

				for (unsigned int q_index = 0; q_index < n_top_q_points; ++q_index)
				{
					// get local temperature
					T_local     = temperature_at_top_q_points[q_index];

					T_avg += (T_local);
					++n;

				}
			}
		}
	}

	std::cout << "Avg diff: " << (abs(T_avg/n -T_eq)) << std::endl;

	if ((abs((T_avg/n) -T_eq))<1) {
		is_equil = true;
		std::cout << "Surface equilibrated!" << std::endl;
	}
	return is_equil;
}


template <int dim>
double HeatEquation<dim>::get_alpha()
{
	return 0.2;
}


template <int dim>
void HeatEquation<dim>::heat_setup_system()
{
	dof_handler.distribute_dofs(fe);

	std::cout << std::endl
			<< "===========================================" << std::endl
			<< "Number of active cells: " << triangulation.n_active_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< std::endl
			<< std::endl;

	// set up constraints
	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);

	constraints.close();

	std::cout << "set up constraints" << std::endl;

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			constraints,
			/*keep_constrained_dofs = */ true);
	sparsity_pattern.copy_from(dsp);

	heat_mass_matrix.reinit(sparsity_pattern);
	heat_laplace_matrix.reinit(sparsity_pattern);
	heat_bc_matrix.reinit(sparsity_pattern);
	heat_bc_matrix_2.reinit(sparsity_pattern);
	heat_system_matrix.reinit(sparsity_pattern);

	// initialize solutions
	// not all of these may be used depending on the run type, but that's ok
	heat_solution.reinit(dof_handler.n_dofs());
	old_heat_solution.reinit(dof_handler.n_dofs());
	kappa_vals.reinit(dof_handler.n_dofs());
	heat_update.reinit(dof_handler.n_dofs());
	heat_bc_rhs.reinit(dof_handler.n_dofs());
	heat_system_rhs.reinit(dof_handler.n_dofs());

	if (cfg.run_type == 1){
		heat_system_rhs = 0;
		heat_system_matrix = 0;
	}

	std::cout << "initialized vectors" << std::endl;

	// if initial step, set initial conditions
	if (timestep_number == 0)
		set_initial_temperature();
		std::cout << "set initial temperature" << std::endl;
}

/* set up after adaptive refinement step */
template <int dim>
void HeatEquation<dim>::postrefine_setup()
{
	dof_handler.distribute_dofs(fe);

	std::cout << std::endl
			<< "===========================================" << std::endl
			<< "Number of active cells: " << triangulation.n_active_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< std::endl
			<< std::endl;

	// set up constraints
	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);

	constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			constraints,
			/*keep_constrained_dofs = */ true); // make_sparsity_pattern or make_flux_sparsity
	sparsity_pattern.copy_from(dsp);


	// intialize matrices
	heat_mass_matrix.reinit(sparsity_pattern);
	heat_laplace_matrix.reinit(sparsity_pattern);
	heat_bc_matrix.reinit(sparsity_pattern);
	heat_bc_matrix_2.reinit(sparsity_pattern);
	heat_system_matrix.reinit(sparsity_pattern);
	heat_bc_rhs.reinit(dof_handler.n_dofs());

	// initialize solutions
	heat_update.reinit(dof_handler.n_dofs());
	heat_system_rhs.reinit(dof_handler.n_dofs());
	heat_solution.reinit(dof_handler.n_dofs());
	old_heat_solution.reinit(dof_handler.n_dofs());
	kappa_vals.reinit(dof_handler.n_dofs());

	if (cfg.run_type == 1){
		heat_system_rhs = 0;
		heat_system_matrix = 0;
	}
}

/* set up after adaptive refinement step */
template <int dim>
void HeatEquation<dim>::postunrefine_setup()
{
	dof_handler.distribute_dofs(fe);

	std::cout << std::endl
			<< "===========================================" << std::endl
			<< "Number of active cells: " << triangulation.n_active_cells()
			<< std::endl
			<< "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< std::endl
			<< std::endl;

	// set up constraints
	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);

	constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			constraints,
			/*keep_constrained_dofs = */ true); // make_sparsity_pattern or make_flux_sparsity
	sparsity_pattern.copy_from(dsp);


	// intialize matrices
	heat_mass_matrix.reinit(sparsity_pattern);
	heat_laplace_matrix.reinit(sparsity_pattern);
	heat_bc_matrix.reinit(sparsity_pattern);
	heat_bc_matrix_2.reinit(sparsity_pattern);
	heat_system_matrix.reinit(sparsity_pattern);
	heat_bc_rhs.reinit(dof_handler.n_dofs());

	// initialize solutions
	heat_update.reinit(dof_handler.n_dofs());
	heat_system_rhs.reinit(dof_handler.n_dofs());
	heat_solution.reinit(dof_handler.n_dofs());
	old_heat_solution.reinit(dof_handler.n_dofs());
	kappa_vals.reinit(dof_handler.n_dofs());

	heat_system_rhs = 0;
	heat_system_matrix = 0;
}

template <int dim>
void HeatEquation<dim>::heat_compute_mass_and_laplace_matrices()
{
	std::cout << "Kappa type: " << cfg.kappa_type << std::endl;

	if (cfg.kappa_type == 1) {

			VectorTools::interpolate(dof_handler,
								k_init,
								kappa_vals);
		}

	// make new mass, laplace, and BC matrices that are the right shape and filled with zeros
	heat_mass_matrix.reinit(sparsity_pattern);
	heat_laplace_matrix.reinit(sparsity_pattern);
	heat_bc_matrix.reinit(sparsity_pattern);
	heat_bc_matrix_2.reinit(sparsity_pattern);

	// initialize rhs
	heat_bc_rhs.reinit(dof_handler.n_dofs());

	heat_system_rhs = 0;
	heat_system_matrix = 0;

	// set heat transfer
	HeatTransfer heat_transfer(cfg.rho, cfg.k, cfg.cp);

	double r_value;
	double z_value;
	double kappa_local;
	double rho_cp;
	double T_local;
	double old_T_local;
	bool is_singular;
	unsigned int m_id;
	const double sb_constant 			= 5.67037*pow(10.,-8.);

	const QGauss<dim> quadrature_formula(fe.degree + 1);
	const QTrapezoid<dim-1> top_quadrature_formula;

	FEValues<dim> fe_values(fe,
			quadrature_formula,
			update_values | update_gradients |
			update_quadrature_points | update_JxW_values);

	// make FEFaceValues (for top boundary condition)
	FEFaceValues<dim> fe_top_values(fe,
				top_quadrature_formula,
				update_values |
				update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points    = quadrature_formula.size();
	const unsigned int n_top_q_points	= top_quadrature_formula.size();


	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	FullMatrix<double> cell_matrix_2(dofs_per_cell, dofs_per_cell);
	Vector<double>     cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	std::vector<double> temperature_at_q_points(n_q_points);
	std::vector<double> kappa_at_q_points(n_q_points);
	std::vector<double> temperature_at_top_q_points(n_top_q_points);
	std::vector<double> old_temperature_at_top_q_points(n_top_q_points);

	/*========== LAPLACE MATRIX ==========
	 * Create Laplace matrix manually.
	 * ===================
	 */

	for (const auto &cell : dof_handler.active_cell_iterators())
		{
		cell_matrix = 0;
		fe_values.reinit(cell);
		fe_values.get_function_values(heat_solution,temperature_at_q_points);
		if (cfg.kappa_type == 1) {
			fe_values.get_function_values(kappa_vals,kappa_at_q_points);
		}


		for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
		{
			r_value = fe_values.quadrature_point(q_index)[0];
			z_value = fe_values.quadrature_point(q_index)[1];
			int m_id_abs = cell->material_id();

			// get index of material id
			vector<int>::iterator itp;
			itp = std::find(cfg.material_id.begin(),cfg.material_id.end(),m_id_abs);
			m_id = std::distance(cfg.material_id.begin(),itp);

			// get conductivity
			T_local     = temperature_at_q_points[q_index]; 

			if (cfg.kappa_type == 1) {
				if (m_id == 0) {
					kappa_local = heat_transfer.get_kappa(m_id,T_local,z_value);

				} else{
					kappa_local = kappa_at_q_points[q_index];
				}
			} else {
				kappa_local = heat_transfer.get_kappa(m_id,T_local,z_value);
			}

			const double current_coefficient = kappa_local * 2*PI*r_value;

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) +=
							(current_coefficient *              // a(x_q)
									fe_values.shape_grad(i, q_index) *      //  phi_i(x_q)
									fe_values.shape_grad(j, q_index) *      //  phi_j(x_q)
									fe_values.JxW(q_index));           // dx
			}
		}

		// Finally, transfer the contributions from cell_matrix and cell_rhs into the global objects.
		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i < dofs_per_cell; ++i)

			for (unsigned int j = 0; j < dofs_per_cell; ++j)

				heat_laplace_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));


		
	}

	/*========== MASS MATRIX ==========
	 * Create mass matrix manually.
	 * ===================
	 */
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		cell_matrix = 0;
		fe_values.reinit(cell);
		fe_values.get_function_values(heat_solution,temperature_at_q_points);

		for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
		{
			r_value = fe_values.quadrature_point(q_index)[0];
			int m_id_abs = cell->material_id();

			// get index of material id
			vector<int>::iterator itp;
			itp = std::find(cfg.material_id.begin(),cfg.material_id.end(),m_id_abs);
			m_id = std::distance(cfg.material_id.begin(),itp);

			// get product of density and heat capacity
			T_local     = temperature_at_q_points[q_index];

			rho_cp      = cfg.rho[m_id] * heat_transfer.get_cp(m_id,T_local);

			const double current_coefficient = rho_cp*2*PI*r_value;

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) +=
							(current_coefficient *               //  a(x_q)
									fe_values.shape_value(i, q_index) * //  phi_i(x_q)
									fe_values.shape_value(j, q_index) * //  phi_j(x_q)
									fe_values.JxW(q_index));            //  dx
			}
		}

		// transfer the contributions from cell_matrix and cell_rhs into the global objects.
		cell->get_dof_indices(local_dof_indices);

		for (unsigned int i = 0; i < dofs_per_cell; ++i)

			for (unsigned int j = 0; j < dofs_per_cell; ++j)

				heat_mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));


	}

	/*========== BC MATRIX ==========
	 * Create B matrix - this creates a boundary condition matrix manually. This is only performed separately from
	 * computation of the mass matrix since it is sensitive to the chosen theta and time step values.
	 * ===================
	 */
	double max_T_local = 0.;
	double min_T_local = 1000.;
	double max_T_diff = 0.;

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		cell_matrix = 0;
		cell_matrix_2 = 0;
		cell_rhs = 0;
		fe_values.reinit(cell);
		fe_values.get_function_values(heat_solution,temperature_at_q_points);

		// compute term from radiative boundary condition
		// iterate over faces

		for (unsigned int f = 0; f<GeometryInfo<dim>::faces_per_cell; ++f)
		{

			// check if face is at top boundary
			if (cell->face(f)->at_boundary() && (cell->face(f)->boundary_id() == 3))
			{
				fe_top_values.reinit(cell, f);

				// get values of heat_solution at top quadrature points
				fe_top_values.get_function_values(heat_solution,temperature_at_top_q_points);
				fe_top_values.get_function_values(old_heat_solution,old_temperature_at_top_q_points);

				for (unsigned int q_index = 0; q_index < n_top_q_points; ++q_index)
				{
					// get radial location
					r_value = fe_top_values.quadrature_point(q_index)[0];



					const double current_coefficient = 2*PI*r_value *sb_constant;

					for (unsigned int i = 0; i < dofs_per_cell; ++i)
					{
						T_local     = temperature_at_top_q_points[q_index];
						old_T_local     = old_temperature_at_top_q_points[q_index];


						if (T_local > max_T_local) {
							max_T_local = T_local;
						}
						if (T_local < min_T_local) {
							min_T_local = T_local;
						}
						if (abs(T_local-old_T_local) > max_T_diff) {
							max_T_diff = abs(T_local - old_T_local);
						}


						if (abs(T_eq - T_local) > 0)//cfg.heat_tolerance_coefficient * time_step) // the bones for a future more sophisticated implementation
						{
							cell_rhs(i) += (current_coefficient *			// sigma
									pow(T_eq,4) *							//T^{n-1}^4
									fe_top_values.shape_value(i,q_index) *	//
									fe_top_values.JxW(q_index));


							for (unsigned int j = 0; j < dofs_per_cell; ++j) {
								// normal B matrix
								cell_matrix(i, j) +=
										(current_coefficient *         //  a(x_q)
												pow(T_local,3) *
												T_local/abs(T_local) *
												fe_top_values.shape_value(i, q_index) * //  phi_i(x_q)
												fe_top_values.shape_value(j, q_index) * //  phi_j(x_q)
												fe_top_values.JxW(q_index));            //  dx

							    //
								cell_matrix_2(i, j) +=
										(current_coefficient *         //  a(x_q)
												pow(old_T_local,3) *
												fe_top_values.shape_value(i, q_index) * //  phi_i(x_q)
												fe_top_values.shape_value(j, q_index) * //  phi_j(x_q)
												fe_top_values.JxW(q_index));            //  dx
							}

						} else {

								cell_rhs(i) += (current_coefficient *			// sigma
										(pow(T_eq,4)-pow(T_local,4)) *							//T^{n-1}^4
										fe_top_values.shape_value(i,q_index) *	//
										fe_top_values.JxW(q_index));

								for (unsigned int j = 0; j < dofs_per_cell; ++j)
									cell_matrix(i, j) +=
											(current_coefficient *         //  a(x_q)
													fe_top_values.shape_value(i, q_index) * //  phi_i(x_q)
													fe_top_values.shape_value(j, q_index) * //  phi_j(x_q)
													fe_top_values.JxW(q_index));            //  dx
							}

						}
					}
				}
			}

		// transfer the contributions from cell_matrix and cell_rhs into the global objects.
		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j < dofs_per_cell; ++j) {
				heat_bc_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
				heat_bc_matrix_2.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_2(i, j));
			}

			heat_bc_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}
}

template <int dim>
void HeatEquation<dim>::heat_compute_mass_and_laplace_matrices_simple()
{
	// make new mass, laplace, and BC matrices that are the right shape and filled with zeros
	heat_mass_matrix.reinit(sparsity_pattern);
	heat_laplace_matrix.reinit(sparsity_pattern);
	heat_bc_matrix.reinit(sparsity_pattern);

	// initialize rhs
	heat_bc_rhs.reinit(dof_handler.n_dofs());

	// set heat transfer
	HeatTransfer heat_transfer(cfg.rho, cfg.k, cfg.cp);

	double r_value;
	double z_value;
	double kappa_local;
	double rho_cp;
	double T_local;
	bool is_singular;
	unsigned int m_id;
	const double sb_constant 			= 5.67037*pow(10.,-8.);

	const QGauss<dim> quadrature_formula(fe.degree + 1);
	const QGauss<dim-1> top_quadrature_formula(fe.degree + 1);

	FEValues<dim> fe_values(fe,
			quadrature_formula,
			update_values | update_gradients |
			update_quadrature_points | update_JxW_values);

	// make FEFaceValues (for top boundary condition)
	FEFaceValues<dim> fe_top_values(fe,
				top_quadrature_formula,
				update_values |
				update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points    = quadrature_formula.size();
	const unsigned int n_top_q_points	= top_quadrature_formula.size();


	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>     cell_rhs(dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	std::vector<double> temperature_at_q_points(n_q_points);
	std::vector<double> temperature_at_top_q_points(n_top_q_points);

	/*========== LAPLACE MATRIX ==========
	 * Create Laplace matrix manually.
	 * ===================
	 */

	for (const auto &cell : dof_handler.active_cell_iterators())
	{

		cell_matrix = 0;
		fe_values.reinit(cell);
		fe_values.get_function_values(heat_solution,temperature_at_q_points);

		for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
		{
			r_value = fe_values.quadrature_point(q_index)[0];
			z_value = fe_values.quadrature_point(q_index)[1];
			m_id = cell->material_id();

			// get conductivity
			T_local     = temperature_at_q_points[q_index];

			kappa_local = heat_transfer.get_kappa(m_id,T_local,z_value);

			const double current_coefficient = kappa_local * 2*PI*r_value;

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) +=
							(current_coefficient *              // a(x_q)
									fe_values.shape_grad(i, q_index) *      //  phi_i(x_q)
									fe_values.shape_grad(j, q_index) *      //  phi_j(x_q)
									fe_values.JxW(q_index));           // dx
			}
		}

		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i < dofs_per_cell; ++i)

			for (unsigned int j = 0; j < dofs_per_cell; ++j)

				heat_laplace_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));

		
	}

	/*========== MASS MATRIX ==========
	 * Create mass m matrix manually
	 * ===================
	 */
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		cell_matrix = 0;
		fe_values.reinit(cell);
		fe_values.get_function_values(heat_solution,temperature_at_q_points);

		for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
		{
			r_value = fe_values.quadrature_point(q_index)[0];
			m_id = cell->material_id();

			// get product of density and heat capacity
			T_local     = temperature_at_q_points[q_index];

			rho_cp      = cfg.rho[m_id] * heat_transfer.get_cp(m_id,T_local);
			const double current_coefficient = rho_cp*2*PI*r_value;

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) +=
							(current_coefficient *               //  a(x_q)
									fe_values.shape_value(i, q_index) * //  phi_i(x_q)
									fe_values.shape_value(j, q_index) * //  phi_j(x_q)
									fe_values.JxW(q_index));            //  dx
			}
		}

		cell->get_dof_indices(local_dof_indices);

		for (unsigned int i = 0; i < dofs_per_cell; ++i)

			for (unsigned int j = 0; j < dofs_per_cell; ++j)

				heat_mass_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));

	}

	/*========== BC MATRIX ==========
	 * Create B matrix - this creates a boundary condition matrix manually. This is only performed separately from
	 * computation of the mass matrix since it is sensitive to the chosen theta and time step values.
	 * ===================
	 */

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		cell_matrix = 0;
		cell_rhs = 0;
		fe_values.reinit(cell);
		fe_values.get_function_values(heat_solution,temperature_at_q_points);

		// compute term from radiative boundary condition
		// iterate over faces
		for (unsigned int f = 0; f<GeometryInfo<dim>::faces_per_cell; ++f)
		{
			// check if face is at top boundary
			if (cell->face(f)->at_boundary() && (cell->face(f)->boundary_id() == 3))
			{
				fe_top_values.reinit(cell, f);

				// get values of heat_solution at top quadrature points
				fe_top_values.get_function_values(heat_solution,temperature_at_top_q_points);

				for (unsigned int q_index = 0; q_index < n_top_q_points; ++q_index)
				{
					// get radial location
					r_value = fe_top_values.quadrature_point(q_index)[0];

					// get local temperature
					T_local     = temperature_at_top_q_points[q_index];

					const double current_coefficient = 2*PI*r_value *sb_constant;

					for (unsigned int i = 0; i < dofs_per_cell; ++i)
					{
						// not currently in use, just the start of infrastructure to stop explicitly calculating radiative cooling
						// once the surface temp is close enough to equilibrium
						if (abs(T_eq - T_local) >= 0)
						{
							cell_rhs(i) += (current_coefficient *			// sigma
									pow(T_eq,4) *							//T^{n-1}^4
									fe_top_values.shape_value(i,q_index) *	//
									fe_top_values.JxW(q_index));

							for (unsigned int j = 0; j < dofs_per_cell; ++j)
								cell_matrix(i, j) +=
										(current_coefficient * pow(T_local,3) *         //  a(x_q)
												fe_top_values.shape_value(i, q_index) * //  phi_i(x_q)
												fe_top_values.shape_value(j, q_index) * //  phi_j(x_q)
												fe_top_values.JxW(q_index));            //  dx

						} else {

								cell_rhs(i) += (current_coefficient *			// sigma
										pow(T_eq,4) *							//T^{n-1}^4
										fe_top_values.shape_value(i,q_index) *	//
										fe_top_values.JxW(q_index));

								for (unsigned int j = 0; j < dofs_per_cell; ++j)
									cell_matrix(i, j) +=
											(current_coefficient * pow(T_local,3) *         //  a(x_q)
													fe_top_values.shape_value(i, q_index) * //  phi_i(x_q)
													fe_top_values.shape_value(j, q_index) * //  phi_j(x_q)
													fe_top_values.JxW(q_index));            //  dx
							}

						}
					}
				}
			}

		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j < dofs_per_cell; ++j)
				heat_bc_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));

			heat_bc_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}
}

template <int dim>
void HeatEquation<dim>::heat_setup_crank_nicolson_nonlinear()
{
	// initialize temporary vectors
	heat_tmp.reinit(heat_solution.size());
	heat_MU_tmp.reinit(heat_solution.size());
	heat_LU_tmp.reinit(heat_solution.size());
	heat_bc_tmp.reinit(heat_solution.size());
	heat_bc_tmp_2.reinit(heat_solution.size());
	heat_forcing_terms.reinit(heat_solution.size());

	// some assembly
	heat_mass_matrix.vmult(heat_system_rhs, old_heat_solution);		// rho Cp M U_(n-1)
	heat_mass_matrix.vmult(heat_MU_tmp, heat_solution);				// **rho Cp M U_n
	heat_laplace_matrix.vmult(heat_tmp, old_heat_solution);			// ** k L U_(n-1)
	heat_laplace_matrix.vmult(heat_LU_tmp, heat_solution);			// ** k L U_n
	heat_bc_matrix.vmult(heat_bc_tmp, heat_solution);				// ** sigma B U_n^4
	heat_bc_matrix_2.vmult(heat_bc_tmp_2, old_heat_solution);		// ** sigma B U_(n-1)^4
	
	// assemble RHS
	heat_system_rhs.add(-1,heat_MU_tmp);							// -rho Cp M U_n
	heat_system_rhs.add(-theta * time_step, heat_LU_tmp);			// -theta dt k L U_n
	heat_system_rhs.add(-(1 - theta) * time_step, heat_tmp);		// -(1-theta) dt k L U_(n-1)
	heat_system_rhs.add(theta * time_step, heat_bc_tmp);			// theta sigma dt B U_n^4
	heat_system_rhs.add((1 - theta) * time_step, heat_bc_tmp_2);	// (1-theta) dt sigma B U_(n-1)^4
	heat_system_rhs.add(-time_step, heat_bc_rhs);					// -sigma dt B Teq^4


	heat_forcing_terms.add(time_step * (1 - theta), heat_tmp);

	heat_system_rhs += heat_forcing_terms;								// this is zero for the current case!

	// assemble system matrix
	heat_system_matrix.copy_from(heat_mass_matrix);						// rho Cp M
	heat_system_matrix.add(theta * time_step, heat_laplace_matrix);		// k dt theta L
	heat_system_matrix.add(-4 * theta * time_step, heat_bc_matrix);		// -4 sigma dt theta B Un^3

	constraints.condense(heat_system_matrix);
	constraints.condense(heat_system_rhs);

	// set boundary values for right and bottom (top boundary is captured in weak form, left boundary is flux-free)
	std::map<types::global_dof_index, double> heat_boundary_values_right;
	std::map<types::global_dof_index, double> heat_boundary_values_bottom;

	VectorTools::interpolate_boundary_values(dof_handler,
			1,
			Functions::ZeroFunction<dim>(),
			heat_boundary_values_bottom);

	VectorTools::interpolate_boundary_values(dof_handler,
			2,
			Functions::ZeroFunction<dim>(),
			heat_boundary_values_right);

	MatrixTools::apply_boundary_values(heat_boundary_values_bottom,
			heat_system_matrix,
			heat_update,
			heat_system_rhs);

	MatrixTools::apply_boundary_values(heat_boundary_values_right,
			heat_system_matrix,
			heat_update,
			heat_system_rhs);


}

template <int dim>
void HeatEquation<dim>::heat_setup_crank_nicolson_linear()
{
	heat_tmp.reinit(heat_solution.size());
	heat_bc_tmp.reinit(heat_solution.size());
	heat_forcing_terms.reinit(heat_solution.size());

	heat_mass_matrix.vmult(heat_system_rhs, old_heat_solution);
	heat_laplace_matrix.vmult(heat_tmp, old_heat_solution);
	heat_bc_matrix.vmult(heat_bc_tmp, old_heat_solution);


	heat_system_rhs.add(-(1 - theta) * time_step, heat_tmp);
	heat_system_rhs.add((1 - 4*theta) * time_step , heat_bc_tmp);
	heat_system_rhs.add(-time_step, heat_bc_rhs);

	HeatRightHandSide<dim> heat_rhs_function;
	heat_rhs_function.set_time(time);
	VectorTools::create_right_hand_side(dof_handler,
			QGauss<dim>(fe.degree + 1),
			heat_rhs_function,
			heat_tmp);
	heat_forcing_terms = heat_tmp;
	heat_forcing_terms *= time_step * theta;

	heat_rhs_function.set_time(time - time_step);
	VectorTools::create_right_hand_side(dof_handler,
			QGauss<dim>(fe.degree + 1),
			heat_rhs_function,
			heat_tmp);

	heat_forcing_terms.add(time_step * (1 - theta), heat_tmp);


	heat_system_rhs += heat_forcing_terms;

	heat_system_matrix.copy_from(heat_mass_matrix);
	heat_system_matrix.add(theta * time_step, heat_laplace_matrix);
	heat_system_matrix.add(-4 * theta * time_step, heat_bc_matrix);

	constraints.condense(heat_system_matrix, heat_system_rhs);

	std::map<types::global_dof_index, double> heat_boundary_values_right;
	std::map<types::global_dof_index, double> heat_boundary_values_bottom;
	std::map<types::global_dof_index, double> heat_boundary_values_top;
	std::map<types::global_dof_index, double> heat_boundary_values_top2;

	VectorTools::interpolate_boundary_values(dof_handler,
			1,
			heat_boundary_values_function_bottom,
			heat_boundary_values_bottom);

	VectorTools::interpolate_boundary_values(dof_handler,
			2,
			heat_boundary_values_function_right,
			heat_boundary_values_right);

	VectorTools::interpolate_boundary_values(dof_handler,
			3,
			heat_boundary_values_function_top,
			heat_boundary_values_top);

	VectorTools::interpolate_boundary_values(dof_handler,
				5,
				heat_boundary_values_function_top,
				heat_boundary_values_top);

	MatrixTools::apply_boundary_values(heat_boundary_values_bottom,
			heat_system_matrix,
			heat_solution,
			heat_system_rhs);

	MatrixTools::apply_boundary_values(heat_boundary_values_right,
			heat_system_matrix,
			heat_solution,
			heat_system_rhs);

	MatrixTools::apply_boundary_values(heat_boundary_values_top,
			heat_system_matrix,
			heat_solution,
			heat_system_rhs);

	MatrixTools::apply_boundary_values(heat_boundary_values_top2,
				heat_system_matrix,
				heat_solution,
				heat_system_rhs);

}

template <int dim>
void HeatEquation<dim>::heat_setup_crank_nicolson_simple()
{
	const double grad_adj = 0.7;

	heat_tmp.reinit(heat_solution.size());
	heat_bc_tmp.reinit(heat_solution.size());
	heat_forcing_terms.reinit(heat_solution.size());
	//cout << old_heat_solution << endl;

	heat_mass_matrix.vmult(heat_system_rhs, old_heat_solution);
	heat_laplace_matrix.vmult(heat_tmp, old_heat_solution);
	heat_bc_matrix.vmult(heat_bc_tmp, old_heat_solution);
	
	
	heat_system_rhs.add(-(1 - theta) * time_step, heat_tmp);
	heat_system_rhs.add((1 - 4*theta) * time_step *grad_adj, heat_bc_tmp);
	heat_system_rhs.add(-time_step*grad_adj, heat_bc_rhs);

	HeatRightHandSide<dim> heat_rhs_function;
	heat_rhs_function.set_time(time);
	VectorTools::create_right_hand_side(dof_handler,
			QGauss<dim>(fe.degree + 1),
			heat_rhs_function,
			heat_tmp);
	heat_forcing_terms = heat_tmp;
	heat_forcing_terms *= time_step * theta;

	heat_rhs_function.set_time(time - time_step);
	VectorTools::create_right_hand_side(dof_handler,
			QGauss<dim>(fe.degree + 1),
			heat_rhs_function,
			heat_tmp);

	heat_forcing_terms.add(time_step * (1 - theta), heat_tmp);

	heat_system_rhs += heat_forcing_terms;

	heat_system_matrix.copy_from(heat_mass_matrix);
	heat_system_matrix.add(theta * time_step, heat_laplace_matrix);
	heat_system_matrix.add(-4 * theta * time_step*grad_adj, heat_bc_matrix);

	constraints.condense(heat_system_matrix, heat_system_rhs);

	{
		std::map<types::global_dof_index, double> heat_boundary_values_right;
		std::map<types::global_dof_index, double> heat_boundary_values_bottom;

		VectorTools::interpolate_boundary_values(dof_handler,
				1,
				heat_boundary_values_function_bottom,
				heat_boundary_values_bottom);

		VectorTools::interpolate_boundary_values(dof_handler,
				2,
				heat_boundary_values_function_right,
				heat_boundary_values_right);

		MatrixTools::apply_boundary_values(heat_boundary_values_bottom,
				heat_system_matrix,
				heat_solution,
				heat_system_rhs);

		MatrixTools::apply_boundary_values(heat_boundary_values_right,
				heat_system_matrix,
				heat_solution,
				heat_system_rhs);

	}
}


template <int dim>
void HeatEquation<dim>::heat_solve_system_nonlinear()
{
 	SparseDirectUMFPACK solvedir;
 	heat_update = heat_system_rhs;
 	solvedir.solve(heat_system_matrix, heat_update);

	constraints.distribute(heat_update);

	alpha = get_alpha();
	std::cout << alpha << std::endl;

	// update cooling time step solution
	heat_solution.add(alpha, heat_update);

}


template <int dim>
void HeatEquation<dim>::heat_solve_system_linear()
{

	SolverControl solver_control(cfg.heat_iteration_coefficient, cfg.heat_tolerance_coefficient * heat_system_rhs.l2_norm());
	SolverCG<>    cg(solver_control);

	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(heat_system_matrix, 1.);

	cg.solve(heat_system_matrix, heat_solution, heat_system_rhs, preconditioner);

	constraints.distribute(heat_solution);

	std::cout << "     " << solver_control.last_step() << " CG iterations."
			<< std::endl;
}



template <int dim>
void HeatEquation<dim>::heat_solve_system_simple()
{

  
	SolverControl solver_control(cfg.heat_iteration_coefficient, cfg.heat_tolerance_coefficient * heat_system_rhs.l2_norm());
	SolverCG<>    cg(solver_control);

	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(heat_system_matrix, 1.0);

	cg.solve(heat_system_matrix, heat_solution, heat_system_rhs, preconditioner);

	constraints.distribute(heat_solution);

	std::cout << "     " << solver_control.last_step() << " CG iterations."
			<< std::endl;
}

template <int dim>
double HeatEquation<dim>::compute_residual(const double alpha) const
{
	Vector<double> residual(dof_handler.n_dofs());

	Vector<double> evaluation_point(dof_handler.n_dofs());

	evaluation_point = heat_solution;
	evaluation_point.add(alpha, heat_update);

	const QGauss<dim> quadrature_formula(fe.degree + 1);
	FEValues<dim>     fe_values(fe,
							quadrature_formula,
							update_values | update_gradients |
							update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points    = quadrature_formula.size();

	std::vector<double> new_temp(n_q_points);
	std::vector<double> prev_temp(n_q_points);

	Vector<double>              cell_residual(dofs_per_cell);
	std::vector<Tensor<1, dim>> gradients(n_q_points);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	residual = 0;
	for (const auto &cell : dof_handler.active_cell_iterators())
	  {

		fe_values.reinit(cell);

		fe_values.get_function_values(heat_solution,prev_temp);
		fe_values.get_function_values(evaluation_point,new_temp);

		for (unsigned int q = 0; q < n_q_points; ++q)
		  {
			residual(q) = (abs(new_temp[q]-prev_temp[q] )           // * \nabla u_n
								   * fe_values.JxW(q));       // * dx
		  }
	  }

	constraints.condense(residual);

	return residual.l2_norm();
}


template <int dim>
void HeatEquation<dim>::do_newton_iter()
{
	double       last_residual_norm = std::numeric_limits<double>::max();
	unsigned int refinement_cycle   = 0;

	do {
		std::cout << "Mesh refinement step " << refinement_cycle << std::endl;

		if (refinement_cycle != 0)
			refine_mesh(0, cfg.adaptive_refinement);
//			postrefine_setup();

			std::cout << "  Initial residual: " << compute_residual(0) << std::endl;

			for (unsigned int inner_iteration = 0; inner_iteration < cfg.max_iters;
			++inner_iteration)
			{
				std::cout << heat_system_rhs.l2_norm() << std::endl;
				// set up system
				heat_compute_mass_and_laplace_matrices();
				heat_setup_crank_nicolson_nonlinear();

				// compute previous residual norm
				last_residual_norm = heat_system_rhs.l2_norm();

				// solve for D
				heat_solve_system_nonlinear();

				// print iteration residual
				std::cout << "  Residual: " << compute_residual(alpha) << std::endl;

			}
			graphical_output_results(1);
			++refinement_cycle;
			std::cout << std::endl;
	}
	while (refinement_cycle < 4);

}

template <int dim>
void HeatEquation<dim>::do_linear()
{
	heat_compute_mass_and_laplace_matrices();
	heat_setup_crank_nicolson_linear();
	heat_solve_system_linear();
	graphical_output_results(1);
//	textual_output_results();
}


template <int dim>
void HeatEquation<dim>::do_heat_step_simple()
{
	heat_compute_mass_and_laplace_matrices_simple();
	heat_setup_crank_nicolson_simple();
	heat_solve_system_simple();
	graphical_output_results_simple();

}

template <int dim>
void HeatEquation<dim>::graphical_output_results(unsigned int newton_iter) const
{
	if (timestep_number % 1 == 0) {
		DataOut<dim> data_out;

		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(heat_solution, "solution");
		data_out.add_data_vector(kappa_vals, "kappa");
//		data_out.add_data_vector(heat_update, "update");

		data_out.build_patches();

		const std::string filename = cfg.output_folder +
				"/solution-" +  Utilities::int_to_string(timestep_number, 3) + ".vtk";
//				"/solution-" +  Utilities::int_to_string(newton_iter, 3) + ".vtk";
		std::ofstream output(filename);
		data_out.write_vtk(output);

	}
}
template <int dim>
void HeatEquation<dim>::graphical_output_results_simple() const
{
	if (timestep_number % 1 == 0) {
		DataOut<dim> data_out;

		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(heat_solution, "U");
		data_out.add_data_vector(old_heat_solution, "U_old");

		data_out.build_patches();

		const std::string filename = cfg.output_folder +
				"/solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
		std::ofstream output(filename);
		data_out.write_vtk(output);

	}
}

template <int dim>
void HeatEquation<dim>::textual_output_results() const
{
	Vector<double> heat_solution_uniform;
	heat_solution_uniform.reinit(dof_handler_preref.n_dofs());

	double r_value;
	double z_value;

	std::cout << heat_solution.size() << "\n";
	std::cout << dof_handler.n_dofs() << "\n";


	//  interpolate heat solution to regular grid
	VectorTools::interpolate_to_different_mesh(dof_handler, heat_solution,
			dof_handler_preref, heat_solution_uniform);


	const std::string filename = cfg.output_folder + "/heat_solutions.txt";
	std::ofstream out(filename, std::ios::app);
	out << heat_solution_uniform << "\n";


}


template <int dim>
void HeatEquation<dim>::set_initial_temperature()
{
	InitialTemperature<dim> t_init;
	t_init.set_initial_temperature_field(x_vec,z_vec,initial_temperature_mat);

	VectorTools::interpolate(dof_handler,
			t_init,
			old_heat_solution);
	
	heat_solution = old_heat_solution;
	
	// graphical_output_results_simple();

	bool evaluation_point_found = false;
	T_bottom = 100;

	for (const auto &cell : dof_handler.active_cell_iterators())
	  if (!evaluation_point_found)
		for (const auto vertex : cell->vertex_indices())
		  if (cell->vertex(vertex) == Point<dim>(cfg.x_right,cfg.z_bottom))
			{
			  T_bottom = heat_solution(cell->vertex_dof_index(vertex, 0));
			  evaluation_point_found = true;
			  break;
			};

	// load kappa data if necessary
	if (cfg.kappa_type == 1) {
		k_init.set_kappa(x_vec,z_vec,kappa_mat);
	}

};

template <int dim>
void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
								  const unsigned int max_grid_level)
{


	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
	KellyErrorEstimator<dim>::estimate(
		  dof_handler,
		  QGauss<dim - 1>(fe.degree + 1),
		  std::map<types::boundary_id, const Function<dim> *>(),
		  heat_solution,
		  estimated_error_per_cell);

	GridRefinement::refine_and_coarsen_fixed_number(triangulation,
													estimated_error_per_cell,
													0.05,//2,
													0.3,
													100000);//.25);


	if (triangulation.n_levels() > max_grid_level)
	  for (const auto &cell :
		   triangulation.active_cell_iterators_on_level(max_grid_level))
		cell->clear_refine_flag();

	for (const auto &cell :
		 triangulation.active_cell_iterators_on_level(min_grid_level))
	  cell->clear_coarsen_flag();


	// transfer solution
	SolutionTransfer<dim> solution_trans(dof_handler);
	SolutionTransfer<dim> old_solution_trans(dof_handler);

	Vector<double> previous_solution;
	previous_solution = heat_solution;

	Vector<double> previous_old_solution;
	previous_old_solution = old_heat_solution;

	triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
	old_solution_trans.prepare_for_coarsening_and_refinement(previous_old_solution);

	triangulation.execute_coarsening_and_refinement();
	postrefine_setup(); // make sure this should be here

	solution_trans.interpolate(previous_solution, heat_solution);
	old_solution_trans.interpolate(previous_old_solution, old_heat_solution);

	constraints.distribute(heat_solution);

};


template <int dim>
void HeatEquation<dim>::unrefine_mesh()
{

	unsigned int unrefinement_cycle   = 0;
	do {
		for (const auto &cell : triangulation.active_cell_iterators()) {
					cell->set_coarsen_flag();
			}

		SolutionTransfer<dim> solution_trans(dof_handler);

		Vector<double> previous_solution;
		previous_solution = heat_solution;

		triangulation.prepare_coarsening_and_refinement();
		solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

		triangulation.execute_coarsening_and_refinement();

		postunrefine_setup();

		solution_trans.interpolate(previous_solution, heat_solution);


		++unrefinement_cycle;
	}

	while (unrefinement_cycle < cfg.adaptive_refinement+1);

};


template <int dim>
void HeatEquation<dim>::refine_mesh_simple(const unsigned int min_grid_level,
								  const unsigned int max_grid_level)
{
	Vector<float> gradient_indicator(triangulation.n_active_cells());

	DerivativeApproximation::approximate_gradient(dof_handler,
												  heat_solution,
												  gradient_indicator);


	GridRefinement::refine_and_coarsen_fixed_number(triangulation,
													gradient_indicator,
													0.1,//2,
													0.3,
													100000);//.25);
	for (const auto &cell : triangulation.active_cell_iterators())
		gradient_indicator[cell->active_cell_index()] *= (cell->diameter());

	if (triangulation.n_levels() > max_grid_level)
	  for (const auto &cell :
		   triangulation.active_cell_iterators_on_level(max_grid_level))
		cell->clear_refine_flag();

	for (const auto &cell :
		 triangulation.active_cell_iterators_on_level(min_grid_level))
	  cell->clear_coarsen_flag();


	// transfer solution
	SolutionTransfer<dim> solution_trans(dof_handler);

	Vector<double> previous_solution;
	previous_solution = heat_solution;

	triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

	triangulation.execute_coarsening_and_refinement();
	if (timestep_number < 2)
		heat_setup_system();
	else
		postrefine_setup(); // make sure this should be here

	solution_trans.interpolate(previous_solution, heat_solution);

	constraints.distribute(heat_solution);


};


template <int dim>
void HeatEquation<dim>::run_simple()
{
	clear_output_directory();
	// read mesh from file
	theta = 0.5;

	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);

	std::ifstream mesh_stream(cfg.mesh_filename,
			std::ifstream::in);

	grid_in.read_ucd(mesh_stream);

	// copy original, unrefined triangulation to use for text output
	triangulation_preref.copy_triangulation(triangulation);

	// do initial global refinement
	if (cfg.global_refinement > 0)
	{
		triangulation.refine_global(cfg.global_refinement);
	};

	// set boundary indicators
	set_boundary_indicators();

	// load initial temperature field
	load_initial_temperature();
	dof_handler_preref.distribute_dofs(fe);
	std::cout << "Loaded initial temp" << std::endl;
	{
		Timer timer("setup system");
		heat_setup_system();
	}

	unsigned int pre_refinement_step = 0;

	start_time_iteration:
		std::cout << "starting time iteration" << std::endl;
		time = 0.0;
		timestep_number = 0;


		// this needs to be fields of HeatProblem
		heat_tmp.reinit(heat_solution.size());
		heat_bc_tmp.reinit(heat_solution.size());
		heat_forcing_terms.reinit(heat_solution.size());


		// set boundary indicators
		set_boundary_indicators();

		// heat_boundary_values_function_top.set_surface_temperature(cfg.T_surf);
		heat_boundary_values_function_right.set_temperature_field(x_vec,z_vec,eq_temperature_mat);
		heat_boundary_values_function_bottom.set_bottom_temperature(T_bottom);
		//std::cout << "Boundary functions set" << std::endl;

		heat_setup_system();

		graphical_output_results_simple();

		while (time <= cfg.final_time)
		{

			time += time_step;
			++timestep_number;

			std::cout << "Time step " << timestep_number << std::endl;

			std::cout << " at t= " << time/SECSINYEAR/1e6 << " My" << std::endl;

			do_heat_step_simple();

			graphical_output_results_simple();

			if ((timestep_number == 1) && (pre_refinement_step < cfg.adaptive_refinement))
			  {

				std::cout << "Initial refinement " << pre_refinement_step << std::endl;
				refine_mesh_simple(cfg.global_refinement,
							cfg.global_refinement +
							  cfg.adaptive_refinement);
				++pre_refinement_step;

				
				
				heat_tmp.reinit(heat_solution.size());
				heat_bc_tmp.reinit(heat_solution.size());
				heat_forcing_terms.reinit(heat_solution.size());

				std::cout << std::endl;
				

				goto start_time_iteration;
			  }
			else if ((timestep_number > 0) && (timestep_number % 10 == 0))
			  {
				refine_mesh_simple(cfg.global_refinement,
							cfg.global_refinement +
							  cfg.adaptive_refinement);

				heat_tmp.reinit(heat_solution.size());
				heat_bc_tmp.reinit(heat_solution.size());
				heat_forcing_terms.reinit(heat_solution.size());
			  }


			old_heat_solution = heat_solution;
			time_step = time_step*1.05;
		}
}

template <int dim>
void HeatEquation<dim>::run_newton()
{
	// runtype options: 0 (linear), 1 (newton iterations before reaching equilibrium)
	
	std::cout << "******** beginning setup **********" << std::endl;

	clear_output_directory();
	theta = 0;

	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);

	std::ifstream mesh_stream(cfg.mesh_filename,
			std::ifstream::in);

	// read mesh from file and set boundary IDs if necessary
	if (cfg.mesh_filename.substr(cfg.mesh_filename.find_last_of(".") + 1) == "inp") {
		grid_in.read_ucd(mesh_stream);
		set_boundary_indicators();

	} else if (cfg.mesh_filename.substr(cfg.mesh_filename.find_last_of(".") + 1) == "msh") {
		grid_in.read_msh(mesh_stream);
	}

	std::cout << "Read in mesh at " + cfg.mesh_filename << std::endl;

	// copy original, unrefined triangulation to use for text output
	triangulation_preref.copy_triangulation(triangulation);

	// load initial temperature field and do setup
	load_initial_temperature();
	dof_handler_preref.distribute_dofs(fe);
	std::cout << "Loaded initial temp" << std::endl;
	{
		Timer timer("setup system");
		heat_setup_system();
	}

	unsigned int pre_refinement_step = 0;
	unsigned int equil_step = 0;

	start_time_iteration:
		std::cout << "******** starting time iteration ********" << std::endl;
		time = 0.0;
		timestep_number = 0;
		bool in_equil = false;

		// define boundary value functions
		heat_boundary_values_function_right.set_temperature_field(x_vec,z_vec,initial_temperature_mat);
		heat_boundary_values_function_bottom.set_bottom_temperature(T_bottom);
		heat_boundary_values_function_top.set_surface_temperature(T_eq);
		heat_boundary_values_function_top2.set_surface_temperature(T_eq);

		if (cfg.kappa_type == 1) {
			VectorTools::interpolate(dof_handler,
								k_init,
								kappa_vals);
		}
		// output initial system state
		graphical_output_results(1);

		// set up system and set boundary values
		heat_setup_system();
		set_system_BCs();

		// start time stepping!
		while (time <= cfg.final_time)
		{
			time += time_step;
			++timestep_number;

			std::cout << "Time step " << timestep_number << std::endl;
			std::cout << " at t= " << time/SECSINYEAR/1e6 << " My" << std::endl;

			// solve system--if surface is close to equilibrium, solve as Dirichlet BC problem, otherwise use Newton
			// iterations to solve nonlinear BC
			
			if (in_equil) {
				std::cout << "Starting simple solver" << std::endl;
				// do refinement if necessary
				start_equil_refine:
					if ((equil_step == 0)&& (pre_refinement_step < cfg.adaptive_refinement)){
						
						refine_mesh(cfg.global_refinement,
									cfg.global_refinement +
									  cfg.adaptive_refinement);
						++pre_refinement_step;

						heat_tmp.reinit(heat_solution.size());
						heat_bc_tmp.reinit(heat_solution.size());
						heat_forcing_terms.reinit(heat_solution.size());

						goto start_equil_refine;

					} else if (equil_step % 10 == 0) {
						refine_mesh(cfg.global_refinement,
									cfg.global_refinement +
									  cfg.adaptive_refinement);

						heat_tmp.reinit(heat_solution.size());
						heat_bc_tmp.reinit(heat_solution.size());
						heat_forcing_terms.reinit(heat_solution.size());
					}

				// do time step
				do_linear();

				++equil_step;

			} else {

				// do newton iterations to solve time step
				do_newton_iter();

				// transfer solution back to old mesh
				unrefine_mesh();
				set_system_BCs();

				// check if surface is in equilibrium
				in_equil = check_equil();

			}

			// set up for next time step
			old_heat_solution = heat_solution;
			time_step = time_step*1.05;

			// distribute constraints
			constraints.distribute(heat_solution);
		}
}

int main(int argc, char* argv[])
{
	try
	{
		using namespace dealii;
		char* config_filename = new char[120];

		if (argc == 1) // if no input parameters (as if launched from eclipse)
		{
			// std::strcpy(config_filename,"/home/basinuser/BasinUser/BasinCooling/BasinData/InPaper/200km/config.cfg");
			std::strcpy(config_filename,"/home/basinuser/BasinUser/BasinCooling/BasinData/InPaper/2200km/config.cfg");
		}
		config_in cfg(config_filename);
		HeatEquation<2> heat_equation_solver(cfg);
		if (cfg.run_type == 0){
			heat_equation_solver.run_simple();
		} else if (cfg.run_type == 1){
			heat_equation_solver.run_newton();
		} else {
			std::cout << "Invalid run type!" << std::endl;
		}
		
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)

	{
		std::cerr << std::endl
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
