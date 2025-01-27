/*
 * initial_temperature.cc
 *
 *  Created on: Feb 16, 2021
 *      Author: eai
 */

#include <deal.II/base/function.h>
#include <armadillo>

using namespace dealii;
using namespace arma;

// Initial temperature function
template <int dim>
class InitialTemperature3D : public Function<dim>
{
public:
	virtual double value(const Point<dim> & p,
			const unsigned int component = 0) const override;

	void set_initial_temperature_field_3D(vec x_veci, vec y_veci, vec z_veci, cube temp_mapi);
	vec x_vec;
	vec y_vec;
	vec z_vec;
	cube temp_mat;
};


template <int dim>
void InitialTemperature3D<dim>::set_initial_temperature_field_3D(vec x_veci, vec y_veci, vec z_veci, cube temp_mapi)
{
	x_vec = x_veci;
	y_vec = y_veci;
	z_vec = z_veci;

	int x_dim = x_vec.n_elem;
    int y_dim = y_vec.n_elem;
    int z_dim = z_vec.n_elem;

	temp_mat = temp_mapi;
	temp_mat.reshape(x_dim, y_dim, z_dim);

	// temp_mat.set_size(z_dim, y_dim, x_dim);

	// for (int z = 0; z < z_dim; ++z) {
    //     for (int y = 0; y < y_dim; ++y) {
    //         for (int x = 0; x < x_dim; ++x) {
    //             temp_mat[z][y][x] = temp_mapi[z][y * x_dim + x];
    //         }
    //     }
    // }
};

template <int dim>
double InitialTemperature3D<dim>::value(const Point<dim> & p,
		const unsigned int component) const
		{
	(void)component;
	Assert(component == 0, ExcIndexRange(component, 0, 1));

	vec x_value_vec(1);
	vec y_value_vec(1);
	vec z_value_vec(1);
	vec dz;
	cube temp_value_mat(1,1,1);

	double x = p.operator ()(0);
	double y = p.operator ()(1);
	double z = p.operator ()(2);

	x_value_vec(0) = x;
	y_value_vec(0) = y;
	z_value_vec(0) = z;

	
	dz = (z_vec-z)*(z_vec-z)
	uword zi = index_min(dz)

	interp2(x_vec, y_vec, temp_mat.slice(uword), x_value_vec, y_value_vec, temp_value_mat,"nearest",800.);
	
	
	double T_out;
	T_out = temp_value_mat(0,0,0);

	//    limit temperature field
	//    if (temp_value_mat(0,0)>273.0)
	//    	T_out = 273.0;
	//    else if (temp_value_mat(0,0)<150.0)
	//    	T_out = 150.0;
	//    else
	//    	T_out = temp_value_mat(0,0);

	return T_out;
		};





template <int dim>
class Kappa : public Function<dim>
{
public:
	virtual double value(const Point<dim> & p,
			const unsigned int component = 0) const override;

	void set_kappa(vec x_veci, vec z_veci, mat kappa_mapi);
	vec x_vec;
	vec z_vec;
	mat kappa_mat;
};

template <int dim>
void Kappa<dim>::set_kappa(vec x_veci, vec z_veci, mat kappa_mapi)
{
	x_vec = x_veci;
	z_vec = z_veci;
	kappa_mat = kappa_mapi;
};

template <int dim>
double Kappa<dim>::value(const Point<dim> & p,
		const unsigned int component) const
		{
	(void)component;
	Assert(component == 0, ExcIndexRange(component, 0, 1));

	vec r_value_vec(1);
	vec z_value_vec(1);
	mat kappa_value_mat(1,1);

	double r = p.operator ()(0);
	double z = p.operator ()(1);

	r_value_vec(0) = r;
	z_value_vec(0) = z;

	interp2(x_vec, z_vec, kappa_mat, r_value_vec, z_value_vec, kappa_value_mat,"nearest",4.);

	double kappa_out;
	kappa_out = kappa_value_mat(0,0);

	//    limit temperature field
	//    if (temp_value_mat(0,0)>273.0)
	//    	T_out = 273.0;
	//    else if (temp_value_mat(0,0)<150.0)
	//    	T_out = 150.0;
	//    else
	//    	T_out = temp_value_mat(0,0);

	return kappa_out;
		};






