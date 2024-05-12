/*
 * boundary_values.cc
 *
 *  Created on: Feb 16, 2021
 *      Author: eai
 */

#include <deal.II/base/function.h>
#include <armadillo>
#include <iterator>
#include <fstream>


using namespace dealii;
using namespace arma;

// Bottom boundary condition
template <int dim>
class HeatBoundaryValuesBottom : public Function<dim>
{
public:
	virtual double value(const Point<dim>& p,
			const unsigned int component = 0) const override;
	void set_bottom_temperature(double);
	double T_bottom;
  
};


template <int dim>
void HeatBoundaryValuesBottom<dim>::set_bottom_temperature(double T_bottomi)
{
	T_bottom = T_bottomi;
}

template <int dim>
double HeatBoundaryValuesBottom<dim>::value(const Point<dim> & /*p*/,
		const unsigned int component) const
		{
	(void)component;
	Assert(component == 0, ExcIndexRange(component, 0, 1));
	return T_bottom;
		};
//template <int dim>
//void HeatBoundaryValuesBottom<dim>::set_temperature_field(vec x_veci, vec z_veci, mat temp_mapi)
//{
//	x_vec = x_veci;
//	z_vec = z_veci;
//	temp_mat = temp_mapi;
//};
//
//template <int dim>
//double HeatBoundaryValuesBottom<dim>::value(const Point<dim>& p,
//		const unsigned int component) const
//		{
//	(void)component;
//	Assert(component == 0, ExcIndexRange(component, 0, 1));
//
//	vec r_value_vec(1);
//	vec z_value_vec(1);
//	mat temp_value_mat(1,1);
//
//	r_value_vec(0) = p.operator ()(0);
//	z_value_vec(0) = p.operator ()(1);
//
//
//
//	interp2(x_vec, z_vec, temp_mat, r_value_vec, z_value_vec, temp_value_mat);
//
////	    std::cout << r_value_vec(0) << " " << z_value_vec(0) << " " << temp_value_mat(0,0) << std::endl;
//	//    return z_value_vec(0);
////	std::cout << temp_value_mat(0,0) << std::endl;
//	return temp_value_mat(0,0);
////		return 150.0;
//		};


// Right boundary condition
template <int dim>
class HeatBoundaryValuesRight : public Function<dim>
{
public:
	virtual double value(const Point<dim>& p,
			const unsigned int component = 0) const override;
	void set_temperature_field(vec x_veci, vec z_veci, mat temp_mapi);
	vec x_vec;
	vec z_vec;
	mat temp_mat;

};

template <int dim>
void HeatBoundaryValuesRight<dim>::set_temperature_field(vec x_veci, vec z_veci, mat temp_mapi)
{
	x_vec = x_veci;
	z_vec = z_veci;
	temp_mat = temp_mapi;

	
}

template <int dim>
double HeatBoundaryValuesRight<dim>::value(const Point<dim>& p,
		const unsigned int component) const
		{
	(void)component;
	Assert(component == 0, ExcIndexRange(component, 0, 1));

	vec r_value_vec(1);
	vec z_value_vec(1);
	mat temp_value_mat(1,1);

	r_value_vec(0) = p.operator ()(0);
	z_value_vec(0) = p.operator ()(1);

	interp2(x_vec, z_vec, temp_mat, r_value_vec, z_value_vec, temp_value_mat);

	//    std::cout << r_value_vec(0) << " " << z_value_vec(0) << " " << temp_value_mat(0,0) << std::endl;
	//    return z_value_vec(0);
	return temp_value_mat(0,0);
	//	return 150.0;
		}

// Top boundary condition
template <int dim>
class HeatBoundaryValuesTop : public Function<dim>
{
public:
	virtual double value(const Point<dim>& p,
			const unsigned int component = 0) const override;
	void set_surface_temperature(double);
	double T_surf;
};


template <int dim>
void HeatBoundaryValuesTop<dim>::set_surface_temperature(double T_surfi)
{
	T_surf = T_surfi;
}

template <int dim>
double HeatBoundaryValuesTop<dim>::value(const Point<dim> & /*p*/,
		const unsigned int component) const
		{
	(void)component;
	Assert(component == 0, ExcIndexRange(component, 0, 1));
	return T_surf;
		};




