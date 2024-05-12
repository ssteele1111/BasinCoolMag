/*
 * right_hand_side.cc
 *
 *  Created on: Feb 16, 2021
 *      Author: eai
 */

#include <deal.II/base/function.h>

using namespace dealii;

// @sect3{Equation data}

// In the following classes and functions, we implement the various pieces
// of data that define this problem (right hand side and boundary values)
// that are used in this program and for which we need function objects. The
// right hand side is chosen as discussed at the end of the
// introduction. For boundary values, we choose zero values, but this is
// easily changed below.
template <int dim>
class HeatRightHandSide : public Function<dim>
{
public:
	HeatRightHandSide() : Function<dim>()
  {}

	virtual double value(const Point<dim> & p, const unsigned int component = 0) const override;
};

template <int dim>
double HeatRightHandSide<dim>::value(const Point<dim> & p, const unsigned int component) const
{
	(void)component;
	AssertIndexRange(component, 1);
	Assert(dim == 2, ExcNotImplemented());

	const double time = this->get_time();

	//	if ((p[0] > 100000.0) && (p[1] > -80000.0) && (p[0] < 150000.0) && (p[1] < -40000.0))
	//	  return 0.0;
	//	else
	//	  return 0.0;

	return 0.0;
};


