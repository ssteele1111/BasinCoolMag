/*
 * constants.h
 *
 *  Created on: Aug 1, 2020
 *      Author: eai
 */

#include <cmath>

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

using namespace std;

const double PI      = M_PI;
const double INVPI   = M_1_PI;
const double TWOPI   = 2.0*PI;
const double FOURPI  = 4.0*PI;
const double HALFPI  = M_PI_2;
const double EULER   = M_E;
const double SQRT2   = M_SQRT2;

const double G                 = 6.67430e-11;      // m^3 kg^–1 s^–2
const double PIG               = PI*G;
const double GASCONSTANT       = 8.31446261815324; // J K^-1 mol^−1
const double BOLTZMANCONSTANT  = 1.380649e-23;     // J K^-1
const double AVOGADROCONTSTANT = 6.02214076e23;    // mol^-1

const double SECSINGY     = 1.0e9*365.2422*86400.0;
const double SECSINMY     = 1.0e6*365.2422*86400.0;
const double SECSINKY     = 1.0e3*365.2422*86400.0;

const double SECSINYEAR   = 365.2422*86400.0;
const double SECSINHOUR   = 3600.0;
const double SECSINMINUTE = 60.0;

const double DAYSINYEAR   = 365.2422;

const double METERSINAU   = 1.495978707e11; // m, from wikipedia
const double LUNARDIST    = 384402000.0;    // m, average Earth-Moon distance, from wikipedia
const double EARTHRADIUS  = 6371000.8;      // m, Earth volume-equivalent, from wikipedia

const double LIGHTSPEED   = 299792458.0; // m

const double R2D = 180.0/PI; // to convert radians to degree multiply by R2D
const double D2R = PI/180.0; // to convert degrees to radians multiply by this D2R
const double ARCSECSINRAD = R2D*3600.0;
const double ARCMINSINRAD = R2D*60.0;

#endif /* CONSTANTS_H_ */
