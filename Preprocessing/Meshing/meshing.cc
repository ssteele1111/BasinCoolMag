
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;


void new_grid(const float x_min,
              const float x_max,
              const float y_min,
              const float y_max,
              const float z_max,
              const std::vector<float>& cell_size,
              const std::vector<int>& material_ids,
              const std::vector<float>& material_depths,
              const std::string& savefp)
{
    // grid including regolith, crust, and mantle
    // specify total depth of regolith and crust boundaries in input, not thicknesses

    float widthm = (x_max - x_min);
    float depthm = z_max;

    Triangulation<3> triangulation;
    float current_depth = 0.0;

	std::cout << "Starting mesh generation" << std::endl;

    for (size_t i = 0; i < material_ids.size(); ++i)
    {
        int material_id = material_ids[i];
        float material_depth = material_depths[i];

		std::cout << "Material " << i << " depth: " << material_depth << std::endl;
		std::cout << "Cell size: " << cell_size[i] << std::endl;
        std::cout << "Current dpth: " << current_depth << std::endl;
        std::cout << "Zmax: " << z_max << std::endl;
        if (current_depth <= z_max)
            break;

        float layer_depth = std::max(material_depth, z_max);
        unsigned int layer_cells = static_cast<unsigned int>((current_depth - layer_depth) / cell_size[i]);
		unsigned int r0 = static_cast<unsigned int>((x_max - x_min) / cell_size[i]);
		unsigned int r1 = static_cast<unsigned int>((y_max - y_min) / cell_size[i]);
		
		std::cout << "Layer cells: " << layer_cells << std::endl;

        std::vector<unsigned int> reps{r0, r1, layer_cells};
		std::cout << "Reps: " << reps[0] << " " << reps[1] << " " << reps[2] << std::endl;

        const Point<3> inner(x_min, y_min, current_depth);
        const Point<3> outer(x_max, y_max, layer_depth);

        Triangulation<3> layer_triangulation;
        GridGenerator::subdivided_hyper_rectangle(layer_triangulation, reps, inner, outer);

        for (const auto &cell : layer_triangulation.active_cell_iterators())
        {
            cell->set_material_id(material_id);
        }

		// combine meshes
		Triangulation<3> triangulation_final;
		GridGenerator::merge_triangulations(triangulation,layer_triangulation,
				triangulation_final,1.0e-12);
        std::cout << "Merged" << i << std::endl;

		triangulation.clear();
        triangulation.copy_triangulation(triangulation_final);
        current_depth = layer_depth;

		std::cout << "Layer " << i << " complete" << std::endl;
    }

    // Save the mesh to a file
    std::ofstream out(savefp);
    GridOut grid_out;
    grid_out.write_ucd(triangulation, out);
    out.close();
}

void new_grid2(const float x_min,
              const float x_max,
              const float y_min,
              const float y_max,
              const float z_max,
              const std::vector<float>& cell_size,
              const std::vector<int>& material_ids,
              const std::vector<float>& material_depths,
              const std::string& savefp)
{
    // grid including regolith, crust, and mantle
    // specify total depth of regolith and crust boundaries in input, not thicknesses

    std::vector<unsigned int> reps1{80, 80, 1};

    const Point<3> inner(-40000.,-40000., 0.);
    const Point<3> outer(40000., 40000., -1000.);

    Triangulation<3> triangulation1;
    GridGenerator::subdivided_hyper_rectangle(triangulation1, reps1, inner, outer);

    for (const auto &cell : triangulation1.active_cell_iterators())
    {
        cell->set_material_id(1);
    }

    std::vector<unsigned int> reps2{40, 40, 5};

    const Point<3> inner2(-40000.,-40000., -1000.);
    const Point<3> outer2(40000., 40000., -11000.);

    // combine meshes
    Triangulation<3> triangulation2;
    GridGenerator::subdivided_hyper_rectangle(triangulation2, reps2, inner2, outer2);

    for (const auto &cell : triangulation1.active_cell_iterators())
    {
        cell->set_material_id(2);
    }

    Triangulation<3> triangulation_final;
    GridGenerator::merge_triangulations({&triangulation1,&triangulation2},
            triangulation_final,1.0e-12);
    

    // Save the mesh to a file
    std::ofstream out(savefp);
    GridOut grid_out;
    grid_out.write_ucd(triangulation_final, out);
    out.close();
} 

int main() 
{
	// rcm_grid(15.,5.,0.1,1.,50.,"/home/sarahcate98/Sarah/BasinCooling/BaseMeshes/5_10km_rcm");
	// new_grid(-10000.,10000.,-8000.,8000.,-10000.,{1000.,1000.,1000.},{1,2,3},{-2000.,-35000.,-45500.},"/home/mike/Sarah/Mercury/LT1000/mesh.inp");
    //new_grid(-40000.,40000.,-40000.,40000.,-8000.,{2000.,2000.,2000.},{1,2,3},{-2000.,-35000.,-45500.},"/home/mike/Sarah/Mercury/DSK10000/mesh.inp");
    new_grid(-80000.,80000.,-80000.,80000.,-28000.,{4000.,4000.,4000.},{2,2,3},{-4000.,-36000.,-45500.},"/home/mike/Sarah/Mercury/DSK25000/mesh.inp");
    // new_grid(-50.,50.,-100.,100.,-600.,{10.,10.,10.},{1,2,3},{-2000.,-35000.,-45500.},"/home/mike/Sarah/Mercury/LT25/mesh.inp");
    // new_grid(-6000.,6000.,-10000.,10000.,-8000.,{2000.,2000.,2000.},{1,2,3},{-2000.,-35000.,-45500.},"/home/mike/Sarah/Mercury/LT1000/mesh.inp");
    //new_grid(x_min,x_max,y_min,y_max,z_max,cell_size,material_ids,material_depths,savefp)
}
