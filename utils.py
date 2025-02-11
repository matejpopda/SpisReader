from simulation import *
import fnmatch
import reader
import numpy as np
import numpy.typing as npt
import typing


def glob_properties(
    input: (
        Simulation
        | Mesh
        | SimulationPreprocessing
        | SimulationResults
        | ExtractedDataFields
        | NumericalResults
        | list[ParticleDetector]
        | list[Mesh]
    ),
    property: str,
    *,
    ignore_num_kernel: bool = True,
    exclude: str | None = None,
) -> list[tuple[Mesh, "str"]]:
    result: list[tuple[Mesh, "str"]] = []
    if isinstance(input, Mesh):
        unloaded_keys = fnmatch.filter(input.loadable_properties.keys(), property)
        if len(unloaded_keys) != 0:
            for key in unloaded_keys:
                reader.load_property_into_mesh(input, input.loadable_properties.pop(key))

        strings = fnmatch.filter(input.properties, property)
        for i in strings:
            if exclude is None or not fnmatch.fnmatch(i, exclude):
                result.append((input, i))

        return result

    if isinstance(input, Simulation):
        result += glob_properties(
            input.preprocessing,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.results,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    if isinstance(input, SimulationPreprocessing):
        result += glob_properties(
            input.model,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    if isinstance(input, SimulationResults):
        result += glob_properties(
            input.extracted_data_fields,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        if not ignore_num_kernel:
            result += glob_properties(
                input.numerical_kernel_output,
                property=property,
                ignore_num_kernel=ignore_num_kernel,
                exclude=exclude,
            )

    if isinstance(input, NumericalResults):
        result += glob_properties(
            input.particle_detectors,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    if isinstance(input, list):
        for i in input:
            if isinstance(i, ParticleDetector):
                result += glob_properties(
                    i.differential_flux_mesh,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )
                result += glob_properties(
                    i.initial_distribution_mesh,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )
                result += glob_properties(
                    i.distribution_function_mesh,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )
            if isinstance(i, Mesh):
                result += glob_properties(
                    i,
                    property=property,
                    ignore_num_kernel=ignore_num_kernel,
                    exclude=exclude,
                )

    if isinstance(input, ExtractedDataFields):
        result += glob_properties(
            input.spacecraft_face,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.spacecraft_mesh,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.spacecraft_vertex,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.volume_vertex,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )
        result += glob_properties(
            input.display_vol_mesh,
            property=property,
            ignore_num_kernel=ignore_num_kernel,
            exclude=exclude,
        )

    return result


def generate_efield_vector_property(simulation: Simulation):
    x_value = glob_properties(simulation, "*finalplasma_elec_field_Ex*")
    assert len(x_value) == 1
    x_property_name = x_value[0][1]
    # print(x_value[0][0].mesh.point_data)
    x_array = x_value[0][0].mesh.get_array(x_property_name, "point") # type: ignore
    x_array = typing.cast(npt.NDArray[np.float64], x_array)
    
    y_value = glob_properties(simulation, "*finalplasma_elec_field_Ey*")
    assert len(y_value) == 1    
    y_property_name = y_value[0][1]
    y_array = y_value[0][0].mesh.get_array(y_property_name, "point") # type: ignore
    y_array = typing.cast(npt.NDArray[np.float64], y_array)
    
    z_value = glob_properties(simulation, "*finalplasma_elec_field_Ez*")
    assert len(z_value) == 1
    z_property_name = z_value[0][1]
    z_array = z_value[0][0].mesh.get_array(z_property_name, "point") # type: ignore
    z_array = typing.cast(npt.NDArray[np.float64], z_array)
    
    assert z_value[0][0] == y_value[0][0] == x_value[0][0]


    result = np.column_stack([x_array, y_array, z_array]) 

    # print(result)

    x_value[0][0].mesh["vector_electric_field"] = result

    return x_value[0][0].mesh