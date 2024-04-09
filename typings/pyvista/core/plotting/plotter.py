
"""PyVista plotting module."""
from typing import Optional

import numpy as np
import pyvista.plotting

import pyvista
from pyvista.core.utilities.misc import abstract_class

import pyvista.core

from .picking import PickingHelper
from .widgets import WidgetHelper

SUPPORTED_FORMATS = [".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff"]




from typing import Sequence, Any

ColorLike = str

@abstract_class
class BasePlotter(PickingHelper, WidgetHelper):
    ...
    
    def add_mesh(
        self,
        mesh: pyvista.core.dataset.DataSet|pyvista.core.composite.MultiBlock,
        color: Optional[ColorLike]=None,
        style: Optional[str]=None,
        scalars: Optional[str]=None, # more stuff
        clim: Optional[Sequence[float]]=None,
        show_edges: Optional[bool]=None,
        edge_color: Optional[ColorLike]=None,
        point_size: Optional[float]=None,
        line_width: Optional[float]=None,
        opacity: Optional[float| str]=None, #more
        flip_scalars: Optional[bool]=False,
        lighting: Optional[bool]=None,
        n_colors: Optional[int]=256,
        interpolate_before_map: Optional[bool]=None,
        cmap: Optional[str| list[str]]=None, # more
        label: Optional[str]=None,
        reset_camera: Optional[bool]=None,
        scalar_bar_args: Optional[dict[Any, Any]]=None,
        show_scalar_bar: Optional[bool]=None,
        multi_colors: bool|str|Sequence[ColorLike]=False,
        name: Optional[str]=None,
        texture: Optional[Any]=None, #TBD
        render_points_as_spheres: Optional[bool]=None,
        render_lines_as_tubes: Optional[bool]=None,
        smooth_shading: Optional[bool]=None,
        split_sharp_edges: Optional[bool]=None,
        ambient: Optional[float]=None,
        diffuse: Optional[float]=None,
        specular: Optional[float]=None,
        specular_power: Optional[float]=None,
        nan_color: Optional[ColorLike]=None,
        nan_opacity: Optional[float]=1.0,
        culling: Optional[str]=None,
        rgb: Optional[bool]=None,
        categories: Optional[bool]=False,
        silhouette: Optional[float| dict[Any, Any]]=None,
        use_transparency: Optional[bool]=False,
        below_color: Optional[ColorLike]=None,
        above_color: Optional[ColorLike]=None,
        annotations: Optional[dict[Any, Any]]=None,
        pickable: Optional[bool]=True,
        preference:str="point",
        log_scale:bool=False,
        pbr: Optional[bool]=None,
        metallic: Optional[float]=None,
        roughness: Optional[float]=None,
        render: bool=True,
        user_matrix: np.ndarray[Any, Any]=np.eye(4),
        component: Optional[int]=None,
        emissive: Optional[bool]=None,
        copy_mesh: bool=False,
        backface_params: Optional[dict[Any, Any]]=None,
        show_vertices: Optional[bool]=None,
        edge_opacity: Optional[float]=None,
        **kwargs: Optional[dict[Any, Any]],
    ) -> pyvista.plotting.actor.Actor:
        ...