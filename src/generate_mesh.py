import os
import gmsh
import numpy as onp
import jax.numpy as np
import meshio
import json
from src.basis import get_elements


class Mesh():
    """A custom mesh manager might be better than just use third-party packages like meshio?
    """
    def __init__(self, points, cells):
        # TODO: Assert that cells must have correct orders
        # TODO: first cells, then points?
        self.points = points
        self.cells = cells


def check_mesh_TET4(points, cells):
    # TODO
    def quality(pts):
        p1, p2, p3, p4 = pts
        v1 = p2 - p1
        v2 = p3 - p1
        v12 = np.cross(v1, v2)
        v3 = p4 - p1
        return np.dot(v12, v3)
    qlts = jax.vmap(quality)(points[cells])
    return qlts


def get_meshio_cell_type(ele_type):
    """Reference:
    https://github.com/nschloe/meshio/blob/9dc6b0b05c9606cad73ef11b8b7785dd9b9ea325/src/meshio/xdmf/common.py#L36
    """
    if ele_type == 'TET4':
        cell_type = 'tetra'
    elif ele_type == 'TET10':
        cell_type = 'tetra10'
    elif ele_type == 'HEX8':
        cell_type = 'hexahedron'
    elif ele_type == 'HEX27':
        cell_type = 'hexahedron27'
    elif  ele_type == 'HEX20':
        cell_type = 'hexahedron20'
    elif ele_type == 'TRI3':
        cell_type = 'triangle'
    elif ele_type == 'TRI6':
        cell_type = 'triangle6'
    else:
        raise NotImplementedError
    return cell_type


def box_mesh(Nx, Ny, Nz, Lx, Ly, Lz, data_dir, ele_type='HEX8'):
    """References:
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/examples/api/hex.py
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_1/tutorial/python/t1.py
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_1/tutorial/python/t3.py
    """

    assert ele_type != 'HEX20', f"gmsh cannot produce HEX20 mesh?"

    cell_type = get_meshio_cell_type(ele_type)
    _, _, _, _, degree, _ = get_elements(ele_type)

    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'box.msh')

    offset_x = 0.
    offset_y = 0.
    offset_z = 0.
    domain_x = Lx
    domain_y = Ly
    domain_z = Lz

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
    if cell_type.startswith('tetra'):
        Rec2d = False  # tris or quads
        Rec3d = False  # tets, prisms or hexas
    else:
        Rec2d = True
        Rec3d = True
    p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)
    l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
    s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
    v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(msh_file)
    gmsh.finalize()
      
    mesh = meshio.read(msh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict[cell_type] # (num_cells, num_nodes)
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})

    return out_mesh


def cylinder_mesh(data_dir, R=5, H=10, circle_mesh=5, hight_mesh=20, rect_ratio=0.4):
    """By Xinxin Wu at PKU in July, 2022
    Reference: https://www.researchgate.net/post/How_can_I_create_a_structured_mesh_using_a_transfinite_volume_in_gmsh
    R: radius
    H: hight
    circle_mesh:num of meshs in circle lines
    hight_mesh:num of meshs in hight
    rect_ratio: rect length/R
    """
    rect_coor = R*rect_ratio
    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    geo_file = os.path.join(msh_dir, 'cylinder.geo')
    msh_file = os.path.join(msh_dir, 'cylinder.msh')
    
    string='''
        Point(1) = {{0, 0, 0, 1.0}};
        Point(2) = {{-{rect_coor}, {rect_coor}, 0, 1.0}};
        Point(3) = {{{rect_coor}, {rect_coor}, 0, 1.0}};
        Point(4) = {{{rect_coor}, -{rect_coor}, 0, 1.0}};
        Point(5) = {{-{rect_coor}, -{rect_coor}, 0, 1.0}};
        Point(6) = {{{R}*Cos(3*Pi/4), {R}*Sin(3*Pi/4), 0, 1.0}};
        Point(7) = {{{R}*Cos(Pi/4), {R}*Sin(Pi/4), 0, 1.0}};
        Point(8) = {{{R}*Cos(-Pi/4), {R}*Sin(-Pi/4), 0, 1.0}};
        Point(9) = {{{R}*Cos(-3*Pi/4), {R}*Sin(-3*Pi/4), 0, 1.0}};

        Line(1) = {{2, 3}};
        Line(2) = {{3, 4}};
        Line(3) = {{4, 5}};
        Line(4) = {{5, 2}};
        Line(5) = {{2, 6}};
        Line(6) = {{3, 7}};
        Line(7) = {{4, 8}};
        Line(8) = {{5, 9}};

        Circle(9) = {{6, 1, 7}};
        Circle(10) = {{7, 1, 8}};
        Circle(11) = {{8, 1, 9}};
        Circle(12) = {{9, 1, 6}};

        Curve Loop(1) = {{1, 2, 3, 4}};
        Plane Surface(1) = {{1}};
        Curve Loop(2) = {{1, 6, -9, -5}};
        Plane Surface(2) = {{2}};
        Curve Loop(3) = {{2, 7, -10, -6}};
        Plane Surface(3) = {{3}};
        Curve Loop(4) = {{3, 8, -11, -7}};
        Plane Surface(4) = {{4}};
        Curve Loop(5) = {{4, 5, -12, -8}};
        Plane Surface(5) = {{5}};

        Transfinite Curve {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} = {circle_mesh} Using Progression 1;

        Transfinite Surface {{1}};
        Transfinite Surface {{2}};
        Transfinite Surface {{3}};
        Transfinite Surface {{4}};
        Transfinite Surface {{5}};
        Recombine Surface {{1, 2, 3, 4, 5}};

        Extrude {{0, 0, {H}}} {{
          Surface{{1:5}}; Layers {{{hight_mesh}}}; Recombine;
        }}

        Mesh 3;'''.format(R=R, H=H, rect_coor=rect_coor, circle_mesh=circle_mesh, hight_mesh=hight_mesh)

    with open(geo_file, "w") as f:
        f.write(string)
    os.system("gmsh -3 {geo_file} -o {msh_file} -format msh2".format(geo_file=geo_file, msh_file=msh_file))

    mesh = meshio.read(msh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict['hexahedron'] # (num_cells, num_nodes)

    # The mesh somehow has two redundant points...
    points = onp.vstack((points[1:14], points[15:]))
    cells = onp.where(cells > 14, cells - 2, cells - 1)

    out_mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
    return out_mesh

def non_uniform_mesh(node_turning, num_long_elem, num_short_elem, dim=1):
    """
    Mesh generator for non-uniform segments
    --- Inputs ---
    node_turning: 1D numpy array of known nodes
    num_long_elem: Number of elements in each long segment
    num_short_elem: Number of elements in each short segment
    dim: Problem dimension (default is 1 for 1D)
    
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    nelem: number of elements
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    
    node_turning = onp.sort(node_turning)  # Ensure the nodes are in increasing order
    num_segments = len(node_turning) - 1  # Number of segments
    nodes_per_elem = 2  # For 1D 2-node linear elements
    
    # Initialize lists to store the coordinates and connectivity
    XY_list = []
    Elem_nodes_list = []
    
    node_counter = 0
    
    for seg in range(num_segments):
        segment_start = node_turning[seg]
        segment_end = node_turning[seg + 1]
        segment_length = segment_end - segment_start
        
        # Determine the number of elements for this segment
        if seg % 2 == 0:  # Long segment
            nelem = num_long_elem
        else:  # Short segment
            nelem = num_short_elem
        
        # Calculate the spacing between nodes for this segment
        dx = segment_length / nelem
        
        for i in range(nelem + 1):
            x = segment_start + i * dx
            if i == 0 and seg > 0:
                continue  # Skip the first node if it's coincident with the previous segment end node
            
            XY_list.append([x])
            if i < nelem:
                Elem_nodes_list.append([node_counter, node_counter + 1])
            
            node_counter += 1
    
    # Convert lists to numpy arrays
    XY = onp.array(XY_list)
    Elem_nodes = onp.array(Elem_nodes_list, dtype=onp.int32)
    
    nnode = len(XY)
    nelem = len(Elem_nodes)
    dof_global = nnode * dim
    
    return XY, Elem_nodes, nelem, nnode, dof_global

def non_uniform_mesh_list(node_turning, num_long_elem, num_short_elem, dim=1):
    """
    Mesh generator for non-uniform segments
    --- Inputs ---
    node_turning: 1D numpy array of known nodes
    num_long_elem: Number of elements in each long segment
    num_short_elem: Number of elements in each short segment
    dim: Problem dimension (default is 1 for 1D)
    
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    nelem: number of elements
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    
    node_turning = onp.sort(node_turning)  # Ensure the nodes are in increasing order
    num_segments = len(node_turning) - 1  # Number of segments
    nodes_per_elem = 2  # For 1D 2-node linear elements
    
    # Initialize lists to store the coordinates and connectivity
    XY = []; Elem_nodes = []
    for seg in range(num_segments):
        XY_list = []
        Elem_nodes_list = []
        node_counter = 0
        segment_start = node_turning[seg]
        segment_end = node_turning[seg + 1]
        segment_length = segment_end - segment_start
        
        # Determine the number of elements for this segment
        if seg % 2 == 0:  # Long segment
            nelem = num_long_elem
        else:  # Short segment
            nelem = num_short_elem
        
        # Calculate the spacing between nodes for this segment
        dx = segment_length / nelem
        
        for i in range(nelem + 1):
            x = segment_start + i * dx
            if i == 0 and seg > 0:
                continue  # Skip the first node if it's coincident with the previous segment end node
            
            XY_list.append([x])
            if i < nelem:
                Elem_nodes_list.append([node_counter, node_counter + 1])
            
            node_counter += 1
        
        XY.append(onp.array(XY_list))
        Elem_nodes.append(onp.array(Elem_nodes_list, dtype=onp.int32))
    # Convert lists to numpy arrays
    
    return XY, Elem_nodes

def uniform_mesh_new(L, nelem_x):
    """ Mesh generator
    --- Inputs ---
    L: length of the domain
    nelem_x: number of elements in x-direction
    dim: problem dimension
    nodes_per_elem: number of nodes in one elements
    elem_type: element type
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    connectivity: elemental connectivity (nelem, node_per_elem*dim)
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    

    dim = 1
    nelem = nelem_x
    nnode = nelem+1 # number of nodes
    dof_global = nnode*dim    
    
    ## Nodes ##
    XY = onp.zeros([nnode, dim], dtype=onp.double)
    dx = L/nelem # increment in the x direction

    n = 0 # This will allow us to go through rows in NL
    for i in range(1, nelem+2):
        if i == 1 or i == nelem+1: # boundary nodes
            XY[n,0] = (i-1)*dx
        else: # inside nodes
            XY[n,0] = (i-1)*dx
        n += 1
        
    ## elements ##
    nodes_per_elem = 2
    Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
    for j in range(1, nelem+1):
        Elem_nodes[j-1, 0] = j-1
        Elem_nodes[j-1, 1] = j 
    
    # XY = np.array(XY)
    # Elem_nodes = np.array(Elem_nodes)               
    return XY, Elem_nodes

def gradient_mesh(L, nelem_x, r):
    """ Gradient Mesh generator with aspect ratio r
    --- Inputs ---
    L: length of the domain
    nelem_x: number of elements in x-direction
    r: aspect ratio between last and first element size (dx_last / dx_first)
    
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    """
    dim = 1
    nelem = nelem_x
    nnode = nelem + 1
    
    ## Nodes ##
    XY = onp.zeros([nnode, dim], dtype=onp.double)
    
    # Determine element sizes using geometric progression
    if r == 1.0:
        dx_array = onp.ones(nelem) * (L / nelem)  # uniform mesh
    else:
        # geometric progression ratio for spacing
        q = r ** (1/(nelem - 1))
        dx_first = L * (1 - q) / (1 - q ** nelem)
        dx_array = dx_first * q ** onp.arange(nelem)
    
    # Compute node positions by cumulative sum
    XY[:,0] = onp.concatenate(([0], onp.cumsum(dx_array)))
    
    ## Elements ##
    nodes_per_elem = 2
    Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
    for j in range(nelem):
        Elem_nodes[j, 0] = j
        Elem_nodes[j, 1] = j + 1
    
    return XY, Elem_nodes

def uniform_mesh(L, nelem_x, dim, nodes_per_elem, elem_type, non_uniform_mesh_bool=False):
    """ Mesh generator
    --- Inputs ---
    L: length of the domain
    nelem_x: number of elements in x-direction
    dim: problem dimension
    nodes_per_elem: number of nodes in one elements
    elem_type: element type
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    connectivity: elemental connectivity (nelem, node_per_elem*dim)
    nnode: number of nodes
    dof_global: global degrees of freedom
    """
    
    if elem_type == 'D1LN2N': # 1D 2-node linear element
        nelem = nelem_x
        nnode = nelem+1 # number of nodes
        dof_global = nnode*dim    
        
        ## Nodes ##
        XY = onp.zeros([nnode, dim], dtype=onp.double)
        dx = L/nelem # increment in the x direction
    
        n = 0 # This will allow us to go through rows in NL
        for i in range(1, nelem+2):
            if i == 1 or i == nelem+1: # boundary nodes
                XY[n,0] = (i-1)*dx
            else: # inside nodes
                XY[n,0] = (i-1)*dx
                if non_uniform_mesh_bool:
                     XY[n,0] += onp.random.normal(0,0.2,1)*dx# for x values
            n += 1
            
        ## elements ##
        Elem_nodes = onp.zeros([nelem, nodes_per_elem], dtype=onp.int32)
        for j in range(1, nelem+1):
            Elem_nodes[j-1, 0] = j-1
            Elem_nodes[j-1, 1] = j 
                   
    return XY, Elem_nodes, nelem, nnode, dof_global


def write_json_from_data(filename, vtk_filenames, times):
    """
    Writes a JSON file with the provided filenames and corresponding times.
    
    :param filename: str - The path of the JSON file to write to.
    :param vtk_filenames: list of str - List of filenames.
    :param times: list of float - List of times corresponding to the filenames.
    
    # Example usage:
    # vtk_files = ["foo1.vtk", "foo2.vtk", "foo3.vtk"]
    # times = [0, 5.5, 11.2]
    # write_json_from_data('output.json', vtk_files, times)
    """
    if len(vtk_filenames) != len(times):
        raise ValueError("The length of vtk_filenames must be equal to the length of times")

    # Create the list of file dictionaries
    files_list = [{"name": name, "time": time} for name, time in zip(vtk_filenames, times)]
    
    # Define the data structure for JSON
    data = {
        "file-series-version": "1.0",
        "files": files_list
    }
    
    # Write the data to the JSON file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)  # Pretty print with indentation
        
