import os
from collections import deque, OrderedDict
from copy import copy

import fiona
import numpy as np
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.polygon import orient

from pyugrid.flexible_mesh import constants
from pyugrid.flexible_mesh.constants import PYUGRID_LINK_ATTRIBUTE_NAME
from pyugrid.flexible_mesh.geom_cabinet import GeomCabinetIterator
from pyugrid.flexible_mesh.logging_ugrid import log, log_entry_exit
from pyugrid.flexible_mesh.mpi import MPI_RANK, create_slices, MPI_COMM, hgather, vgather, MPI_SIZE, dgather
from pyugrid.flexible_mesh.spatial_index import SpatialIndex


def convert_multipart_to_singlepart(path_in, path_out, new_uid_name=PYUGRID_LINK_ATTRIBUTE_NAME, start=0):
    """
    Convert a vector GIS file from multipart to singlepart geometries. The function copies all attributes and
    maintains the coordinate system.

    :param str path_in: Path to the input file containing multipart geometries.
    :param str path_out: Path to the output file.
    :param str new_uid_name: Use this name as the default for the new unique identifier.
    :param int start: Start value for the new unique identifier.
    """

    with fiona.open(path_in) as source:
        len_source = len(source)
        log.info('Records to convert to singlepart: {}'.format(len_source))
        source.meta['schema']['properties'][new_uid_name] = 'int'
        with fiona.open(path_out, mode='w', **source.meta) as sink:
            for ctr, record in enumerate(source, start=1):
                log.debug('processing: {} of {}'.format(ctr, len_source))
                geom = shape(record['geometry'])
                if isinstance(geom, BaseMultipartGeometry):
                    for element in geom:
                        record['properties'][new_uid_name] = start
                        record['geometry'] = mapping(element)
                        sink.write(record)
                        start += 1
                else:
                    record['properties'][new_uid_name] = start
                    sink.write(record)
                    start += 1

#tdk: remove
# @log_entry_exit
# def get_coordinates_dict(gm, n_faces=None):
#     n_faces = n_faces or len(gm)
#     cdict = OrderedDict()
#     n_coords = 0
#     for ctr, (fid, record) in enumerate(gm.iter_records(return_uid=True), start=1):
#         log.debug('extracting {} of {} (rank={})'.format(ctr, n_faces, MPI_RANK))
#         coordinates_list, n_coords = get_coordinates_list_and_update_n_coords(record, n_coords)
#         cdict[fid] = coordinates_list
#     return cdict, n_coords


def get_coordinates_list_and_update_n_coords(record, n_coords):
    geom = record['geom']
    if isinstance(geom, MultiPolygon):
        itr = iter(geom)
    else:
        itr = [geom]
    coordinates_list = []
    for element in itr:
        current_coordinates = np.array(element.exterior.coords)
        # Assert last coordinate is repeated for each polygon.
        assert current_coordinates[0].tolist() == current_coordinates[-1].tolist()
        # Remove this repeated coordinate.
        current_coordinates = current_coordinates[0:-1, :]
        coordinates_list.append(current_coordinates)
        n_coords += current_coordinates.shape[0]
    return coordinates_list, n_coords


@log_entry_exit
def get_coordinate_dict_variables(cdict, n_coords, polygon_break_value=None):
    polygon_break_value = polygon_break_value or constants.PYUGRID_POLYGON_BREAK_VALUE
    dtype_int = np.int32
    face_nodes = np.zeros(len(cdict), dtype=object)
    idx_start = 0
    for idx_face_nodes, coordinates_list in enumerate(cdict.itervalues()):
        if idx_face_nodes == 0:
            coordinates = np.zeros((n_coords, 2), dtype=coordinates_list[0].dtype)
            edge_nodes = np.zeros_like(coordinates, dtype=dtype_int)
        for ctr, coordinates_element in enumerate(coordinates_list):
            shape_coordinates_row = coordinates_element.shape[0]
            idx_stop = idx_start + shape_coordinates_row

            new_face_nodes = np.arange(idx_start, idx_stop, dtype=dtype_int)
            edge_nodes[idx_start: idx_stop, :] = get_edge_nodes(new_face_nodes)
            if ctr == 0:
                face_nodes_element = new_face_nodes
            else:
                face_nodes_element = np.hstack((face_nodes_element, polygon_break_value))
                face_nodes_element = np.hstack((face_nodes_element, new_face_nodes))
            coordinates[idx_start:idx_stop, :] = coordinates_element
            idx_start += shape_coordinates_row
        face_nodes[idx_face_nodes] = face_nodes_element

    return face_nodes, coordinates, edge_nodes


def get_edge_nodes(face_nodes):
    first = face_nodes.reshape(-1, 1)
    second = first + 1
    edge_nodes = np.hstack((first, second))
    edge_nodes[-1, 1] = edge_nodes[0, 0]
    return edge_nodes


@log_entry_exit
def get_variables(gm, use_ragged_arrays=False, with_connectivity=True):
    """
    :param gm: The geometry manager containing geometries to convert to mesh variables.
    :type gm: :class:`pyugrid.flexible_mesh.helpers.GeometryManager`
    :param pack: If ``True``, de-deduplicate shared coordinates.
    :type pack: bool
    :returns: A tuple of arrays with index locations corresponding to:

    ===== ================ =============================
    Index Name             Type
    ===== ================ =============================
    0     face_nodes       :class:`numpy.ma.MaskedArray`
    1     face_edges       :class:`numpy.ma.MaskedArray`
    2     edge_nodes       :class:`numpy.ndarray`
    3     node_x           :class:`numpy.ndarray`
    4     node_y           :class:`numpy.ndarray`
    5     face_links       :class:`numpy.ndarray`
    6     face_ids         :class:`numpy.ndarray`
    7     face_coordinates :class:`numpy.ndarray`
    ===== ================ =============================

    Information on individual variables may be found here: https://github.com/ugrid-conventions/ugrid-conventions/blob/9b6540405b940f0a9299af9dfb5e7c04b5074bf7/ugrid-conventions.md#2d-flexible-mesh-mixed-triangles-quadrilaterals-etc-topology

    :rtype: tuple (see table for array types)
    :raises: ValueError
    """
    #tdk: update doc
    if len(gm) < MPI_SIZE:
        raise ValueError('The number of geometries must be greater than or equal to the number of processes.')

    result = get_face_variables(gm, with_connectivity=with_connectivity)

    if MPI_RANK == 0:
        face_links, nmax_face_nodes, face_ids, face_coordinates, cdict, n_coords = result
    else:
        return

    face_nodes, coordinates, edge_nodes = get_coordinate_dict_variables(cdict, n_coords, polygon_break_value=constants.PYUGRID_POLYGON_BREAK_VALUE)
    face_edges = face_nodes
    face_ids = np.array(cdict.keys(), dtype=np.int32)

    if not use_ragged_arrays:
        log.debug('converting ragged arrays to rectangular arrays (rank={})'.format(MPI_RANK))
        new_arrays = []
        for a in (face_links, face_nodes, face_edges):
            new_arrays.append(get_rectangular_array_from_object_array(a, (a.shape[0], nmax_face_nodes)))
        face_links, face_nodes, face_edges = new_arrays

    return face_nodes, face_edges, edge_nodes, coordinates, face_links, face_ids, face_coordinates


def get_rectangular_array_from_object_array(target, shape):
    new_face_links = np.ma.array(np.zeros(shape, dtype=target[0].dtype), mask=True)
    for idx, f in enumerate(target):
        new_face_links[idx, 0:f.shape[0]] = f
    face_links = new_face_links
    assert(face_links.ndim == 2)
    return face_links


def iter_touching(si, gm, shapely_object):
    select_uid = list(si.iter_rtree_intersection(shapely_object))
    select_uid.sort()
    for uid_target, record_target in gm.iter_records(return_uid=True, select_uid=select_uid):
        if shapely_object.touches(record_target['geom']):
            yield uid_target


@log_entry_exit
def get_face_variables(gm, with_connectivity=True):

    n_face = len(gm)
    log.debug('number of faces is {} (rank={})'.format(n_face, MPI_RANK))

    if MPI_RANK == 0:
        sections = create_slices(n_face)
    else:
        sections = None

    section = MPI_COMM.scatter(sections, root=0)

    # Create a spatial index to find touching faces.
    if with_connectivity:
        log.debug('getting spatial index (rank={})'.format(MPI_RANK))
        si = gm.get_spatial_index()

    face_ids = np.zeros(section[1] - section[0], dtype=np.int32)
    assert face_ids.shape[0] > 0

    face_links = {}
    max_face_nodes = 0
    face_coordinates = deque()

    log.debug('section={0} (rank={1})'.format(section, MPI_RANK))

    len_section = section[1] - section[0]

    cdict = OrderedDict()
    n_coords = 0

    for ctr, (uid_source, record_source) in enumerate(gm.iter_records(return_uid=True, slice=section)):
        log.debug('processing {} of {} (rank={})'.format(ctr + 1, len_section, MPI_RANK))

        coordinates_list, n_coords = get_coordinates_list_and_update_n_coords(record_source, n_coords)
        cdict[uid_source] = coordinates_list

        face_ids[ctr] = uid_source
        ref_object = record_source['geom']

        # Get representative points for each polygon.
        face_coordinates.append(np.array(ref_object.representative_point()))

        # For polygon geometries the first coordinate is repeated at the end of the sequence. UGRID clients do not want
        # repeated coordinates (i.e. ESMF).
        try:
            ncoords = len(ref_object.exterior.coords) - 1
        except AttributeError:
            # Likely a multipolygon...
            ncoords = sum([len(e.exterior.coords) - 1 for e in ref_object])
            # A -1 flag will be placed between elements.
            ncoords += (len(ref_object) - 1)
        if ncoords > max_face_nodes:
            max_face_nodes = ncoords

        if with_connectivity:
            touching = deque()
            for uid_target in iter_touching(si, gm, ref_object):
                # If the objects only touch they are neighbors and may share nodes.
                touching.append(uid_target)
            # If nothing touches the faces, indicate this with a flag value.
            if len(touching) == 0:
                touching.append(-1)
            face_links[uid_source] = touching

    face_ids = MPI_COMM.gather(face_ids, root=0)
    max_face_nodes = MPI_COMM.gather(max_face_nodes, root=0)
    face_links = MPI_COMM.gather(face_links, root=0)
    face_coordinates = MPI_COMM.gather(np.array(face_coordinates), root=0)
    cdict = MPI_COMM.gather(cdict, root=0)
    n_coords = MPI_COMM.gather(n_coords, root=0)

    if MPI_RANK == 0:
        face_ids = hgather(face_ids)
        face_coordinates = vgather(face_coordinates)
        cdict = dgather(cdict)
        n_coords = sum(n_coords)

        max_face_nodes = max(max_face_nodes)

        if with_connectivity:
            face_links = get_mapped_face_links(face_ids, face_links)
        else:
            face_links = None

        return face_links, max_face_nodes, face_ids, face_coordinates, cdict, n_coords


def get_mapped_face_links(face_ids, face_links):
    """
    :param face_ids: Vector of unique, integer face identifiers.
    :type face_ids: :class:`numpy.ndarray`
    :param face_links: List of dictionaries mapping face unique identifiers to neighbor face unique identifiers.
    :type face_links: list
    :returns: A numpy object array with slots containing numpy integer vectors with values equal to neighbor indices.
    :rtype: :class:`numpy.ndarray`
    """

    face_links = dgather(face_links)
    new_face_links = np.zeros(len(face_links), dtype=object)
    for idx, e in enumerate(face_ids.flat):
        to_fill = np.zeros(len(face_links[e]), dtype=np.int32)
        for idx_f, f in enumerate(face_links[e]):
            # This flag indicates nothing touches the faces. Do not search for this value in the face identifiers.
            if f == -1:
                to_fill_value = f
            # Search for the index location of the face identifier.
            else:
                to_fill_value = np.where(face_ids == f)[0][0]
            to_fill[idx_f] = to_fill_value
        new_face_links[idx] = to_fill
    return new_face_links

# tdk: need to support multipolygons
def flexible_mesh_to_fiona(out_path, face_nodes, node_x, node_y, crs=None, driver='ESRI Shapefile',
                           indices_to_load=None, face_uid=None):

    if face_uid is None:
        properties = {}
    else:
        properties = {face_uid.name: 'int'}

    schema = {'geometry': 'Polygon', 'properties': properties}
    with fiona.open(out_path, 'w', driver=driver, crs=crs, schema=schema) as f:
        for feature in iter_records(face_nodes, node_x, node_y, indices_to_load=indices_to_load, datasets=[face_uid]):
            feature['properties'][face_uid.name] = int(feature['properties'][face_uid.name])
            f.write(feature)
    return out_path


def iter_records(face_nodes, node_x, node_y, indices_to_load=None, datasets=None, shapely_only=False):
    if indices_to_load is None:
        feature_indices = range(face_nodes.shape[0])
    else:
        feature_indices = indices_to_load

    for feature_idx in feature_indices:
        coordinates = deque()

        try:
            current_face_node = face_nodes[feature_idx, :]
        except IndexError:
            # Likely an object array.
            assert face_nodes.dtype == object
            current_face_node = face_nodes[feature_idx]

        try:
            nodes = current_face_node.compressed()
        except AttributeError:
            # Likely not a masked array.
            nodes = current_face_node.flatten()

        # Construct the geometry object by collecting node coordinates using indicies stored in "nodes".
        for node_idx in nodes:
            coordinates.append((node_x[node_idx], node_y[node_idx]))
        polygon = Polygon(coordinates)

        # Collect properties if datasets are passed.
        properties = OrderedDict()
        if datasets is not None:
            for ds in datasets:
                properties[ds.name] = ds.data[feature_idx]
        feature = {'id': feature_idx, 'properties': properties}

        # Add coordinates or shapely objects depending on parameters.
        if shapely_only:
            feature['geom'] = polygon
        else:
            feature['geometry'] = mapping(polygon)

        yield feature


def create_rtree_file(gm, path):
    """
    :param gm: Target geometries to index.
    :type gm: :class:`pyugrid.flexible_mesh.helpers.GeometryManager`
    :param path: Output path for the serialized spatial index. See http://toblerity.org/rtree/tutorial.html#serializing-your-index-to-a-file.
    """

    si = SpatialIndex(path=path)
    for uid, record in gm.iter_records(return_uid=True):
        si.add(uid, record['geom'])


class GeometryManager(object):
    """
    Provides iteration, validation, and other management routines for collecting vector geometries from record lists or
    flat files.
    """

    def __init__(self, name_uid, path=None, records=None, path_rtree=None, allow_multipart=False):
        if path_rtree is not None:
            assert os.path.exists(path_rtree + '.idx')

        self.path = path
        self.path_rtree = path_rtree
        self.name_uid = name_uid
        self.records = copy(records)
        self.allow_multipart = allow_multipart

        self._has_provided_records = False if records is None else True

    def __len__(self):
        if self.records is None:
            ret = len(GeomCabinetIterator(path=self.path))
        else:
            ret = len(self.records)
        return ret

    def get_spatial_index(self):
        si = SpatialIndex(path=self.path_rtree)
        # Only add new records to the index if we are working in-memory.
        if self.path_rtree is None:
            for uid, record in self.iter_records(return_uid=True):
                si.add(uid, record['geom'])
        return si

    def iter_records(self, return_uid=False, select_uid=None, slice=None):
        # Use records attached to the object or load records from source data.
        to_iter = self.records or self._get_records_(select_uid=select_uid, slice=slice)

        if self.records is not None and slice is not None:
            to_iter = to_iter[slice[0]:slice[1]]

        for ctr, record in enumerate(to_iter):
            if self._has_provided_records and 'geom' not in record:
                record['geom'] = shape(record['geometry'])
                # Only use the geometry objects from here. Maintaining the list of coordinates is superfluous.
                record.pop('geometry')
            self._validate_record_(record)

            # Counter-clockwise orientations required by clients such as ESMF Mesh regridding.
            record['geom'] = format_geometry(record['geom'])

            if return_uid:
                uid = record['properties'][self.name_uid]
                yld = (uid, record)
            else:
                yld = record
            yield yld

    def _get_records_(self, select_uid=None, slice=slice):
        gi = GeomCabinetIterator(path=self.path, uid=self.name_uid, select_uid=select_uid, slice=slice)
        return gi

    def _validate_record_(self, record):
        geom = record['geom']

        # This should happen before any buffering. The buffering check may result in a single polygon object.
        if not self.allow_multipart and isinstance(geom, BaseMultipartGeometry):
            msg = 'Only singlepart geometries allowed. Perhaps "ugrid.convert_multipart_to_singlepart" would be useful?'
            raise ValueError(msg)


def format_geometry(geom):
    if isinstance(geom, MultiPolygon):
        #tdk: test orientation of multipolgyon
        # Orient each element of a multi-geometry.
        new_element = []
        for idx, e in enumerate(geom):
            e = get_oriented_and_valid_geometry(e)
            new_element.append(e)
        new_element = geom.__class__(new_element)
    elif isinstance(geom, Polygon):
        new_element = get_oriented_and_valid_geometry(geom)
    else:
        raise NotImplementedError(type(geom))

    return new_element


def get_oriented_and_valid_geometry(geom):

    if not geom.exterior.is_ccw:
        geom = orient(geom)

    try:
        assert geom.is_valid
    except AssertionError:
        geom = geom.buffer(0)
        assert geom.is_valid
    return geom


@log_entry_exit
def flexible_mesh_to_esmf_format(fm, ds):
    """
    Convert to an ESMF format NetCDF files. Only supports ragged arrays.

    :param fm: Flexible mesh object to convert.
    :type fm: :class:`pyugrid.flexible_mesh.core.FlexibleMesh`
    :param ds: An open netCDF4 dataset object.
    :type ds: :class:`netCDF4.Dataset`
    """
    # tdk: doc

    # Dimensions #######################################################################################################

    node_count = ds.createDimension('nodeCount', fm.nodes.shape[0])
    element_count = ds.createDimension('elementCount', fm.faces.shape[0])
    coord_dim = ds.createDimension('coordDim', 2)
    element_conn_vltype = ds.createVLType(fm.faces[0].dtype, 'elementConnVLType')

    # Variables ########################################################################################################

    node_coords = ds.createVariable('nodeCoords', fm.nodes.dtype, (node_count.name, coord_dim.name))
    node_coords.units = 'degrees'
    node_coords[:] = fm.nodes

    element_conn = ds.createVariable('elementConn', element_conn_vltype, (element_count.name,))
    element_conn.long_name = 'Node indices that define the element connectivity.'
    element_conn[:] = fm.faces

    num_element_conn = ds.createVariable('numElementConn', np.int64, (element_count.name,))
    num_element_conn.long_name = 'Number of nodes per element.'
    num_element_conn[:] = [e.shape[0] for e in fm.faces.flat]

    center_coords = ds.createVariable('centerCoords', fm.face_coordinates.dtype, (element_count.name, coord_dim.name))
    center_coords.units = 'degrees'
    center_coords[:] = fm.face_coordinates

    # tdk: compute area required?
    # element_area = ds.createVariable('elementArea', fm.nodes.dtype, (element_count.name,))
    # element_area.units = 'degrees'
    # element_area.long_name = 'area weights'

    # tdk: element mask required?
    # element_mask = ds.createVariable('elementMask', np.int32, (element_count.name,))

    # Global Attributes ################################################################################################

    ds.gridType = 'unstructured'
    ds.version = '0.9'
    setattr(ds, coord_dim.name, "longitude latitude")

