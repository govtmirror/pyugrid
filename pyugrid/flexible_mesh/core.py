import numpy as np

from pyugrid import UGrid, DataSet
from pyugrid.flexible_mesh.mpi import MPI_RANK
from pyugrid.ugrid import IND_DT


class FlexibleMesh(UGrid):
    """
    Manages flexible mesh UGRID convention variables.

    See https://github.com/ugrid-conventions/ugrid-conventions/blob/v0.9.0/ugrid-conventions.md#2d-flexible-mesh-mixed-triangles-quadrilaterals-etc-topology.
    """
    check_array_order = False

    @UGrid.face_face_connectivity.setter
    def face_face_connectivity(self, face_face_connectivity):
        if face_face_connectivity is not None:
            face_face_connectivity = self._format_masked_or_object_array_(face_face_connectivity)
        self._face_face_connectivity = face_face_connectivity

    @UGrid.faces.setter
    def faces(self, faces_indexes):
        # tdk: update doc for ragged array
        # Flexible meshes have varying node counts. Faces with less nodes than the maximum are masked.
        if faces_indexes is not None:
            self._faces = self._format_masked_or_object_array_(faces_indexes)
        else:
            self._faces = None
            # Other things are no longer valid.
            self._face_face_connectivity = None
            self._face_edge_connectivity = None

    @UGrid.face_edge_connectivity.setter
    def face_edge_connectivity(self, face_edge_connectivity):
        # Add more checking?
        if face_edge_connectivity is not None:
            face_edge_connectivity = self._format_masked_or_object_array_(face_edge_connectivity)
        self._face_edge_connectivity = face_edge_connectivity

    def _format_masked_or_object_array_(self, faces_indexes):
        try:
            ret = np.ma.array(faces_indexes, dtype=IND_DT)
        except ValueError:
            # Likely an object array for ragged array usage.
            if faces_indexes.dtype == object:
                ret = faces_indexes
            else:
                raise
        return ret

    @classmethod
    def from_geometry_manager(cls, gm, mesh_name='mesh', pack=False, use_ragged_arrays=False):
        # tdk: doc
        return get_flexible_mesh(gm, mesh_name, pack, use_ragged_arrays)

    @classmethod
    def from_shapefile(cls, path, name_uid, mesh_name='mesh', path_rtree=None, pack=False, use_ragged_arrays=False):
        """
        Create a flexible mesh from a target shapefile.

        >>> path = '/input/target.shp'
        >>> name_uid = 'UID'
        >>> fm = FlexibleMesh.from_shapefile(path, name_uid)

        :param path: Path to the target shapefile.
        :type path: str
        :param name_uid: Name of the integer unique identifier in the target shapefile. This value will be maintained on
         the output mesh object.
        :type name_uid: str
        :param mesh_name: Name of the mesh catalog variable.
        :type mesh: str
        :param path_rtree: Path to a serialized spatial index object created using ``rtree``. Use :func:`pyugrid.flexible_mesh.helpers.create_rtree_file`
         to create a persistent ``rtree`` spatial index file.
        :type path_rtree: str
        :param pack: If ``True``, de-duplicate coordinate values by checking for shared nodes. This may dramatically
         increase processing times especially for large shapefiles. File size reductions will vary with input shapefile
         topology.
        :type pack: bool
        :rtype: :class:`pyugrid.flexible_mesh.core.FlexibleMesh`
        """
        # tdk: update doc
        from helpers import GeometryManager

        gm = GeometryManager(name_uid, path=path, path_rtree=path_rtree)
        ret = get_flexible_mesh(gm, mesh_name, pack, use_ragged_arrays)

        return ret

    def iter_records(self, shapely_only=False):
        """
        Yield record dictionaries containing face coordinates and properties (i.e. data set names and values).

        >>> fm = FlexibleMesh(...)
        >>> for record in fm.iter_records():
        >>>     print(record)
        >>> {'geometry': 'type': 'Polygon', 'coordinates': (...,), properties: {'a': 5, ...}}

        :param shapely_only: If ``True``, yield Shapely geometries instead of GeoJSON mappings for the face coordinates.
         In place of a ``'geometry'`` key, there is a ``'geom'`` key mapped to a Shapely geometry object.
        :type shapely_only: bool
        :rtype: dict
        """

        from helpers import iter_records

        for record in iter_records(self.faces, self.nodes[:, 0], self.nodes[:, 1], datasets=self.data.values(),
                                   shapely_only=shapely_only):
            yield record

    def save_as_shapefile(self, path, face_uid_name=None):
        """
        Save object as a shapefile.

        >>> path = '/out_location/my.shp'
        >>> fm = FlexibleMesh(...)
        >>> fm.save_as_shapefile(path, face_uid_name='UID_VAR')

        :param path: Path to the output shapefile.
        :type path: str
        :param face_uid_name: Name of the unique, integer identifier variable contained in :attr:`pyugrid.ugrid.UGrid.data`.
        :type face_uid_name: str
        """

        from helpers import flexible_mesh_to_fiona

        if face_uid_name is not None:
            face_uid = self.data[face_uid_name]
        else:
            face_uid = None

        flexible_mesh_to_fiona(path, self.faces, self.nodes[:, 0], self.nodes[:, 1], face_uid=face_uid)


def get_flexible_mesh(gm, mesh_name, pack, use_ragged_arrays):
    from helpers import get_variables

    result = get_variables(gm, pack=pack, use_ragged_arrays=use_ragged_arrays)
    if MPI_RANK == 0:
        face_nodes, face_edges, edge_nodes, nodes, face_links, face_ids, face_coordinates = result
        data_attrs = {'long_name': 'Face unique identifiers.'}
        # TODO (bekozi): necessary to use a dictionary here? key of dictionary is never used.
        data = {'': DataSet(gm.name_uid, location='face', data=face_ids, attributes=data_attrs)}
        # TODO (bekozi): add boundaries, boundary_coordinates, and edge_coordinates
        ret = FlexibleMesh(nodes=nodes, faces=face_nodes, edges=edge_nodes, boundaries=None,
                           face_face_connectivity=face_links, face_edge_connectivity=face_edges,
                           edge_coordinates=None, face_coordinates=face_coordinates, boundary_coordinates=None,
                           data=data, mesh_name=mesh_name)
    else:
        ret = None
    return ret
