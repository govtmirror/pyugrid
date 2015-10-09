import os

import numpy as np
import fiona
from numpy.core.multiarray import ndarray
import pytest as pytest
from shapely.geometry import Polygon, shape, MultiPolygon, mapping, Point

from pyugrid import DataSet, FlexibleMesh
from pyugrid.flexible_mesh.constants import PYUGRID_LINK_ATTRIBUTE_NAME
from pyugrid.flexible_mesh.geom_cabinet import GeomCabinetIterator
from pyugrid.flexible_mesh.helpers import convert_multipart_to_singlepart, get_face_variables, get_variables, \
    iter_records, GeometryManager, create_rtree_file, flexible_mesh_to_esmf_format
from pyugrid.flexible_mesh.mpi import MPI_RANK, MPI_SIZE
from pyugrid.flexible_mesh.spatial_index import SpatialIndex
from test.test_flexible_mesh.base import AbstractFlexibleMeshTest


class TestHelpers(AbstractFlexibleMeshTest):
    def test_convert_multipart_to_singlepart(self):
        # check there are multipolygons in the source shapefile
        fail = True
        with fiona.open(self.tdata_shapefile_path_state_boundaries) as source:
            for record in source:
                geom = shape(record['geometry'])
                if isinstance(geom, MultiPolygon):
                    fail = False
                    break
        if fail:
            raise AssertionError('they are polygons')

        path_out = self.get_temporary_file_path('singlepart.shp')
        convert_multipart_to_singlepart(self.tdata_shapefile_path_state_boundaries, path_out)

        with fiona.open(path_out) as source:
            self.assertNotEqual(source.crs, {})
            for record in source:
                self.assertIn(PYUGRID_LINK_ATTRIBUTE_NAME, record['properties'])
                geom = shape(record['geometry'])
                self.assertIsInstance(geom, Polygon)

    def test_flexible_mesh_to_esmf_format(self):
        records, _, name_uid = self.tdata_records_three
        gm = GeometryManager(name_uid, records=records)
        fm = FlexibleMesh.from_geometry_manager(gm, use_ragged_arrays=True)
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            flexible_mesh_to_esmf_format(fm, ds)
        with self.nc_scope(path) as ds:
            res = ds.variables['numElementConn'][:]
            self.assertEqual(res.tolist(), [4, 3, 6])
            res = ds.variables['elementConn'][:]
            res = [e.tolist() for e in res.flat]
            actual = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11, 12]]
            self.assertEqual(res, actual)

    @pytest.mark.mpi4py
    def test_get_face_variables(self):
        gm = GeometryManager('SPECIAL', records=self.tdata_records_three[0])
        result = get_face_variables(gm)
        if MPI_RANK == 0:
            face_links, nmax_face_nodes, face_ids, face_coordinates = result
            self.assertEqual(face_coordinates.shape, (3, 2))
            self.assertEqual(face_ids.tolist(), [100, 101, 102])
            self.assertEqual(nmax_face_nodes, 6)
            to_test = [f.tolist() for f in face_links.flat]
            self.assertEqual(to_test, [[1], [0, 2], [1]])
        else:
            self.assertIsNone(result)

    def test_get_face_variables_single_and_disjoint(self):
        """Test with single and disjoint polygons."""

        for records, schema, name_uid in [self.tdata_records_disjoint, self.tdata_records_single]:
            gm = GeometryManager(name_uid, records=records)
            result = get_face_variables(gm)
            face_links, nmax_face_nodes, face_ids, face_coordinates = result
            for f in face_links:
                self.assertEqual(f.shape[0], 1)
                self.assertEqual(f[0], -1)

    @pytest.mark.mpi4py
    def test_get_variables(self):
        gm = GeometryManager('SPECIAL', records=self.tdata_records_three[0])
        shp_path_three_polygons = self.get_temporary_file_path('three.shp')
        records, schema, name_uid = self.tdata_records_three
        self.write_fiona(shp_path_three_polygons, records, schema)

        keywords = dict(pack=[True, False], use_ragged_arrays=[True, False])

        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            result = get_variables(gm, pack=k.pack, use_ragged_arrays=k.use_ragged_arrays)

            if MPI_RANK == 0:
                face_nodes, face_edges, edge_nodes, nodes, face_links, face_ids, face_coordinates = result

                # Test rectangular array returned when use_ragged_arrays=False.
                if not k.use_ragged_arrays:
                    self.assertEqual(face_links.shape, (3, face_nodes.shape[1]))

                # There are three faces/elements with an (x, y) for each.
                self.assertEqual(face_coordinates.shape, (3, 2))

                # There are three shared nodes in this example.
                actual = 10 if k.pack else 13
                self.assertEqual(nodes.shape[0], actual)

                out_shp = self.get_temporary_file_path('reconstructed.shp')
                schema = {'geometry': 'Polygon', 'properties': {}}
                with fiona.open(out_shp, 'w', driver='ESRI Shapefile', schema=schema) as sink:
                    itr = GeomCabinetIterator(path=shp_path_three_polygons, uid='SPECIAL')
                    for idx, record in enumerate(itr):
                        uid = record['properties']['SPECIAL']
                        self.assertEqual(uid, face_ids[idx])
                        actual_geom = record['geom']

                        try:
                            node_indices = face_nodes[idx].compressed()
                        except AttributeError:
                            # Likely a ragged array.
                            self.assertTrue(k.use_ragged_arrays)
                            node_indices = face_nodes[idx]

                        # Last node is not repeated in UGRID data.
                        len_exterior_coords = len(actual_geom.exterior.coords)
                        self.assertEqual(len_exterior_coords - 1, len(node_indices))
                        xy = nodes[node_indices, :]
                        target_polygon = Polygon(xy)
                        to_write = {'geometry': mapping(target_polygon), 'properties': {}}
                        sink.write(to_write)

                        self.assertPolygonAlmostEqual(actual_geom, target_polygon)
            else:
                self.assertIsNone(result)

    def test_get_variables_multipart(self):
        """Test allowing multipolygons."""

        records, schema, name_uid = self.tdata_records_three
        mp = MultiPolygon([r['geom'] for r in records])
        self.assertEqual(len(mp), 3)
        new_records = [{'geom': mp, 'properties': {name_uid: 1000}}]
        gm = GeometryManager(name_uid, records=new_records, allow_multipart=True)
        for e in gm.iter_records():
            self.assertEqual(len(e['geom']), 3)
        result = get_variables(gm, use_ragged_arrays=True, pack=False)
        face_nodes, face_edges, edge_nodes, coordinates, face_links, face_ids, \
        face_coordinates = result
        self.assertIsInstance(edge_nodes, ndarray)
        self.assertFalse((edge_nodes == -1).any())
        self.assertEqual(face_nodes[0].shape[0], 15)
        self.assertEqual((face_nodes[0] == -1).sum(), 2)
        self.assertEqual((face_edges[0] == -1).sum(), 2)
        self.assertEqual(face_nodes[0].shape, face_edges[0].shape)
        self.assertNumpyAll(face_nodes[0], face_edges[0])
        # tdk: RESUME: add conversion back to multipolygon objects and test polygons are equal.

    @pytest.mark.mpi4py
    def test_get_variables_disjoint_and_single(self):
        """Test converting a shapefile that contains two disjoint elements and a single element."""

        for s, r in zip(['disjoint', 'single'], [self.tdata_records_disjoint[0], self.tdata_records_single[0]]):
            gm = GeometryManager('FID', records=r)

            try:
                result = get_variables(gm)
            except ValueError:
                self.assertLess(len(gm), MPI_SIZE)
                continue

            if MPI_RANK == 0:
                face_nodes, face_edges, edge_nodes, nodes, face_links, face_ids, face_coordinates = result

                self.assertEqual(face_links.shape, face_nodes.shape)

                for idx_f in range(face_links.shape[0]):
                    self.assertEqual(face_links[idx_f][0], -1)
                    self.assertTrue(face_links[idx_f].mask[1:].all())

                if s == 'disjoint':
                    shp = (2, 2)
                else:
                    shp = (1, 2)
                self.assertEqual(face_coordinates.shape, shp)
            else:
                self.assertIsNone(result)

    def test_iter_records(self):
        Mesh2_face_nodes = np.array([[0, 2, 3, 4], [0, 2, 3, 4]])
        Mesh2_node_x = np.array([-0.53064513, -1.25698924, 0.14301075, 1.19552982, 0.54253644])
        Mesh2_node_y = np.array([0.53817207, -0.73010755, -0.73763442, -0.34940514, 0.73571628])
        datasets = [DataSet('uid', location='face', data=[60, 61])]
        datasets.append(DataSet('summer', location='face', data=[67.9, 68.9]))

        itr = iter_records(Mesh2_face_nodes, Mesh2_node_x, Mesh2_node_y, datasets=datasets)
        records = list(itr)
        self.assertEqual(records[0]['properties']['uid'], 60)
        self.assertEqual(records[1]['properties']['uid'], 61)
        self.assertEqual(records[0]['properties']['summer'], 67.9)
        self.assertEqual(records[1]['properties']['summer'], 68.9)
        self.assertEqual(len(records), 2)


class TestGeometryManager(AbstractFlexibleMeshTest):

    def get(self, *args, **kwargs):
        return GeometryManager(*args, **kwargs)

    def get_archetype_shapefile_path(self):
        path = self.get_temporary_file_path('archetype.shp')
        records, schema, name_uid = self.tdata_records_three
        self.write_fiona(path, records, schema)
        return path

    def test_get_spatial_index(self):
        shp_path = self.get_archetype_shapefile_path()
        gm = GeometryManager('SPECIAL', path=shp_path)
        si_path = os.path.join(self.path_current_tmp, 'out.rtree')
        create_rtree_file(gm, si_path)
        gm = GeometryManager('SPECIAL', path=shp_path, path_rtree=si_path)
        si = gm.get_spatial_index()
        self.assertIsInstance(si, SpatialIndex)
        contents = os.listdir(self.path_current_tmp)
        self.assertIn('out.rtree.dat', contents)
        self.assertIn('out.rtree.idx', contents)

    def test_iter_records(self):
        # Test with a MultiPolygon.
        shp_path = self.get_archetype_shapefile_path()
        with fiona.open(shp_path) as source:
            polygons = [shape(record['geometry']) for record in source]
        multi = MultiPolygon(polygons)
        records = [{'geometry': mapping(multi), 'properties': {}}]
        gm = self.get('SPECIAL', records=records)
        with self.assertRaises(ValueError):
            list(gm.iter_records())

        # Test coordinate ordering is counter-clockwise.
        cw_coords = [[64.581791186564672, 51.542112406325586], [71.813555074983526, 46.95368290470811],
                     [64.581791186564672, 42.764247272796496], [59.344996646675163, 47.851419111546306]]
        polygon = Polygon(cw_coords)
        self.assertFalse(polygon.exterior.is_ccw)
        feature = {'geometry': mapping(polygon), 'properties': {'id': 0}}
        gm = self.get('id', records=[feature])
        updated_polygon = list(gm.iter_records())[0]['geom']
        self.assertTrue(updated_polygon.exterior.is_ccw)

    def test_len(self):
        r = [{'geom': Point(1, 2), 'properties': {'UGID': 1}}]
        gm = GeometryManager('UGID', records=r)
        self.assertEqual(len(gm), 1)
