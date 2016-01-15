import os

import fiona
import numpy as np
import pytest as pytest
from numpy.core.multiarray import ndarray
from shapely import wkt
from shapely.geometry import Polygon, shape, MultiPolygon, mapping, Point

from pyugrid import DataSet, FlexibleMesh
from pyugrid.flexible_mesh import constants
from pyugrid.flexible_mesh.constants import PYUGRID_LINK_ATTRIBUTE_NAME
from pyugrid.flexible_mesh.geom_cabinet import GeomCabinetIterator
from pyugrid.flexible_mesh.helpers import convert_multipart_to_singlepart, get_face_variables, get_variables, \
    iter_records, GeometryManager, create_rtree_file, flexible_mesh_to_esmf_format, get_coordinate_dict_variables, \
    get_split_array
from pyugrid.flexible_mesh.mpi import MPI_RANK, MPI_SIZE, MPI_COMM
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

    @pytest.mark.mpi4py
    def test_flexible_mesh_to_esmf_format(self):
        records, _, name_uid = self.tdata_records_three
        gm = GeometryManager(name_uid, records=records)
        fm = FlexibleMesh.from_geometry_manager(gm, use_ragged_arrays=True, with_connectivity=False)

        # Variables are not necessary for ESMF format.
        if MPI_RANK == 0:
            self.assertIsNone(fm.face_face_connectivity)
        else:
            self.assertIsNone(fm)

        if MPI_RANK == 0:
            path = self.get_temporary_file_path('out.nc')
            with self.nc_scope(path, 'w') as ds:
                flexible_mesh_to_esmf_format(fm, ds, polygon_break_value=-80)
            with self.nc_scope(path) as ds:
                res = ds.variables['numElementConn'][:]
                self.assertEqual(res.tolist(), [4, 3, 6])
                element_conn = ds.variables['elementConn']
                self.assertEqual(element_conn.polygon_break_value, -80)
                res = element_conn[:]
                res = [e.tolist() for e in res.flat]
                actual = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11, 12]]
                self.assertEqual(res, actual)

                for v in ds.variables.values():
                    self.assertTrue(v[:].flatten().shape[0] > 1)

        MPI_COMM.Barrier()

    def test_get_coordinate_dict_variables(self):
        gm = GeometryManager('UGID', path=self.tdata_shapefile_path_state_boundaries, allow_multipart=True)
        result = get_face_variables(gm)
        _, _, _, _, cdict, n_coords = result

        polygon_break_value = -99
        face_nodes, coordinates, edge_nodes = get_coordinate_dict_variables(cdict, n_coords,
                                                                            polygon_break_value=polygon_break_value)
        for f in face_nodes:
            self.assertEqual(f.dtype, np.int32)

        self.assertEqual(coordinates.shape, (n_coords, 2))
        self.assertEqual(face_nodes.shape, (len(cdict),))
        self.assertEqual(edge_nodes.shape, coordinates.shape)
        self.assertNotIn(polygon_break_value, edge_nodes)
        self.assertEqual(np.unique(edge_nodes[:, 0]).shape[0], edge_nodes.shape[0])
        self.assertEqual(np.unique(edge_nodes[:, 1]).shape[0], edge_nodes.shape[0])
        self.assertEqual(face_nodes[-1][-1], n_coords - 1)

    @pytest.mark.mpi4py
    def test_get_face_variables(self):
        gm = GeometryManager('SPECIAL', records=self.tdata_records_three[0])
        result = get_face_variables(gm)
        if MPI_RANK == 0:
            face_links, nmax_face_nodes, face_ids, face_coordinates, cdict, n_coords = result
            self.assertEqual(face_coordinates.shape, (3, 2))
            self.assertEqual(face_ids.tolist(), [100, 101, 102])
            self.assertEqual(nmax_face_nodes, 6)
            to_test = [f.tolist() for f in face_links.flat]
            self.assertEqual(to_test, [[1], [0, 2], [1]])

            self.assertEqual(n_coords, 13)
            self.assertEqual(len(cdict), 3)
            for v in cdict.itervalues():
                self.assertGreater(len(v), 0)
                self.assertIsInstance(v[0], ndarray)

        else:
            self.assertIsNone(result)

        MPI_COMM.Barrier()

    def test_get_face_variables_single_and_disjoint(self):
        """Test with single and disjoint polygons."""

        for records, schema, name_uid in [self.tdata_records_disjoint, self.tdata_records_single]:
            gm = GeometryManager(name_uid, records=records)
            result = get_face_variables(gm)
            face_links, nmax_face_nodes, face_ids, face_coordinates, cdict, n_coords = result
            for f in face_links:
                self.assertEqual(f.shape[0], 1)
                self.assertEqual(f[0], -1)

    @pytest.mark.mpi4py
    def test_get_variables(self):
        gm = GeometryManager('SPECIAL', records=self.tdata_records_three[0])
        shp_path_three_polygons = self.get_temporary_file_path('three.shp')
        records, schema, name_uid = self.tdata_records_three
        self.write_fiona(shp_path_three_polygons, records, schema)

        keywords = dict(use_ragged_arrays=[True, False])

        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            result = get_variables(gm, use_ragged_arrays=k.use_ragged_arrays)

            if MPI_RANK == 0:
                face_nodes, face_edges, edge_nodes, nodes, face_links, face_ids, face_coordinates = result

                # Test rectangular array returned when use_ragged_arrays=False.
                if not k.use_ragged_arrays:
                    self.assertEqual(face_links.shape, (3, face_nodes.shape[1]))

                # There are three faces/elements with an (x, y) for each.
                self.assertEqual(face_coordinates.shape, (3, 2))

                # There are three shared nodes in this example.
                actual = 13
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

        MPI_COMM.Barrier()

    def test_get_split_array(self):
        arr = np.array([1, 2, 3, -100, 4, 5, 6, 7, -100, 4, 5, -100, 6])
        splits = get_split_array(arr, -100)
        actual = [ii.tolist() for ii in splits]
        desired = [[1, 2, 3], [4, 5, 6, 7], [4, 5], [6]]
        self.assertEqual(actual, desired)

    def test_get_variables_allow_multipart(self):
        """Test allowing multipolygons."""

        records, schema, name_uid = self.tdata_records_three
        mp = MultiPolygon([r['geom'] for r in records])
        self.assertEqual(len(mp), 3)
        new_records = [{'geom': mp, 'properties': {name_uid: 1000}}]
        gm = GeometryManager(name_uid, records=new_records, allow_multipart=True)
        for e in gm.iter_records():
            self.assertEqual(len(e['geom']), 3)
        result = get_variables(gm, use_ragged_arrays=True)
        face_nodes, face_edges, edge_nodes, coordinates, face_links, face_ids, face_coordinates = result
        self.assertIsInstance(edge_nodes, ndarray)
        self.assertFalse((edge_nodes == -1).any())
        self.assertEqual(face_nodes[0].shape[0], 15)
        self.assertEqual((face_nodes[0] == constants.PYUGRID_POLYGON_BREAK_VALUE).sum(), 2)
        self.assertEqual((face_edges[0] == constants.PYUGRID_POLYGON_BREAK_VALUE).sum(), 2)
        self.assertEqual(face_nodes[0].shape, face_edges[0].shape)
        self.assertNumpyAll(face_nodes[0], face_edges[0])

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

        MPI_COMM.Barrier()

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

    def test_iter_records_multipart(self):
        """Test with a multipart break flag."""

        wkt1 = 'Polygon ((-0.78579234972677603 0.45573770491803278, -0.56721311475409841 0.50601092896174871, -0.45573770491803289 0.38142076502732236, -0.72896174863387997 0.15846994535519121, -0.8775956284153007 0.32896174863387984, -0.78579234972677603 0.45573770491803278))'
        wkt2 = 'Polygon ((-0.32240437158469959 0.13005464480874318, -0.18032786885245922 0.13224043715846989, -0.10382513661202197 0.01857923497267755, -0.20000000000000018 -0.12131147540983611, -0.37049180327868858 -0.08852459016393444, -0.40109289617486343 0.04262295081967216, -0.32240437158469959 0.13005464480874318))'
        multi = MultiPolygon([wkt.loads(w) for w in [wkt1, wkt2]])
        record = {'geom': multi, 'properties': {'UGID': 10}}
        gm = GeometryManager('UGID', records=[record], allow_multipart=True)
        fm = FlexibleMesh.from_geometry_manager(gm)
        node_x = fm.nodes[:, 0]
        node_y = fm.nodes[:, 1]
        itr = iter_records(fm.faces, node_x, node_y, shapely_only=True,
                           polygon_break_value=constants.PYUGRID_POLYGON_BREAK_VALUE)
        records = list(itr)
        geom = records[0]['geom']
        for part_actual, part_desired in zip(geom, multi):
            self.assertPolygonAlmostEqual(part_actual, part_desired)
        self.assertIsInstance(geom, MultiPolygon)


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
