import unittest
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(Path(__file__).parent.parent / "fileIO"))
sys.path.append(str(Path(__file__).parent.parent / "data_structure"))
sys.path.append(str(Path(__file__).parent.parent / "meshsize"))
sys.path.append(str(Path(__file__).parent.parent / "visualization"))
sys.path.append(str(Path(__file__).parent.parent / "adfront2"))
sys.path.append(str(Path(__file__).parent.parent / "optimize"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from read_cas import parse_fluent_msh
from vtk_io import parse_vtk_msh
from basic_elements import Unstructured_Grid, Triangle, Quadrilateral


class TestCASParser(unittest.TestCase):
    """CAS文件解析测试套件"""

    @classmethod
    def setUpClass(cls):
        test_dir = Path(__file__).parent
        cls.sample_file = test_dir / "test_files/naca0012-hybrid.cas"

        # 添加文件存在性检查
        if not cls.sample_file.exists():
            raise FileNotFoundError(f"测试文件未找到：{cls.sample_file}")

        cls.grid = parse_fluent_msh(str(cls.sample_file))

    def test_basic_data_structure(self):
        """验证基础数据结构完整性"""
        self.assertIn("nodes", self.grid)
        self.assertIn("faces", self.grid)
        self.assertIn("zones", self.grid)

        # 验证二维网格
        self.assertEqual(self.grid["dimensions"], 2, "应为二维网格")

    def test_wall_boundary(self):
        """验证壁面边界解析"""
        wall_zones = [
            z for z in self.grid["zones"].values() if z.get("bc_type") == "wall"
        ]
        self.assertGreater(len(wall_zones), 0, "未找到壁面边界")

        # 验证壁面节点连接
        for zone in wall_zones:
            self.assertTrue(
                all(len(face["nodes"]) == 2 for face in zone["data"]),
                "壁面应为线性单元",
            )

    def test_node_parsing(self):
        """验证节点坐标解析"""
        nodes = self.grid["nodes"]
        self.assertEqual(len(nodes), 4411, "节点数量异常")

        # 预期坐标列表（保留5位小数精度）
        expected_coords = [
            (-5.00000, 0.00000),
            (-4.92499, -0.90527),
            (-4.70199, -1.78585),
            (-4.33711, -2.61771),
            (-3.84027, -3.37817),
            (-3.22505, -4.04648),
            (-2.50821, -4.60442),
            (-1.70932, -5.03675),
            (-0.85017, -5.33170),
            (0.04581, -5.48121),
        ]

        # 验证前10个节点坐标
        for i, (node, (exp_x, exp_y)) in enumerate(zip(nodes[:10], expected_coords)):
            # 格式验证
            self.assertEqual(len(node), 2, f"第{i+1}个节点维度异常")
            self.assertIsInstance(node[0], float, f"第{i+1}个节点X坐标类型错误")
            self.assertIsInstance(node[1], float, f"第{i+1}个节点Y坐标类型错误")

            # 数值精度验证（保留5位小数）
            self.assertAlmostEqual(
                node[0], exp_x, places=5, msg=f"第{i+1}个节点X坐标误差过大"
            )
            self.assertAlmostEqual(
                node[1], exp_y, places=5, msg=f"第{i+1}个节点Y坐标误差过大"
            )

    def test_faces_parsing(self):
        """验证面数据解析"""
        faces = self.grid["faces"]
        self.assertEqual(len(faces), 10059, "面数量异常")

        # 预期前5个面的数据（十六进制转十进制）
        expected_faces = [
            {"nodes": [0x35E, 0x2ED], "left": 2, "right": 3},
            {"nodes": [0x538, 0x50F], "left": 6, "right": 7},
            {"nodes": [0x527, 0x50F], "left": 7, "right": 8},
            {"nodes": [0x526, 0x50F], "left": 8, "right": 9},
            {"nodes": [0x576, 0x53A], "left": 0xC, "right": 0xD},
        ]

        # 验证前5个面的数据
        for i, (actual, expected) in enumerate(zip(faces[:5], expected_faces)):
            self.assertEqual(len(actual["nodes"]), 2, f"第{i+1}个面应为线性单元")
            self.assertEqual(
                actual["nodes"], expected["nodes"], f"第{i+1}个面节点连接错误"
            )
            self.assertEqual(
                actual["left_cell"], expected["left"], f"第{i+1}个面左侧单元错误"
            )
            self.assertEqual(
                actual["right_cell"], expected["right"], f"第{i+1}个面右侧单元错误"
            )
            self.assertGreaterEqual(
                actual["left_cell"], 0, f"第{i+1}个面左侧单元索引异常"
            )
            self.assertGreaterEqual(
                actual["right_cell"], 0, f"第{i+1}个面右侧单元索引异常"
            )

    def test_cells_parsing(self):
        """验证单元数据解析"""
        cell_count = self.grid["cell_count"]
        self.assertEqual(cell_count, 5648, "单元数量异常")


class TestVTKParser(unittest.TestCase):
    """VTK文件解析测试套件"""

    @classmethod
    def setUpClass(cls):
        test_dir = Path(__file__).parent
        cls.sample_file = test_dir / "test_files/naca0012.vtk"

        if not cls.sample_file.exists():
            raise FileNotFoundError(f"测试文件未找到：{cls.sample_file}")

        cls.unstr_grid = parse_vtk_msh(str(cls.sample_file))

    def test_ring_nodes(self):
        """验证环节点构建"""
        self.unstr_grid.cyclic_node2node()
        self.assertEqual(
            self.unstr_grid.node2node[2198],
            [2167, 2187, 1587, 1593, 1599, 2197, 2208, 2157],
        )
        self.assertEqual(
            self.unstr_grid.node2node[2206], [1597, 1592, 2205, 2182, 2161, 2183]
        )

    def test_node_parsing(self):
        """验证节点坐标解析"""
        node_coords = self.unstr_grid.node_coords
        self.assertEqual(len(node_coords), 2210, "节点数量异常")

        # 验证前5个节点坐标
        expected_coords = [
            (-4.33711323788159, 2.617711594749755),
            (-4.70200777925994, 1.785852780007827),
            (5.924987168714973, -0.9052702465440348),
            (6.0, 0.0),
            (-0.8501701792743945, -5.331701462666318),
        ]

        for i, node in enumerate(node_coords[:5]):
            self.assertAlmostEqual(node[0], expected_coords[i][0], places=5)
            self.assertAlmostEqual(node[1], expected_coords[i][1], places=5)

    def test_cell_parsing(self):
        """验证单元数据解析"""
        cells = self.unstr_grid.cell_container
        self.assertEqual(len(cells), 2930, "总单元数量异常")

        # 验证最后5个三角形单元
        expected_cells = [
            [2148, 2209, 2169],
            [2169, 2209, 2200],
            [2196, 2192, 2158],
            [2187, 1587, 2198],
            [2177, 2169, 2200],
        ]
        
        for i, cell in enumerate(cells[-5:]):
            self.assertIsInstance(cell, Triangle)  # 确保单元类型正确
            self.assertEqual(cell.node_ids, expected_cells[i])  # 验证节点索引

    def test_boundary_nodes(self):
        """验证边界节点解析"""
        boundary_nodes = self.unstr_grid.boundary_nodes
        self.assertEqual(len(boundary_nodes), 314, "边界节点数量异常")

    def test_hybrid_mesh(self):
        """验证混合网格解析"""
        tri_count = sum(
            1 for c in self.unstr_grid.cell_container if isinstance(c, Triangle)
        )
        quad_count = sum(
            1 for c in self.unstr_grid.cell_container if isinstance(c, Quadrilateral)
        )

        self.assertEqual(tri_count, 1576, "三角形单元数量错误")
        self.assertEqual(quad_count, 1354, "四边形单元单元数量错误")


if __name__ == "__main__":
    unittest.main()
