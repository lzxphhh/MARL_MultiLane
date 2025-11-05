"""
XML生成器 - 根据配置文件生成multilane.rou.xml
"""

import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any

class RouteXMLGenerator:
    """路线XML文件生成器"""

    def __init__(self, config_file: str = "vehicle_config.json"):
        """
        初始化XML生成器

        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = None
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在")

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            raise Exception(f"加载配置文件失败: {e}")

        print(f"已加载配置文件: {self.config_file}")

    def generate_xml(self, output_file: str = "multilane.rou.xml"):
        """
        生成multilane.rou.xml文件

        Args:
            output_file: 输出XML文件路径
        """
        if not self.config:
            raise ValueError("配置文件未加载")

        # 创建根元素
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

        # 生成车辆类型定义
        self._generate_vehicle_types(root)

        # 生成路线定义
        self._generate_routes(root)

        # 格式化并保存XML
        self._save_xml(root, output_file)

        print(f"已成功生成 {output_file}")

    def _generate_vehicle_types(self, root: ET.Element):
        """生成车辆类型定义"""
        vehicle_types = self.config.get("vehicle_types", {})
        speed_ranges = self.config.get("speed_ranges", {})

        for base_type, type_config in vehicle_types.items():
            # 添加注释
            if "description" in type_config:
                comment = ET.Comment(f" {type_config['description']} ")
                root.append(comment)

            base_params = type_config.get("base_params", {})
            speed_variants = type_config.get("speed_variants", {})

            # ego类型车辆特殊处理：只生成一个统一的ego类型
            if base_type == "ego":
                vtype = ET.SubElement(root, "vType")
                vtype.set("id", "ego")

                # 使用medium等级的参数作为ego的统一参数
                ego_params = speed_variants.get("medium", speed_variants.get("low", {}))
                all_params = {**base_params, **ego_params}

                # 设置所有参数
                for param_name, param_value in all_params.items():
                    vtype.set(param_name, str(param_value))

                print(f"生成车辆类型: ego - {type_config.get('description', '')}")
                print(f"  统一参数: tau={ego_params.get('tau')}, minGap={ego_params.get('minGap')}")

            else:
                # 其他类型车辆按原逻辑为每个速度等级生成vType
                for speed_level, speed_params in speed_variants.items():
                    vtype = ET.SubElement(root, "vType")

                    # 生成类型ID
                    level_mapping = {"low": "_0", "medium": "_1", "high": "_2"}
                    suffix = level_mapping.get(speed_level, "_1")
                    type_id = f"{base_type}{suffix}"
                    vtype.set("id", type_id)

                    # 合并基础参数和速度相关参数
                    all_params = {**base_params, **speed_params}

                    # 设置所有参数
                    for param_name, param_value in all_params.items():
                        vtype.set(param_name, str(param_value))

                    # 打印生成的车辆类型信息
                    print(f"生成车辆类型: {type_id} - {type_config.get('description', '')}")
                    print(f"  速度等级: {speed_level} ({speed_ranges.get(speed_level, {}).get('min', 0)}-{speed_ranges.get(speed_level, {}).get('max', 20)} m/s)")
                    print(f"  参数: tau={speed_params.get('tau')}, minGap={speed_params.get('minGap')}")

    def _generate_routes(self, root: ET.Element):
        """生成路线定义"""
        routes = self.config.get("routes", {})

        for route_id, route_config in routes.items():
            route = ET.SubElement(root, "route")
            route.set("id", route_id)
            route.set("edges", route_config["edges"])

            print(f"生成路线: {route_id} - {route_config['edges']}")

    def _save_xml(self, root: ET.Element, output_file: str):
        """格式化并保存XML文件"""
        # 生成XML字符串
        rough_string = ET.tostring(root, 'unicode')

        # 使用minidom进行格式化
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ")

        # 移除空行和XML声明重复
        lines = pretty_xml.split('\n')
        # 过滤空行
        filtered_lines = [line for line in lines if line.strip()]

        # 重新构造XML内容
        final_lines = []
        for i, line in enumerate(filtered_lines):
            if i == 0:  # XML声明
                final_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
                final_lines.append('')  # 空行
            elif '<routes' in line:  # 根元素
                final_lines.append(line)
            elif '<!--' in line:  # 注释前加空行
                final_lines.append('')
                final_lines.append(line)
            else:
                final_lines.append(line)

        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(final_lines))

    def print_config_summary(self):
        """打印配置文件摘要信息"""
        if not self.config:
            print("配置文件未加载")
            return

        print("\n=== 配置文件摘要 ===")

        # 速度范围
        speed_ranges = self.config.get("speed_ranges", {})
        print(f"\n速度范围配置:")
        for level, range_info in speed_ranges.items():
            print(f"  {level}: {range_info['min']}-{range_info['max']} m/s")

        # 车辆类型
        vehicle_types = self.config.get("vehicle_types", {})
        print(f"\n车辆类型配置 ({len(vehicle_types)} 种基础类型):")
        for base_type, config in vehicle_types.items():
            print(f"  {base_type}: {config.get('description', '无描述')}")
            variants = config.get("speed_variants", {})
            print(f"    速度变体: {list(variants.keys())}")

        # 路线
        routes = self.config.get("routes", {})
        print(f"\n路线配置 ({len(routes)} 条路线):")
        for route_id, route_config in routes.items():
            print(f"  {route_id}: {route_config['edges']}")

        print("=" * 20)


def main():
    """主函数"""
    try:
        # 创建XML生成器
        generator = RouteXMLGenerator("vehicle_config.json")

        # 打印配置摘要
        generator.print_config_summary()

        # 生成XML文件
        print("\n开始生成XML文件...")
        generator.generate_xml("SUMO_files/multilane.rou.xml")

        print("\nXML生成完成！")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()