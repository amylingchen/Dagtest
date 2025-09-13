import time

from svgpathtools import svg2paths
import fitz  # PyMuPDF
import svgwrite
import math


def pdf_to_svg(pdf_path, svg_path, pagenum=0):
    doc = fitz.open(pdf_path)
    page = doc[pagenum]

    rect = page.rect
    rotation = page.rotation
    original_width = rect.width
    original_height = rect.height

    # 确定页面宽高和旋转参数
    if rotation in [90, 270]:
        page_width, page_height = original_height, original_width
    else:
        page_width, page_height = original_width, original_height

    # 创建SVG画布
    dwg = svgwrite.Drawing(svg_path, size=(f"{page_width}pt", f"{page_height}pt"))

    # 坐标转换函数
    def transform_point(x, y):
        # 处理页面旋转
        if rotation == 90:
            x, y = y, original_width - x
        elif rotation == 180:
            x, y = original_width - x, original_height - y
        elif rotation == 270:
            x, y =  x,original_height - y

            

        # 翻转Y轴适配SVG坐标系
        return (x, page_height - y)

    # 提取并转换绘图指令
    for drawing in page.get_drawings():
        path_commands = []
        current_pos = None

        for item in drawing["items"]:
            # 处理直线
            if item[0] == "l":
                start = item[1]
                end = item[2]
                start_flipped = transform_point(start.x, start.y)
                end_flipped = transform_point(end.x, end.y)

                if current_pos != start_flipped:
                    path_commands.append(f"M {start_flipped[0]} {start_flipped[1]}")
                path_commands.append(f"L {end_flipped[0]} {end_flipped[1]}")
                current_pos = end_flipped

            # 处理贝塞尔曲线
            elif item[0] == "c":
                cp1 = item[1]
                cp2 = item[2]
                end_point = item[3]
                cp1_flipped = transform_point(cp1.x, cp1.y)
                cp2_flipped = transform_point(cp2.x, cp2.y)
                end_flipped = transform_point(end_point.x, end_point.y)

                if current_pos is None:
                    path_commands.append(f"M {cp1_flipped[0]} {cp1_flipped[1]}")
                path_commands.append(
                    f"C {cp1_flipped[0]} {cp1_flipped[1]} {cp2_flipped[0]} {cp2_flipped[1]} {end_flipped[0]} {end_flipped[1]}")
                current_pos = end_flipped

        # 添加路径到SVG
        if path_commands:
            dwg.add(dwg.path(d=" ".join(path_commands), fill="none", stroke="black"))

    dwg.save()
    print('Save successful!')


# 示例：将 PDF 第一页转换为 SVG
pdf_to_svg("pdf/Loop-0771.pdf", "pdf/output.svg",2)


paths, attributes = svg2paths("pdf/output.svg")
for elem in attributes:
    if "rect" in elem.get("id", ""):  # 假设矩形表示元件
        x = elem["x"]
        y = elem["y"]
        print(f"元件位置：({x}, {y})")