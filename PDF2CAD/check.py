import fitz  # PyMuPDF
import ezdxf

def check_pdf_is_vector(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]

    # 检查页面中的图像块数量
    image_count = len(page.get_images())

    # 检查矢量路径数量
    drawing_count = len(page.get_drawings())

    if image_count > 0:
        print(f"⚠️ 包含位图图像（可能是扫描件或混合PDF）")
    else:
        print(f"✅ 是矢量PDF（无嵌入位图）")

    print(f"图像块数量: {image_count}")
    print(f"矢量路径数量: {drawing_count}")

# 示例
# check_pdf_is_vector("data/Loop-0771.pdf")

# 加载 DXF 文件
doc = ezdxf.readfile("../../ElectricalCircuitDiagramIdentification/data/3.dxf")
msp = doc.modelspace()

# 遍历所有插入块（INSERT）
for insert in msp.query('INSERT'):
    block_name = insert.dxf.name
    position = insert.dxf.insert
    for attrib in insert.attribs:
        print(f"  属性名: {attrib.dxf.tag}, 值: {attrib.dxf.text}")

    # print(f"元器件: {block_name}, 位置: {position}")


