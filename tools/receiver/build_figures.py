"""Build editable handoff diagrams. Rendering uses installed draw.io separately."""
from pathlib import Path
from xml.etree import ElementTree as E
from html import escape

DEST = Path(__file__).resolve().parents[2] / 'docs/meeting_20261104/figures'
PALETTE = {'data':'#F2F5F8', 'physics':'#E7F4F1', 'net':'#EAF1FB', 'future':'#FFF4E4'}


def label(value, rich=False):
    """Render deliberate mathematical markup; keep source paths literal."""
    return (value if rich else escape(value)).replace('\n', '<br>')


class Diagram:
    def __init__(self, name, title, subtitle, height=700):
        self.name=name
        self.xml=E.Element('mxfile', host='app.diagrams.net')
        page=E.SubElement(self.xml,'diagram',id=name,name=name)
        model=E.SubElement(page,'mxGraphModel',dx='1400',dy=str(height),grid='0',page='1',pageScale='1',pageWidth='1400',pageHeight=str(height),math='0',background='#FFFFFF',lineJumpsEnabled='1')
        self.root=E.SubElement(model,'root')
        E.SubElement(self.root,'mxCell',id='0')
        E.SubElement(self.root,'mxCell',id='1',parent='0')
        self.text('title',title,36,25,1328,45,30,True)
        self.text('subtitle',subtitle,36,80,1328,40,19)

    def node(self,id,value,x,y,w,h,kind='physics',size=20,dashed=False,rich=False):
        style=f'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={PALETTE[kind]};strokeColor=#60788A;strokeWidth=1.5;fontColor=#17324D;fontFamily=Arial;fontSize={size};spacing=12;align=center;verticalAlign=middle;'
        if dashed:style+='dashed=1;dashPattern=6 4;'
        cell=E.SubElement(self.root,'mxCell',id=id,value=label(value,rich),style=style,vertex='1',parent='1')
        E.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})

    def text(self,id,value,x,y,w,h,size=18,bold=False,rich=False):
        cell=E.SubElement(self.root,'mxCell',id=id,value=label(value,rich),style=f'text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;fontColor=#17324D;fontFamily=Arial;fontSize={size};fontStyle={1 if bold else 0};',vertex='1',parent='1')
        E.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})

    def edge(self,id,source,target,points=(),style=''):
        cell=E.SubElement(self.root,'mxCell',id=id,value='',source=source,target=target,style='edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;endFill=1;strokeColor=#60788A;strokeWidth=2;jumpStyle=arc;jumpSize=12;'+style,edge='1',parent='1')
        geom=E.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
        if points:
            arr=E.SubElement(geom,'Array',attrib={'as':'points'})
            for x,y in points:E.SubElement(arr,'mxPoint',x=str(x),y=str(y))

    def save(self):
        E.indent(self.xml)
        E.ElementTree(self.xml).write(DEST/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)


d=Diagram('SIMULATOR','V5 模拟器：先形成空间变化模糊，再按位移场采样','固定清晰图 C 与同一组 raw ξ，改变 g 才是在沿同一条物理路径移动。',760)
d.node('rawz','原始随机源 <i>ξ</i><sub>z</sub>\n[B, 36, 512, 512]',40,150,250,90,'data',rich=True)
d.node('zfield','相关混合 + FFT 滤波\n幅度 <i>g</i><sup>5/6</sup>',355,150,270,90,rich=True)
d.node('p2s','冻结 P2S：36 → 100\n每像素 PSF 基权重',690,150,270,90,'net')
d.node('rawt','原始随机源 <i>ξ</i><sub>tilt</sub>\n[B, 512, 512, 2]',40,300,250,90,'data',rich=True)
d.node('flow','<i>g</i><sup>5/6</sup> 缩放 + 完整复 IFFT\n取实部 → 像素位移 <i>u</i><sub>g</sub>',355,300,270,90,size=18,rich=True)
d.node('grid','像素中心采样网格\n2(<i>p</i> + <i>u</i><sub>g</sub> + 0.5)/256 − 1',690,300,330,90,rich=True)
d.node('clean','清晰图 C\n[B,3,256,256]',40,480,250,100,'data')
d.node('conv','平均核 + 100 基核卷积\nreflect 边界',355,480,270,100)
d.node('smooth','逐像素加权 + Smooth\n(1−<i>g</i>/4)<i>C</i> + (<i>g</i>/4)<i>B</i><sub>g</sub><sup>phys</sup>(<i>C</i>)',690,480,330,100,size=19,rich=True)
d.node('warp','双线性 warp\n边界：border\n输出 <i>Y</i><sub>g</sub>',1100,480,250,100,rich=True)
d.edge('e1','rawz','zfield'); d.edge('e2','zfield','p2s')
d.edge('e3','rawt','flow'); d.edge('e4','flow','grid')
d.edge('e5','clean','conv'); d.edge('e6','conv','smooth');d.edge('e7','smooth','warp')
d.edge('e8','p2s','smooth',[(1055,195),(1055,450),(855,450)],'exitX=1;exitY=0.5;entryX=0.5;entryY=0;')
d.edge('e9','grid','warp',[(1225,345)],'exitX=1;exitY=0.5;entryX=0.5;entryY=0;')
d.edge('e10','clean','smooth',[(165,625),(855,625)],'exitX=0.5;exitY=1;entryX=0.5;entryY=1;')
d.text('note','恢复时间 t = 1 − g/4：t=0 对应 g=4；t=1 对应 g=0。原图直连 Smooth；权重与位移分别控制两条支路。',40,660,1310,38,19)
d.text('boundary','g=0 前向 identity 有验证；近零误差与部分差分仍有历史 FAIL。图中 raw ξ、像素位移和归一化 grid 是不同对象。',40,705,1310,32,18)
d.save()

d=Diagram('METHODS_HISTORY','三条路线必须分开理解','D0是各自版本的直接恢复基线；扩散内部步数与外层物理递归次数不是同一个计数。',800)
d.text('v5','① 已完成：V5 / Concertormer 物理递归',40,135,1300,40,24,True)
d.node('obs','原观测 <i>Y</i><sub>4</sub>\n<i>Z</i><sub>0</sub> = <i>Y</i><sub>4</sub>',40,205,210,100,'data',rich=True)
d.node('network','<i>N</i><sub>θ</sub>(<i>Z</i><sub>i</sub>, <i>t</i><sub>i</sub>)\n预测 clean 与 raw <i>ξ</i>',330,205,270,100,'net',rich=True)
d.node('render','<i>F</i>(预测 clean, 预测 <i>ξ</i>, <i>g</i><sub>i+1</sub>)\n构造下一状态 <i>Z</i><sub>i+1</sub>',690,205,355,100,'physics',19,rich=True)
d.node('final','第4次 <i>N</i><sub>θ</sub> → clean 输出\n正式 refinement OFF',1120,205,240,100,'net',18,rich=True)
d.edge('a','obs','network');d.edge('b','network','render');d.edge('c','render','final',style='dashed=1;')
d.text('v5note','中间状态来自预测，训练保留 Full-BPTT。灰虚线省略重复步骤：前3次预测后重渲染，第4次直接返回 clean。',40,325,1320,42,19)
d.text('uni','② 已完成：UniDB D0 / 像素空间扩散桥',40,395,1300,40,24,True)
d.node('y','退化图 Y\n固定条件 μ = Y',40,455,250,90,'data')
d.node('sde','原生100步 reverse SDE\nConditionalUNet 预测噪声',390,455,440,90,'net')
d.node('pred','恢复图\n无外层 Simulator / raw-ξ head',930,455,430,90,'data')
d.edge('d','y','sde');d.edge('e','sde','pred')
d.text('future','③ 会议提出、尚待实施：物理路径 + consistency',40,595,1300,40,24,True)
d.node('path','复用 F(C, ξ, g(t))\n替换原线性插值路径',40,660,350,80,'future',19,True)
d.node('consistency','接手方确认原 consistency 方法\n边界、目标、采样器与梯度需求',485,660,440,80,'future',19,True)
d.node('compare','新实验：与匹配 D0 对比\n不是已取得的性能结论',1020,660,340,80,'future',19,True)
d.edge('f','path','consistency',style='dashed=1;strokeColor=#B77B2D;');d.edge('g','consistency','compare',style='dashed=1;strokeColor=#B77B2D;')
d.save()

d=Diagram('METHODS','当前交接：两个已有模块与一个待接方向','先分别理解和运行模拟器、UniDB；物理路径与新 consistency 的接入尚未完成。',800)
d.text('sim','① 可复用模块：V5 物理模拟器',40,135,1300,40,24,True)
d.node('clean_xi','清晰图 <i>C</i>、固定 <i>ξ</i>\n退化强度 <i>g</i>',40,205,300,100,'data',rich=True)
d.node('physical','<i>F</i>(<i>C</i>, <i>ξ</i>, <i>g</i>)\nSmooth 模糊 → 位移采样',430,205,460,100,'physics',rich=True)
d.node('degraded','退化图 <i>Y</i><sub>g</sub>\n固定 <i>ξ</i>，改变 <i>g</i> 得到物理路径',980,205,380,100,'data',19,rich=True)
d.edge('s1','clean_xi','physical');d.edge('s2','physical','degraded')
d.text('simnote','恢复时间 t = 1 − g/4；模拟器负责生成退化，不是恢复网络。',40,325,1320,42,19)
d.text('uni','② 最新恢复基线：UniDB D0 / U67180 EMA',40,395,1300,40,24,True)
d.node('y','退化图 Y\n固定条件 μ = Y',40,455,250,90,'data')
d.node('sde','原生100步 reverse SDE\nConditionalUNet 预测噪声',390,455,440,90,'net')
d.node('pred','恢复图\n当前无外层 Simulator / raw-ξ head',930,455,430,90,'data',19)
d.edge('u1','y','sde');d.edge('u2','sde','pred')
d.text('future','③ 待接入：物理路径 + 接手方 consistency 方法',40,595,1300,40,24,True)
d.node('path','复用 F(C, ξ, g(t))\n作为新方法的物理路径',40,660,350,80,'future',19,True)
d.node('consistency','确认接手方论文与代码\n时间、目标、采样器、梯度需求',485,660,440,80,'future',19,True)
d.node('compare','适配后开展新实验\n与匹配的 D0 对比',1020,660,340,80,'future',19,True)
d.edge('f','path','consistency',style='dashed=1;strokeColor=#B77B2D;');d.edge('g','consistency','compare',style='dashed=1;strokeColor=#B77B2D;')
d.save()

d=Diagram('NPU_DEPENDENCIES','NPU 环境：入口是一个 Python，依赖来自多个目录','2026-11-04 只读核对的 de36 环境关系；此图不是全封闭容器或当前卡号分配证明。',640)
d.node('cmd','新增接手示例\ntools/receiver/demo.py',40,165,300,100,'data')
d.node('python','V6 venv / Python 3.10\nv6_diffuir_l_20261027/env/bin/python',430,165,490,100,'net',20)
d.node('cann','Ascend CANN\nset_env.sh + torch_npu',1010,165,350,100,'physics')
d.edge('p1','cmd','python');d.edge('p2','python','cann')
d.node('stage','DiffUIR Stage1\n.../env/lib/python3.10/site-packages',40,380,400,100,'data',19)
d.node('resshift','ResShift Stage1\n.../env/lib/python3.10/site-packages',500,380,400,100,'data',19)
d.node('base','基础 physical_bridge 环境\n/data0/yanglin/envs/physical_bridge',960,380,400,100,'data',19)
d.edge('p3','python','stage',[(675,320),(240,320)],'exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
d.edge('p4','python','resshift',[(700,330)],'exitX=0.55;exitY=1;entryX=0.5;entryY=0;')
d.edge('p5','python','base',[(730,320),(1160,320)],'exitX=0.6;exitY=1;entryX=0.5;entryY=0;')
d.text('pth','三条下行箭头表示 V6 的 .pth 显式包含路径；torch 实际从基础环境加载，venv 目录不是完整依赖闭包。',40,520,1320,44,20)
d.text('rule','先确认解释器与模块路径，再分配物理卡；ASCEND_RT_VISIBLE_DEVICES=N 后程序中的 npu:0 指该可见卡。',40,580,1320,38,19)
d.save()
