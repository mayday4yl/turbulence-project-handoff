#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the offline handoff reader using only Python standard library."""
from pathlib import Path
import html, json, re
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/meeting_20261104'
CHAPTERS = [('CURRENT_VERSION','01 · 当前交接版本'),('SIMULATOR_GUIDE','02 · 模拟器原理'),('CODE_MAP','03 · 对应代码'),('NPU_RUNBOOK','04 · NPU 环境与运行'),('METHODS_AND_NEXT_STEP','05 · 当前方法与下一步接口'),('PAPERS','06 · 论文'),('PAPER_MATERIALS','附录 A · 论文与汇报素材'),('MEETING_CHECKLIST','附录 B · 接收清单'),('VALIDATION','附录 C · 本次验证'),('COMPLETENESS_REVIEW','附录 D · 齐备范围与待对接项'),('METHODS_HISTORY','附录 E · 历史方法细节')]
ids = {name+'.md': name.lower() for name,_ in CHAPTERS}
def inline(s):
 s = html.escape(s, quote=False)
 # Only these exact, author-controlled presentational tags are accepted.
 # Keep identifiers inside backticks protected from any notation conversion.
 for tag in ['sub','sup','i']:
  s=s.replace('&lt;'+tag+'&gt;','<'+tag+'>').replace('&lt;/'+tag+'&gt;','</'+tag+'>')
 tokens = []
 def code(m):
  tokens.append('<code>'+m.group(1)+'</code>'); return '\x00'+str(len(tokens)-1)+'\x00'
 s = re.sub(r'`([^`]+)`',code,s)
 def link(m):
  image,label,url=m.groups()
  if url.split('#',1)[0] in ids: url='#'+ids[url.split('#',1)[0]]
  url=html.escape(html.unescape(url),quote=True)
  return '<img loading="lazy" alt="'+label+'" src="'+url+'">' if image else '<a href="'+url+'">'+label+'</a>'
 s=re.sub(r'(!?)\[([^\]]+)\]\(([^)]+)\)',link,s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
 for i,t in enumerate(tokens): s=s.replace('\x00'+str(i)+'\x00',t)
 return s

def render(text):
 lines=text.splitlines(); out=[]; i=0; lst=None
 def endlist():
  nonlocal lst
  if lst: out.append('</'+lst+'>'); lst=None
 while i<len(lines):
  s=lines[i]
  if s.startswith('```'):
   endlist(); code=[]; i+=1
   while i<len(lines) and not lines[i].startswith('```'): code.append(lines[i]); i+=1
   out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>')
  elif s.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:|\-]+\|$',lines[i+1]):
   endlist(); heads=s.strip('|').split('|'); i+=2; rows=[]
   while i<len(lines) and lines[i].startswith('|'): rows.append(lines[i].strip('|').split('|')); i+=1
   out.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+inline(c.strip())+'</th>' for c in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(c.strip())+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'); i-=1
  elif re.match(r'^#{1,6} ',s):
   endlist(); m=re.match(r'^(#+) (.*)',s); level=min(len(m[1])+1,6); out.append(f'<h{level}>'+inline(m[2])+f'</h{level}>')
  elif re.match(r'^\s*(?:[-*]|\d+\.) ',s):
   typ='ol' if re.match(r'^\s*\d+\.',s) else 'ul'
   if lst!=typ: endlist(); out.append('<'+typ+'>'); lst=typ
   out.append('<li>'+inline(re.sub(r'^\s*(?:[-*]|\d+\.) ','',s))+'</li>')
  elif s.strip():
   endlist(); out.append(('<blockquote>'+inline(s[2:])+'</blockquote>') if s.startswith('> ') else '<p>'+inline(s)+'</p>')
  else: endlist()
  i+=1
 endlist(); return '\n'.join(out)

css='''
:root{--ink:#15324d;--muted:#526579;--teal:#087f80;--line:#d9e3e9;--paper:#fff;--soft:#f2f6f8;--orange:#a96819}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:80px}body{margin:0;background:#f5f7f9;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.75}header{background:#15324d;color:white;padding:56px max(24px,calc((100vw - 1120px)/2));}header p{max-width:850px;color:#d4e8ed}.eyebrow{font-size:13px;letter-spacing:2px}h1{font-size:clamp(28px,4vw,44px);line-height:1.3;margin:16px 0}h2{font-size:28px;line-height:1.4}h3{font-size:22px}h4{font-size:18px}main{max-width:1180px;margin:auto;padding:20px 24px 70px}nav{position:sticky;top:0;background:#ffffffed;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);z-index:3;display:flex;gap:22px;overflow-x:auto;padding:14px max(24px,calc((100vw - 1120px)/2));white-space:nowrap;font-size:14px}a{color:#076d85;text-underline-offset:3px}nav a{text-decoration:none}section{margin:28px 0;padding:30px;background:white;border:1px solid var(--line);border-radius:12px}p{margin:12px 0}img{max-width:100%;height:auto}figure{margin:22px 0}figcaption,.caption{font-size:13px;color:var(--muted)}.lead{font-size:19px}.notice{padding:15px 19px;border-left:4px solid var(--teal);background:#eef8f7}.status{display:flex;gap:10px;flex-wrap:wrap}.status span{font-size:13px;background:#e9f3f2;padding:5px 12px;border-radius:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pair figure{margin:10px 0}.sample{width:100%;max-width:420px;border:1px solid #d9e3e9;image-rendering:auto}.demo-controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:var(--soft);padding:18px;border-radius:8px}.demo-controls label{font-weight:600}.demo-controls input{flex:1;min-width:150px;accent-color:var(--teal)}button{border:1px solid #9ab5c2;background:white;color:var(--ink);padding:8px 14px;border-radius:6px;cursor:pointer;font:inherit;font-size:14px}button[aria-pressed=true]{background:var(--teal);border-color:var(--teal);color:white}button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #edac53;outline-offset:3px}output{font-variant-numeric:tabular-nums;font-weight:700}.metric{font-size:14px;color:var(--muted)}details{background:white;border:1px solid var(--line);border-radius:10px;margin:14px 0;scroll-margin-top:80px}summary{padding:20px 24px;cursor:pointer;font-size:20px;font-weight:650}.chapter{padding:0 28px 28px;overflow-wrap:anywhere}.chapter>h2:first-child{display:none}.table-scroll{overflow:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left}th{background:#eef4f6}th,td{padding:10px 12px;vertical-align:top;border-bottom:1px solid var(--line)}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em;background:#edf3f6;padding:2px 4px;border-radius:4px}pre{background:#122b40;color:#e4f0f4;padding:20px;border-radius:8px;overflow:auto;line-height:1.65;font-size:13px}pre code{background:none;padding:0;font-size:inherit;white-space:pre}blockquote{border-left:3px solid var(--teal);padding-left:18px;margin-left:0}.downloads{display:flex;flex-wrap:wrap;gap:16px}footer{color:var(--muted);font-size:13px;padding-top:28px}@media(max-width:650px){header{padding:32px 20px}main{padding:8px 12px 36px}section{padding:20px 16px}.pair{grid-template-columns:1fr}.chapter{padding:0 16px 20px}summary{padding:16px;font-size:18px}nav{padding:12px 16px;gap:18px}h2{font-size:24px}.demo-controls{padding:12px;gap:10px}th,td{min-width:110px}}@media print{body{background:white}header{padding:24px;color:#15324d;background:white}header p{color:#15324d}nav,.demo-controls,button{display:none}main{padding:0;max-width:none}section,details{border:0;padding:10px 0;break-inside:avoid}details .chapter{display:block;padding:0}summary{padding:10px 0}.table-scroll,pre{overflow:visible}pre{white-space:pre-wrap;color:#15324d;background:#eef4f6}pre code{white-space:pre-wrap}.pair{grid-template-columns:1fr 1fr}a{color:inherit}img{max-height:620px;object-fit:contain}}
'''
report=json.loads((DOC/'evidence/SIMULATOR_CPU.json').read_text())
parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>湍流图像恢复 · 接手图解与运行指南</title><style>'+css+'</style></head><body>']
parts.append('''<header><div class="eyebrow">RESEARCH HANDOFF · 2026.11.04</div><h1>湍流图像恢复<br>当前版本与接手运行</h1><p class="lead">供接手老师阅读的离线指南：聚焦可复用的模拟器、最新 UniDB 基线，以及在 de36 上继续运行的步骤。历史过程放在附录。</p><p>本页随仓库一起交付。现有实验已经收尾；会议提出的 physics + consistency 尚待接手方实现。</p></header><nav aria-label="页面导航"><a href="#overview">接手顺序</a><a href="#current_version">当前版本</a><a href="#simulator">模拟器图解</a><a href="#interactive">固定 ξ 示例</a><a href="#routes">当前方法</a><a href="#chapters">完整指南</a><a href="#npu_runbook">NPU 运行</a><a href="#papers">论文</a></nav><main>
<section id="overview"><div class="status"><span>单帧 RGB 恢复</span><span>现有模块与待接入方法</span><span>离线可读 · 无外部脚本</span></div><h2>接手从这里开始</h2><p>当前重点是两个已有模块：<strong>V5 物理模拟器</strong>用于构造退化路径；<strong>最新 UniDB D0</strong>用于运行恢复基线，示例从原路线 U67180 EMA 开始。二者都有已验证的运行入口，但尚未耦合成会议提出的新 consistency 方法。</p><ol><li>先看<a href="#current_version">当前版本、权重与状态</a>，再理解 <strong>F(C, ξ, g)</strong> 的两条退化支路。</li><li>沿<a href="#code_map">代码导航</a>定位实际实现，再看三条方法路线。</li><li>按<a href="#npu_runbook">运行指南</a>核对环境与资产，跑 CPU 示例，再使用分配到的 NPU。</li><li>由新负责人确定 consistency 方法、接口、资源和新实验目录。</li></ol><div class="downloads"><a href="START_HERE.md">Markdown 总入口</a><a href="../../handoff/documents/Turbulence_Project_Handoff_FINAL.pdf">原交接 PDF</a><a href="../../handoff/documents/Turbulence_Project_Handoff_FINAL.docx">原交接 Word</a><a href="#meeting_checklist">接收清单</a></div></section>
<section id="simulator"><h2>01 / 先模糊，再按位移采样</h2><p>清晰图 C 与固定随机源 ξ 共同输入退化生成器。改变 g，会改变相位系数、空间 PSF 和位移；这条路径不等于把最终退化图与 clean 做固定线性混合。</p><figure><a href="figures/SIMULATOR.png"><img src="figures/SIMULATOR.png" alt="V5 模拟器三支路：随机相位到P2S权重、随机倾斜到采样网格、清晰图到基核卷积，再经过Smooth和warp"></a><figcaption>依据 V5 冻结源码绘制。<a href="figures/SIMULATOR.drawio">下载可编辑 draw.io</a> · <a href="#simulator_guide">公式、张量形状与已知局限</a></figcaption></figure><div class="notice"><strong>三个对象不能混用：</strong>raw ξ 是高斯随机源；位移以像素为单位；RenderResult.flow 是归一化采样网格。冻结模拟器参数仍允许输入 C 与 ξ 的一阶反向。</div></section>
<section id="interactive"><h2>02 / 同一 ξ，沿强度路径看变化</h2><p>下面来自本次服务器 CPU 实际渲染：256 × 256 合成坡度与棋盘图，固定两块 raw ξ。滑块切换已保存结果，不在浏览器重新运行模拟器。</p><div class="demo-controls"><label for="strength">强度 g</label><input id="strength" type="range" min="0" max="5" step="1" value="5" aria-valuetext="g = 4"><output id="strength-value" for="strength">4</output><span id="time-value">恢复时间 t = 0</span></div><div class="pair"><figure><img class="sample" src="figures/clean.png" alt="输入清晰合成图"><figcaption>清晰输入 C；两块 raw ξ 固定不变。</figcaption></figure><figure><img id="strength-image" class="sample" src="figures/g4.png" alt="固定随机源，g等于4的模拟器输出"><figcaption id="strength-caption">完整退化 Y₄：Smooth blur → warp。</figcaption></figure></div><p class="metric" id="flow-metric"></p><p class="caption">显示 PNG 裁到 [0,1]；保存的 NPY 与运算不裁剪。这是机制示例，不是恢复效果或数据集排名。g=0 的前向端点检查不能代替时间导数、有限差分或高阶梯度验收。</p><h3>单独看 G4 的 blur 与 warp</h3><div class="demo-controls" role="group" aria-label="选择退化组成"><button id="blur-button" type="button" aria-pressed="true">仅 Smooth blur</button><button id="warp-button" type="button" aria-pressed="false">再加 warp</button></div><figure><img id="stage-image" class="sample" src="figures/g4_blur_only.png" alt="G4经过Smooth的模糊结果，未加warp"><figcaption id="stage-caption">blur-only：空间变化模糊，尚未按位移场采样。</figcaption></figure><details><summary>查看历史自然图与中间强度</summary><div class="chapter"><img src="figures/HISTORICAL_SIMULATOR_GATE.png" alt="2026年10月21日原Gate中平滑、边缘和自然图的固定强度比较"><p class="caption">原 2026-10-21 Gate 图，非本次新增恢复结果。原 Gate 中部分近零阈值和有限差分检查仍为 FAIL，后续研究接收未改写这些记录。</p></div></details></section>
<section id="routes"><h2>03 / 当前模块与接入关系</h2><figure><a href="figures/METHODS.png"><img src="figures/METHODS.png" alt="当前可复用的V5模拟器、最新UniDB基线与尚待接入的consistency方法"></a><figcaption>V5 模拟器与最新 UniDB 当前独立运行；橙色虚线部分是下一步待接入。<a href="figures/METHODS.drawio">可编辑方法图</a> · <a href="#methods_and_next_step">当前方法与新接口</a></figcaption></figure><h3>运行环境也有依赖链</h3><figure><a href="figures/NPU_DEPENDENCIES.png"><img src="figures/NPU_DEPENDENCIES.png" alt="V6虚拟环境通过pth依赖DiffUIR、ResShift和physical_bridge环境，并使用Ascend CANN"></a><figcaption>仅有一个 venv 文件夹不够。<a href="figures/NPU_DEPENDENCIES.drawio">可编辑环境图</a> · <a href="#npu_runbook">完整命令与排错</a></figcaption></figure></section><h2 id="chapters">完整接手指南</h2><p>点击章节展开；代码、命令、论文链接和接收清单均在下面。也可打印本页，打印时自动展开全部章节。</p>''')
for name,label in CHAPTERS:
 p=DOC/(name+'.md')
 if not p.exists(): continue
 if name=='PAPER_MATERIALS': parts.append('<h2 id="appendices">附录 · 按需查阅</h2><p>完整历史、旧方法公式、论文素材与验收证据保留在这里，首次接手无需逐项读完。</p>')
 parts.append('<details id="'+name.lower()+'"><summary>'+label+'</summary><article class="chapter">'+render(p.read_text())+'</article></details>')
parts.append('<footer>资料来源：2026-11-04 会议纪要与逐字记录、项目冻结源码与原验收材料、本次有界运行记录。原始 handoff 与 source_snapshot 保持字节身份；本页为新增导读。接收完成须由接手方核对访问、资产和运行结果。</footer></main><script>')
parts.append('const strengths=[0,0.5,1,2,3,4]; const measurements='+json.dumps(report['strengths'])+';')
parts.append('''
const slider=document.getElementById('strength');
function updateStrength(){const g=strengths[Number(slider.value)],m=measurements[String(g)],img=document.getElementById('strength-image');document.getElementById('strength-value').textContent=g;document.getElementById('time-value').textContent='恢复时间 t = '+(1-g/4);slider.setAttribute('aria-valuetext','g = '+g);img.src='figures/g'+g+'.png';img.alt='固定随机源，g等于'+g+'的模拟器输出';document.getElementById('strength-caption').textContent='完整退化 Y，g = '+g+'：Smooth blur → warp。';document.getElementById('flow-metric').textContent='实际像素位移 RMS：'+m.flow_rms_pixels.toFixed(4)+' px · 原始浮点范围：['+m.min.toFixed(4)+', '+m.max.toFixed(4)+']';}
slider.addEventListener('input',updateStrength);updateStrength();
function stage(warp){document.getElementById('stage-image').src='figures/'+(warp?'g4':'g4_blur_only')+'.png';document.getElementById('stage-image').alt=warp?'G4完整退化，包含warp':'G4经过Smooth的模糊结果，未加warp';document.getElementById('stage-caption').textContent=warp?'完整输出：在 blur-only 上按位移网格双线性采样。':'blur-only：空间变化模糊，尚未按位移场采样。';document.getElementById('blur-button').setAttribute('aria-pressed',String(!warp));document.getElementById('warp-button').setAttribute('aria-pressed',String(warp));}
document.getElementById('blur-button').addEventListener('click',()=>stage(false));document.getElementById('warp-button').addEventListener('click',()=>stage(true));
function reveal(){const target=document.getElementById(location.hash.slice(1));if(target&&target.tagName==='DETAILS'){target.open=true;setTimeout(()=>target.scrollIntoView(),0);}}
window.addEventListener('hashchange',reveal);reveal();let previous=[];window.addEventListener('beforeprint',()=>{previous=[...document.querySelectorAll('details')].map(d=>d.open);document.querySelectorAll('details').forEach(d=>d.open=true)});window.addEventListener('afterprint',()=>document.querySelectorAll('details').forEach((d,i)=>d.open=previous[i]));
</script></body></html>''')
(DOC/'visual_guide.html').write_text('\n'.join(parts))
print('Built', DOC/'visual_guide.html')
