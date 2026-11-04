"""Strict final MD -> DOCX entry, prepared only. No model or training calls.
--check never constructs a document. --build requires real native final evidence,
fresh registers/content and a final MD, and still leaves render/root QA pending.
The old nine-page draft and its builder are never modified or called via main.
"""
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREP=HERE.parent
PROJECT=PREP.parent.parent
DEPENDENCIES=Path('/Users/lucas/.cache/codex-runtimes/codex-primary-runtime/dependencies')
PYTHON=DEPENDENCIES/'python/bin/python3'
NODE=DEPENDENCIES/'node/bin/node'
SKILL=Path('/Users/lucas/.codex/plugins/cache/openai-primary-runtime/documents/26.1026.10807/skills/documents')
MARKER=SKILL/'container_tools/mark_artifact_operation_started.mjs'
RENDERER=SKILL/'render_docx.py'
SOURCES={'style_builder':(PREP/'document_build/build_handoff_doc.py','16e0758f1a43ae0da7a3a0990c99d9935cecb21742a9bdc59b000910d2e89108'),
 'fontconfig':(PREP/'document_build/fonts.conf','de72bf264957531d8aabde517fe50e0e72c80fbc0a89b54b69590afc8151d14b'),
 'native_validator':(PREP/'validate_unidb_final_receipt.py','387c69e49244a744b3db8fd513498d1c4df68e90d753ea60189f9c1177335c00'),
 'package_gate':(PREP/'package_preparation/build_handoff_package.py','41ed23a8e75e88ffaf091e47ea2c6d5023e743e515cbef76389c036d9b557fd9')}
STALE=['本稿尚不能作为最终交付','最终评价和CPU验收仍在进行','以上最终接收项目尚未全部完成',
       '终点评价、CPU全状态和保存预测审计、实际best/latest身份、本地备份仍待最终收尾',
       '午夜UniDB实际best/latest备份需由本次最终验收单独补齐','正式交付时必须补入当前root的最终报告']
class Refused(ValueError):pass
def need(value,message):
 if not value:raise Refused(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bound(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def load_source(label):
 path,expected=SOURCES[label];need(path.is_file() and sha(path)==expected,'Reviewed source changed/missing: '+label)
 spec=importlib.util.spec_from_file_location('_final_doc_'+label,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def check_content(text,identities):
 opening='\n'.join(text.splitlines()[:5])
 need('准备稿' not in opening and not re.search(r'(?i)\bdraft\b',opening),'Final MD still starts with draft preparation label')
 need(not any(x in text for x in STALE),'Final MD still has this campaign pending/draft paragraphs')
 need(not re.search(r'(?m)^\s*(?:```|~~~|\$\$|\\\[)',text) and '![' not in text,'Unsupported fenced code/image/display math must be intentionally formatted before DOCX generation')
 need(not re.search(r'(?i)\b(?:TBD|TODO|INSERT_FINAL_RESULTS)\b',text),'Unfilled final-content placeholder')
 need(text.lstrip().startswith('# '),'Final MD must have a descriptive title')
 for u in {q['update'] for q in identities.values()}:need(re.search(r'(?<!\d)'+str(u)+r'(?!\d)',text),'Final MD does not identify actual selected update '+str(u))

def preflight(config):
 package=load_source('package_gate')
 path=package.project_path(config)
 need(path.suffix.lower()=='.json' and package.copy_policy(path) is None,'Final DOCX config must be small project JSON')
 cfg=package.jload(path);need(cfg.get('schema')=='final_handoff_document_inputs_v1' and cfg.get('status')=='READY_FOR_FINAL_DOCX','Final document config remains preparation-only')
 source=package.file_binding(cfg.get('source_markdown',{}));need(source.suffix=='.md' and 'draft' not in source.name.lower(),'Only explicitly bound final MD is accepted')
 _,refresh=package.bound_json(cfg.get('final_handoff_refresh',{}))
 evidence=package.file_binding(refresh.get('current_unidb_evidence',{}));registry=package.jload(evidence)
 validator=load_source('native_validator');native=validator.validate(PREP,registry)
 current=package.final_content_refresh(cfg,native['identities'])
 text=source.read_text();check_content(text,native['identities'])
 output=package.project_path(cfg.get('output_docx',''),must_exist=False)
 need(output.is_relative_to(HERE) and output.suffix=='.docx' and 'FINAL' in output.name and 'draft' not in output.name.lower(),'Final DOCX must use a new FINAL filename inside this independent directory')
 need(not output.exists(),'Final output already exists; refuse overwrite')
 dependencies={name:bound(p) for name,(p,expected) in SOURCES.items() if p.is_file() and sha(p)==expected}
 need(len(dependencies)==len(SOURCES),'A reviewed layout/runtime input changed')
 rows=[]
 for ref in package.jload(package.CAMPAIGN/'evidence/CPU_POSTPROCESS_ACCEPTANCE.json')['refs']:
  summary_path=package.local_from_remote(ref['summary']);summary=package.jload(summary_path)
  rows.append({'branch':ref['branch'],'role':ref['role'],'update':ref['update'],'checkpoint_sha256':ref['checkpoint_sha256'],
    'summary':bound(summary_path),'by_g':summary['by_g']})
 facts={'identities':native['identities'],'quality':rows,'effective_batch':96,'training_combinations':160000,
        'original_total_presentations':96*native['identities']['original:latest']['update'],
        'original_new_since_D0_presentations':96*(native['identities']['original:latest']['update']-45411),
        'improved_total_presentations':96*native['identities']['improved:latest']['update'],
        'original_total_equivalent_epochs':96*native['identities']['original:latest']['update']/160000,
        'original_since_D0_equivalent_epochs':96*(native['identities']['original:latest']['update']-45411)/160000,
        'improved_total_equivalent_epochs':96*native['identities']['improved:latest']['update']/160000,
        'population':'Five G strengths0.5/1/2/3/4, each1000=250sources*4xi, EMA native100-stepSDE; RAW monitor separate; no locked Test/external-real success implied'}
 return package,cfg,source,text,output,{'status':'READY_FOR_DOCX_AUTHORING_RENDER_NOT_YET_DONE','inputs':bound(path),'source_markdown':bound(source),
   'reviewed_dependencies':dependencies,'native_final_proof':native,'final_content_refresh':current,
   'verified_facts':facts,'output_docx':str(output),'network_calls':0,'model_calls':0,'optimizer_updates':0,'checkpoint_payload_reads':0}

def create_docx(layout,text,target):
 # Exact layout and parser used by the root-reviewed nine-page reference. Only
 # the footer/status differ. Final length is determined by content and rendering.
 doc=layout.Document();s=doc.sections[0]
 s.page_height=layout.Cm(29.7);s.page_width=layout.Cm(21);s.top_margin=layout.Cm(1.8);s.bottom_margin=layout.Cm(1.8);s.left_margin=layout.Cm(2.1);s.right_margin=layout.Cm(2.1);s.header_distance=layout.Cm(.8);s.footer_distance=layout.Cm(.8)
 for name,size in [('Normal',10.5),('Title',22),('Heading 1',15),('Heading 2',12),('Heading 3',11)]:
  st=doc.styles[name];st.font.name=layout.FONT;st.font.size=layout.Pt(size);st.font.color.rgb=layout.RGBColor(0,0,0);st._element.get_or_add_rPr().get_or_add_rFonts().set(layout.qn('w:eastAsia'),layout.FONT)
  fmt=st.paragraph_format;fmt.line_spacing=1.24;fmt.space_after=layout.Pt(6)
  if name.startswith('Heading'):fmt.space_before=layout.Pt(11);fmt.keep_with_next=True
 footer=s.footer.paragraphs[0];footer.alignment=layout.WD_ALIGN_PARAGRAPH.CENTER
 run=footer.add_run('湍流项目交接说明  ·  ');layout.setfont(run,8)
 field=layout.OxmlElement('w:fldSimple');field.set(layout.qn('w:instr'),'PAGE');footer._p.append(field)
 lines=text.splitlines();i=0
 while i<len(lines):
  line=lines[i].strip()
  if not line:i+=1;continue
  if line.startswith('|'):
   block=[]
   while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
   layout.maketable(doc,block);continue
  if line.startswith('# '):p=doc.add_paragraph(style='Title');layout.addtext(p,line[2:]);p.paragraph_format.space_after=layout.Pt(12)
  elif line.startswith('## '):p=doc.add_paragraph(style='Heading 1');layout.addtext(p,line[3:])
  elif line.startswith('### '):p=doc.add_paragraph(style='Heading 2');layout.addtext(p,line[4:])
  else:
   p=doc.add_paragraph()
   if line.startswith('- '):line='□ '+line[2:]
   layout.addtext(p,line)
  i+=1
 for element in [doc._element,doc.styles._element]:
  for border in list(element.iter(layout.qn('w:pBdr'))):border.getparent().remove(border)
  for fonts in element.iter(layout.qn('w:rFonts')):
   for key in list(fonts.attrib):
    if 'Theme' in key:del fonts.attrib[key]
 doc.core_properties.title='湍流图像恢复项目交接说明';doc.core_properties.subject='全项目方法 历史结果 资产与接手说明';doc.core_properties.author=''
 doc.save(target)
 return {'tables':len(doc.tables),'paragraphs':len(doc.paragraphs)}

def main():
 parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
 group.add_argument('--check',action='store_true');group.add_argument('--build',action='store_true');parser.add_argument('--inputs',type=Path,default=HERE/'FINAL_DOCUMENT_INPUTS.template.json');args=parser.parse_args()
 output=None;temp=None;marker_calls=0;published=False
 try:
  package,cfg,source,text,output,report=preflight(args.inputs)
  if args.check:
   (HERE/'FINAL_DOCUMENT_PREFLIGHT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
   print(json.dumps({'status':report['status'],'output_created':False,'report':str(HERE/'FINAL_DOCUMENT_PREFLIGHT.json')},ensure_ascii=False));return 0
  need(Path(sys.executable).resolve()==PYTHON.resolve(),'Actual authoring must use bundled Python from load_workspace_dependencies')
  need(NODE.is_file() and MARKER.is_file(),'Bundled Node/artifact marker missing')
  layout=load_source('style_builder')
  output.parent.mkdir(parents=True,exist_ok=True);package.project_path(output,must_exist=False)
  # Exactly one marker for the one forthcoming DOCX creation operation. Root
  # must not run a second manual marker around this wrapper.
  started=subprocess.run([str(NODE),str(MARKER),'--operation-kind','create','--expected-output-count','1','--output-format','docx'],cwd=HERE,capture_output=True,text=True)
  marker_calls=1
  need(started.returncode==0,'Artifact operation marker failed; DOCX not constructed')
  temp=output.parent/('.unaccepted_'+str(time.time_ns())+'.docx')
  stats=create_docx(layout,text,temp)
  need(sha(source)==report['source_markdown']['sha256'],'Final MD changed during creation; temp retained, final path not published')
  for item in report['reviewed_dependencies'].values():need(sha(item['path'])==item['sha256'],'Reviewed dependency changed during creation')
  need(not output.exists(),'Final output appeared during creation; refuse overwrite')
  # Exclusive publication: a file appearing after preflight is never replaced.
  os.link(temp,output);published=True;temp.unlink()
  report.update({'status':'FINAL_DOCX_CREATED_RENDER_AND_ROOT_QA_PENDING','docx':bound(output),'builder':bound(__file__),**stats,
   'artifact_operation_markers':1,'rendering_required':True,'root_all_pages_QA_required':True,'visual_QA_pass':False,'final_delivery_complete':False,
   'renderer':str(RENDERER),'FONTCONFIG_FILE':str(SOURCES['fontconfig'][0]),'bundled_python':str(PYTHON)})
  (output.parent/'FINAL_DOCX_BUILD.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':report['status'],'docx':str(output),'visual_QA_pass':False},ensure_ascii=False));return 0
 except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
  final_exists=bool(output and output.is_file());temp_exists=bool(temp and temp.is_file())
  print(json.dumps({'status':'FAILED_OUTPUT_PRESERVED' if final_exists or temp_exists else 'REFUSED_BEFORE_FINAL_DOCX',
    'reason':str(exc),'final_DOCX_generated':published and final_exists,'output_path_exists':final_exists,
    'unaccepted_temp_DOCX_preserved':str(temp) if temp_exists else None,
    'artifact_operation_markers':marker_calls,'render_or_QA_pass':False},ensure_ascii=False));return 2
if __name__=='__main__':raise SystemExit(main())
