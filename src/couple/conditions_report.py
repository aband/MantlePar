#!/usr/bin/env python3
"""Create a PDF of saved initial/boundary conditions, without running a solver."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape

YEAR = 365*24*3600
PDF_NAME = 'conditions.pdf'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scales(config):
    p = {k: float(v) for k,v in config['phase']['parameters'].items()}
    length = math.sqrt(p['mus']*p['k0']/p['mul'])
    velocity = p['k0']*p['rhor']*p['g']/p['mul']
    return dict(length_m=length, velocity_m_s=velocity, time_s=length/velocity,
                temperature_K=p['Tm0']-p['Te0'], enthalpy_J_kg=p['cp']*(p['Tm0']-p['Te0']),
                pressure_Pa=p['rhor']*p['g']*length)


def renderer_python():
    """Use the active environment, an explicit PDF_PYTHON, or the bundled runtime."""
    if os.environ.get('PDF_PYTHON'):
        candidate = os.environ['PDF_PYTHON']
    elif importlib.util.find_spec('reportlab'):
        return sys.executable
    else:
        candidate = str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3')
    try:
        subprocess.run([candidate, '-c', 'import reportlab'], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('PDF output requires ReportLab. Install reportlab in your Python environment or set PDF_PYTHON to a Python executable with ReportLab.') from error
    return candidate


def load_recorded(directory):
    import numpy as np
    import yaml
    directory = Path(directory).resolve()
    final = json.loads((directory/'visualization.json').read_text())
    if final.get('status') != 'complete' or final.get('evolution_model') != 'full phase coupling':
        raise ValueError('Conditions PDF requires a completed full-evolution output directory')
    source = json.loads((directory/'source_state.json').read_text())
    preheat_path = directory/'preheat_input_used.yaml'
    archived = preheat_path.is_file()
    if not archived:
        preheat_path = Path(source['directory'])/'input_used.yaml'
    inputs = [preheat_path, directory/'input_used.yaml']
    configurations = [yaml.safe_load(path.read_text()) for path in inputs]
    if configurations[1]['mesh']['cells'] != [final['nx'], final['ny']]:
        raise ValueError('Saved input mesh and completed evolution output do not match')
    if configurations[1]['time']['end'] != final['time']:
        raise ValueError('Saved input end time and completed evolution output do not match')
    imported = []
    for field in ('H','C'):
        path = directory/f'starting_cell{field}.dat'
        values = np.loadtxt(path, ndmin=2)
        if values.shape != (final['ny'],final['nx']) or not np.isfinite(values).all():
            raise ValueError(f'Invalid imported {field} matrix for conditions report')
        imported.append(dict(field=field, file=str(path), sha256=digest(path),
                             minimum=float(values.min()), maximum=float(values.max()), count=values.size))
    return dict(schema_version=1, mode='Recorded completed run', configurations=configurations,
                inputs=[dict(path=str(path), sha256=digest(path)) for path in inputs],
                source=source, imported=imported, output_directory=str(directory),
                preheat_provenance=('Archived with evolution output' if archived else
                                   'Older run: preheat input read from the source directory; no archived copy was saved.'))


def preview_data(preheat, evolve, inputs):
    return dict(schema_version=1, mode='Configuration preview - no simulation run',
                configurations=[preheat,evolve], inputs=inputs, imported=[],
                source={'directory':preheat['output']['directory'], 'source_time':preheat['time']['end'],
                        'evolution_start':evolve['time']['start']},
                output_directory=evolve['output']['directory'],
                preheat_provenance='Planned settings. No computed H/C data are claimed.')


def generate(data, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    packet = output.with_suffix('.json')
    temporary = packet.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    try:
        subprocess.run([renderer_python(), str(Path(__file__).resolve()), '--render-json', str(temporary), '--output', str(output)], check=True)
        temporary.replace(packet)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def number(value):
    if isinstance(value, (list,tuple)):
        return '['+', '.join(number(v) for v in value)+']'
    if isinstance(value, (int,float)):
        return f'{value:.8g}'
    return str(value)


def profile_text(profile, scale=None, unit=None):
    name = profile.get('name', 'unspecified')
    p = profile.get('parameters', {})
    text = name+': '+ '; '.join(f'{key}={number(value)}' for key,value in p.items())
    if name == 'erf_ramp':
        text += '\nf=A*erf((x-x0)/w)/erf((xc-x0)/w) for x<xc; f=A for x>=xc.'
        if scale is not None:
            text += f'\nPeak physical speed: {number(float(p["amplitude"])*scale)} {unit}.'
    elif name == 'affine':
        text += '\nf=value + gradient dot ([x,y]-origin); omitted origin=[0,0].'
    elif name == 'piecewise_constant':
        text += '\nf=value_above for y>=interface_y; value_below otherwise.'
    if scale is not None and name == 'constant':
        value = p.get('value',0)
        physical = [float(v)*scale for v in value] if isinstance(value,list) else float(value)*scale
        text += f'\nPhysical value: {number(physical)} {unit}.'
    elif scale is not None and name != 'erf_ramp':
        text += f'\nPhysical value = f * {number(scale)} {unit}.'
    return text


def region_extent(config, definition):
    mesh = config['mesh']; side = definition['side']; selector = definition['selector']
    axis = 0 if side in ('top','bottom') else 1
    domain = mesh['domain']['x' if axis==0 else 'y']; n = mesh['cells'][axis]
    mode = selector['mode']
    if mode == 'whole_side':
        bounds = domain
    elif mode == 'physical_interval':
        bounds = selector['interval']
    elif mode == 'boundary_cells':
        step = (domain[1]-domain[0])/n
        bounds = [domain[0]+selector['start']*step, domain[0]+(selector['start']+selector['count'])*step]
    else:
        raise ValueError(f'Unsupported region selector: {mode}')
    scale = scales(config)['length_m']/1000
    return (f'{side}: {"x" if axis==0 else "y"} in {number(bounds)}\n'
            f'{number([v*scale for v in bounds])} km\n'
            f'{mode}'+(f'; start={selector["start"]}, count={selector["count"]}' if mode=='boundary_cells' else ''))


def make_pdf(data, output):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
    ink = colors.HexColor('#18334B'); teal = colors.HexColor('#147D88'); pale = colors.HexColor('#EDF4F6')
    body = ParagraphStyle('body', fontName='Helvetica', fontSize=9, leading=12, spaceAfter=6, textColor=ink)
    small = ParagraphStyle('small', parent=body, fontSize=8, leading=10.5, spaceAfter=0, splitLongWords=True)
    heading = ParagraphStyle('heading', parent=body, fontName='Helvetica-Bold', fontSize=19, leading=23, spaceAfter=14)
    sub = ParagraphStyle('sub', parent=body, fontName='Helvetica-Bold', fontSize=11, leading=14, spaceBefore=10, spaceAfter=7, textColor=teal, keepWithNext=True)
    def p(text, style=body):
        return Paragraph(escape(str(text)).replace('\n','<br/>'), style)
    story=[]
    def text(value): story.append(p(value))
    def title(value): story.append(p(value,heading))
    def section(value): story.append(p(value,sub))
    def table(headers, rows, widths):
        values = [[p(h,small) for h in headers]]+[[p(v,small) for v in row] for row in rows]
        t = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),pale), ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,0),(-1,0),.7,teal), ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#CCD7DC'))]))
        story.append(t); story.append(Spacer(1,7))
    configs=data['configurations']; labels=('Preheat','Full evolution')
    title('Initial and boundary conditions')
    text('MantlePar / two-stage corner case')
    table(['Report basis','Value'], [['Status',data['mode']],['Evolution output',data['output_directory']],
          ['Preheat provenance',data['preheat_provenance']]], [108,403])
    section('Stage timing and geometry')
    rows=[]
    for label,c in zip(labels,configs):
        s=scales(c); t=c['time']; mesh=c['mesh']
        rows.append([label, f'{mesh["cells"][0]} x {mesh["cells"][1]} / {mesh["family"]}\n'
                     f'x={number(mesh["domain"]["x"])}; y={number(mesh["domain"]["y"])}',
                     f't={number(t["start"])} to {number(t["end"])}\n'
                     f'{number((t["end"]-t["start"])*s["time_s"]/YEAR)} years duration'])
    table(['Stage','Mesh and nondimensional domain','Time'], rows, [77,250,184])
    section('Initial state and handoff')
    c=configs[0]; s=scales(c)
    for field,unit,scale in [('enthalpy','J/kg',s['enthalpy_J_kg']),('composition','fraction',1)]:
        text('Preheat '+field+': '+profile_text(c['transport']['initial_conditions'][field],scale,unit))
    text('Full evolution starts from the final preheat cell-average H and C, without interpolation. '
         'Its clock starts at '+number(data['source']['evolution_start'])+'. The analytic H/C profiles in '
         'the evolution YAML are initialization placeholders overwritten by the imported fields.')
    text('Source preheat time: '+number(data['source']['source_time'])+' (nondimensional). '
         'H is h/(cp*dT); C is an unscaled fraction. Matrix rows run bottom to top, with x varying fastest.')
    if data['imported']:
        table(['Imported field','Cell count','Minimum / maximum'],
              [[v['field'],str(v['count']),f'{number(v["minimum"])} / {number(v["maximum"])}'] for v in data['imported']], [110,100,301])
    else:
        text('Preview only: the final preheat matrices have not been computed for this configuration.')
    for label,c in zip(labels,configs):
        s=scales(c); regions=c['boundary_regions']['definitions']; flow=c['flow']['boundary_conditions']; bc=c['transport']['boundary_conditions']
        story.append(PageBreak()); title(label+' / regions and initial settings')
        text('Coordinates and profile parameters below use the input nondimensional coordinates. Physical extents are also listed.')
        table(['Region','Resolved extent and selection'], [[key,region_extent(c,value)] for key,value in regions.items()], [108,403])
        section('Phase, pressure and porosity initialization')
        text('Phase model: '+c['phase']['model']+'. Equilibrium phase and temperature are derived from H/C and phase pressure, not independently prescribed.')
        text('Phase pressure: '+json.dumps(c['phase']['pressure'],sort_keys=True)+'. For lithostatic pressure, '
             'P=rho*g*max(surface_y-y,0)*l0 in Pa, using rho='+number(c['phase']['parameters']['rho'])+' kg/m^3.')
        text('Flow porosity configuration: '+json.dumps(c['porosity'],sort_keys=True))
        text('Flow velocities and pressures are obtained by the Darcy-Stokes solve; no separate initial velocity field is prescribed.')
        if label=='Preheat':
            text('Legacy preheat holds flow and composition fixed. The thermal update diffuses H directly; phase-equilibrium temperature is diagnostic.')
        else:
            text('H/C, phase and Darcy-Stokes flow are updated during full evolution. The configured analytic H/C profiles are overwritten by the handoff:')
            for field in ('enthalpy','composition'):
                text(field+': '+profile_text(c['transport']['initial_conditions'][field]))
        story.append(PageBreak()); title(label+' / flow boundaries')
        text('Stokes components are '+flow['stokes'].get('components','cartesian')+'. Dirichlet prescribes solid velocity; Neumann prescribes the configured traction component. '
             'Segments override defaults by priority; omitted segment components retain their applicable default. '
             'At boundary vertices, the solver resolves compatible Dirichlet conditions ahead of natural loads.')
        rows=[]
        rules=[('All sides / default',0,flow['stokes']['default'])]+[(v['region'],v.get('priority',10),v) for v in flow['stokes'].get('segments',[])]
        for region,priority,rule in rules:
            for component in ('x','y','normal','tangent'):
                if component not in rule: continue
                value=rule[component]; velocity=value['type']=='dirichlet'
                rows.append([region+'\npriority '+str(priority),component+' / '+value['type'],
                             profile_text(value['value'],s['velocity_m_s']*100*YEAR if velocity else s['pressure_Pa']/1e6,
                                          'cm/year' if velocity else 'MPa')])
        table(['Region','Component / type','Configured profile and physical units'],rows,[91,86,334])
        section('Darcy boundaries')
        darcy=flow['darcy']
        text('Dirichlet variable: '+darcy['dirichlet_variable']+'. Neumann variable: '+darcy['neumann_variable']+'. '
             'Zero assembled normal velocity closes normal Darcy transport. Prescribed pressure potential allows the solver to determine normal flow; it is not a prescribed liquid velocity.')
        rows=[]
        for region,rule in [('All sides / default',darcy['default'])]+[(v['region'],v) for v in darcy.get('segments',[])]:
            rows.append([region,rule['type']+'\npriority '+str(rule.get('priority',0 if region.startswith('All sides') else 10)),profile_text(rule['value'])])
        table(['Region','Type','Profile (input units)'], rows, [108,104,299])
        text('Darcy assembled velocity is the rescaled relative variable. Physical segregation flux is q=u0*phi^(1+theta)*ud; '
             'theta='+number(c['flow']['parameters']['theta'])+'. Physical normal flux depends on porosity.')
        story.append(PageBreak()); title(label+' / transport boundaries')
        text('Advection and thermal diffusion are assigned independently. Regions refer to the extents listed for this stage.')
        for field,heading_text in [('enthalpy','H / enthalpy advection'),('composition','C / composition advection'),('temperature','Thermal diffusion')]:
            section(heading_text)
            rows=[]
            for region,rule in [('All sides / default',bc[field]['default'])]+[(v['region'],v) for v in bc[field].get('segments',[])]:
                kind=rule['type']
                meaning={'outflow':'Interior state is extrapolated; backflow is rejected.',
                         'zero_flux':'Zero normal advective flux.' if field!='temperature' else 'Zero normal diffusive heat flux (insulated).',
                         'inflow_outflow':'Prescribed state on inflow; interior state on outflow.',
                         'prescribed_temperature':'Dirichlet boundary for the diffused thermal quantity.',
                         'prescribed_flux':'Prescribed outward diffusive flux.'}.get(kind,kind)
                if 'value' in rule:
                    scale,unit=(s['enthalpy_J_kg'],'J/kg') if field=='enthalpy' else ((1,'fraction') if field=='composition' else (s['temperature_K'],'K'))
                    if kind=='prescribed_flux': scale=unit=None
                    meaning+='\n'+profile_text(rule['value'],scale,unit)
                rows.append([region+'\npriority '+str(rule.get('priority',0 if region.startswith('All sides') else 10)),kind,meaning])
            table(['Region','Policy','Boundary state / meaning'],rows,[100,115,296])
        text('Zero normal characteristic speed gives zero advection. H uses mixture velocity; C uses the effective composition velocity in full evolution. '
             'Preheat advects H with the fixed solid velocity and does not evolve C. No composition diffusion is included.')
        k=float(c['transport']['thermal_diffusion']['diffusivity'])
        text('Thermal diffusivity: '+number(k)+' nondimensional = '+number(k*s['length_m']*s['velocity_m_s'])+' m^2/s. '
             'Thermal sampling: '+json.dumps(c['transport']['thermal_diffusion'],sort_keys=True)+'.')
        text(('Preheat applies the temperature boundary numbers directly to H in its legacy H-diffusion approximation. '
              'The printed K conversion describes that thermal proxy, not a separately constrained equilibrium phase temperature.') if label=='Preheat' else
             'Full evolution diffuses the equilibrium temperature T(H,C,P). Temperature is absolute: zero means 0 K, not 0 degrees Celsius.')
    story.append(PageBreak()); title('Reference settings and provenance')
    for label,c in zip(labels,configs):
        section(label+' / reference scales and pressure settings')
        s=scales(c)
        table(['Scale','Value'],[[key,number(value)] for key,value in s.items()],[170,341])
        text('Phase/material parameters (SI, except composition Xe): '+json.dumps(c['phase']['parameters'],sort_keys=True))
        text('Flow formulation and pressure constraints: '+json.dumps(c['flow']['linear_system'],sort_keys=True))
        text('Body forcing / pressure sources: '+json.dumps(c['flow']['forcing'],sort_keys=True))
    story.append(PageBreak()); title('Configuration provenance')
    section('Exact configuration sources')
    for label,source in zip(labels,data['inputs']):
        text(label+': '+source['path']+'\nSHA-256: '+source['sha256'])
    for item in data['imported']:
        text('Imported '+item['field']+': '+item['file']+'\nSHA-256: '+item['sha256'])
    text('The adjacent conditions JSON preserves the complete stage input mappings used to generate this report. '
         'The PDF reports the supplied configuration and saved initial states; it does not infer settings from plot colors or current example templates.')
    output=Path(output); temporary=output.with_suffix('.tmp.pdf')
    def footer(canvas,doc):
        canvas.saveState(); canvas.setStrokeColor(teal); canvas.line(42,37,553,37)
        canvas.setFont('Helvetica',8); canvas.setFillColor(ink)
        canvas.drawString(42,24,'MantlePar | Initial and boundary conditions')
        canvas.drawRightString(553,24,f'Page {doc.page}'); canvas.restoreState()
    document=SimpleDocTemplate(str(temporary),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=50,
                              title='MantlePar initial and boundary conditions',author='MantlePar')
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    temporary.replace(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir',type=Path,help='Completed evolution output; reads saved inputs and imported H/C')
    inputs.add_argument('--render-json',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.render_json:
        if not args.output: parser.error('--render-json requires --output')
        make_pdf(json.loads(args.render_json.read_text()),args.output)
    else:
        output=generate(load_recorded(args.input_dir),args.output or args.input_dir/'visualization'/PDF_NAME)
        print(f'Initial and boundary conditions PDF: {output}')


if __name__=='__main__':
    try: main()
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as error:
        print(f'Conditions report failed: {error}',file=sys.stderr)
        raise SystemExit(1)
