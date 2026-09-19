#!/usr/bin/env python3
"""Create cell-center phase/temperature movies and Gauss-point velocity movies."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'mantlepar-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from movie_fields import SpatialGrid, VelocityDrawing, frame_indices, reference_speed, velocity_vectors

CENTER_FIELDS = ['temperature_K','eutectic_temperature_K','phi','phi1','phi2','region','cl','cs']
GAUSS_FIELDS = ['vs_x_m_s','vs_y_m_s','q_x_m_s','q_y_m_s','flow_phi']
REGIONS = ['Pure solid','Two solids','Eutectic','Solid + melt','Liquid','Pure melting']
REGION_COLORS = ['#435269','#76a5ae','#e5bc47','#d97942','#b74356','#9372b2']
MOVIE_KINDS = ('velocity', 'temperature', 'porosity', 'phase')
# Preserve every panel at its native pixel size: velocity on the left, the three
# scalar movies stacked on the right. All inputs use the same accepted-state clock.
COMBINED_LAYOUT = '0_540|1280_0|1280_720|1280_1440'


def combine_movies(output):
    """Compose the four synchronized movies and their posters without cropping."""
    output = Path(output)
    base = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-filter_complex_threads', '1']
    graph = f'[0:v][1:v][2:v][3:v]xstack=inputs=4:layout={COMBINED_LAYOUT}:fill=white:shortest=1[out]'
    for poster in (False, True):
        suffix = '_poster.png' if poster else '.mp4'
        temporary = output/('combined_evolution.tmp'+suffix)
        command = base.copy()
        for kind in MOVIE_KINDS:
            command += ['-i', str(output/(kind+'_evolution'+suffix))]
        command += ['-filter_complex', graph, '-map', '[out]', '-an']
        command += ['-frames:v', '1', '-threads', '1'] if poster else [
            '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-threads', '2']
        subprocess.run(command+[str(temporary)], check=True)
        temporary.replace(output/('combined_evolution'+suffix))
    return 'combined_evolution.mp4'


def local_file(directory, name):
    path = (directory/name).resolve()
    if path.parent != directory:
        raise ValueError('Series files must be inside the output directory')
    return path


class Series:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.meta = m = json.loads((self.directory/'evolution_series.json').read_text())
        self.final = final = json.loads((self.directory/'visualization.json').read_text())
        if m.get('schema_version')!=1 or m.get('status')!='complete' or final.get('status')!='complete':
            raise ValueError('Run with -couple_movie_snapshots to completion before rendering')
        if m.get('dtype')!='<f8' or m.get('layout')!='frame,point,field' or m['center_fields']!=CENTER_FIELDS or m['gauss_fields']!=GAUSS_FIELDS:
            raise ValueError('Unsupported movie data format')
        if m['cell_count']!=final['cell_count'] or m['frames']!=final['accepted_steps']+1 or m['time_end']!=final['time']:
            raise ValueError('Movie series and final output do not match')
        if len(m['files'])!=final['ranks'] or sorted(e['rank'] for e in m['files'])!=list(range(final['ranks'])):
            raise ValueError('Missing or repeated MPI rank')
        self.clock = np.loadtxt(local_file(self.directory,m['times']),delimiter=',',skiprows=1,ndmin=2)
        if self.clock.shape!=(m['frames'],3) or len(self.clock)<2 or not np.isfinite(self.clock).all():
            raise ValueError('Need a finite time series with at least two frames')
        if not np.array_equal(self.clock[:,0],np.arange(m['frames'])) or not (np.diff(self.clock[:,1])>0).all():
            raise ValueError('Missing or unordered accepted steps')
        np.testing.assert_allclose(self.clock[-1,1],final['time'],atol=1e-14)
        np.testing.assert_allclose(self.clock[:,2],self.clock[:,1]*final['scales']['time_s']/(365*24*3600),rtol=1e-14,atol=1e-14)
        centers, center_points, gauss_points = [], [], []
        self.gauss_parts = []
        names = []
        for entry in m['files']:
            for kind, fields, target in [('centers',CENTER_FIELDS,centers),('gauss',GAUSS_FIELDS,self.gauss_parts)]:
                count = entry['center_count' if kind=='centers' else 'gauss_count']
                path = local_file(self.directory,entry[kind]); names.append(path.name)
                if count<1 or path.stat().st_size!=m['frames']*count*len(fields)*8:
                    raise ValueError(f'Incomplete binary movie file: {path.name}')
                target.append(np.memmap(path,dtype='<f8',mode='r',shape=(m['frames'],count,len(fields))))
            for key,target,count in [('center_points',center_points,entry['center_count']),('gauss_points',gauss_points,entry['gauss_count'])]:
                points=np.atleast_1d(np.genfromtxt(local_file(self.directory,entry[key]),delimiter=',',names=True))
                if len(points)!=count or not all(np.isfinite(points[k]).all() for k in points.dtype.names):
                    raise ValueError('Invalid movie sample coordinates')
                target.append(points)
        if len(set(names))!=len(names):
            raise ValueError('Repeated movie data file')
        cp=np.concatenate(center_points); gp=np.concatenate(gauss_points)
        order=np.argsort(cp['cell_id'])
        if not np.array_equal(cp['cell_id'][order],np.arange(m['cell_count'])):
            raise ValueError('Missing or repeated cell-center samples')
        self.centers=np.concatenate(centers,axis=1)[:,order,:]
        self.center_xy=np.c_[cp['x_m'][order],cp['y_m'][order]]/1000
        self.gauss_order=np.lexsort((gp['q'],gp['cell_id']))
        gp=gp[self.gauss_order]; nq=m['cell_points_per_axis']**2
        if not np.array_equal(gp['cell_id'],np.repeat(np.arange(m['cell_count']),nq)) or not np.array_equal(gp['q'],np.tile(np.arange(nq),m['cell_count'])):
            raise ValueError('Missing or repeated cell Gauss points')
        self.gauss_xy=np.c_[gp['x_m'],gp['y_m']]/1000
        if not np.isfinite(self.centers[:,:,:6]).all() or (self.centers[:,:,:2]<0).any():
            raise ValueError('Invalid temperature or phase movie values')
        fractions=self.centers[:,:,2:5]
        if (fractions<0).any() or (fractions>1).any():
            raise ValueError('Invalid phase fractions')
        np.testing.assert_allclose(fractions.sum(axis=2),1,atol=1e-12)
        if not np.isin(self.centers[:,:,5],np.arange(1,7)).all():
            raise ValueError('Invalid phase region IDs')
        self.domain=np.asarray(final['domain'])*final['scales']['length_m']/1000
        self.references=np.zeros(3)
        for k in range(len(self.clock)):
            g=self.gauss(k)
            if not np.isfinite(g).all() or (g[:,4]<0).any() or (g[:,4]>=1).any():
                raise ValueError('Invalid Gauss velocity/porosity movie values')
            for v,vec in enumerate(velocity_vectors(g)):
                self.references[v]=max(self.references[v],reference_speed(vec))
        # Independent final CSV export must agree with the binary movie data.
        final_centers=np.concatenate([np.atleast_1d(np.genfromtxt(local_file(self.directory,e['centers']),delimiter=',',names=True)) for e in final['files']])
        final_centers=np.sort(final_centers,order='entity_id')
        for field in ('temperature_K','eutectic_temperature_K','phi','phi1','phi2','region'):
            np.testing.assert_allclose(self.centers[-1,:,CENTER_FIELDS.index(field)],final_centers[field],rtol=1e-12,atol=1e-13)
        final_flow=np.concatenate([np.atleast_1d(np.genfromtxt(local_file(self.directory,e['flow_cells']),delimiter=',',names=True)) for e in final['files']])
        final_flow=final_flow[np.lexsort((final_flow['q'],final_flow['entity_id']))]
        for j,field in enumerate(GAUSS_FIELDS):
            np.testing.assert_allclose(self.gauss(-1)[:,j],final_flow['flow_phi' if field=='flow_phi' else field],rtol=1e-12,atol=1e-14)

    def gauss(self, index):
        return np.concatenate([part[index] for part in self.gauss_parts],axis=0)[self.gauss_order]


class Movie:
    def __init__(self, series, kind):
        self.s,self.kind=series,kind
        if kind=='velocity':
            self.fig,self.axes=plt.subplots(3,2,figsize=(12.8,14.4),dpi=100)
            self.fig.subplots_adjust(left=.075,right=.94,bottom=.06,top=.88,hspace=.35,wspace=.3)
            self.velocity=VelocityDrawing(self.fig,self.axes,series.gauss_xy,series.domain,series.references)
        else:
            shape=(2,2) if kind=='phase' else (1,2)
            self.fig,self.axes=plt.subplots(*shape,figsize=(12.8,10.8) if kind=='phase' else (12.8,7.2),dpi=100,squeeze=False)
            self.fig.subplots_adjust(left=.065,right=.87 if kind=='phase' else .95,bottom=.14,top=.79 if kind!='phase' else .84,hspace=.32,wspace=.34)
            self.grid=SpatialGrid(series.center_xy,series.domain,resolution=180)
            self.images=[]
            self.contours=[]
            if kind=='temperature':
                t=series.centers[:,:,0]; delta=t-t[0]
                change=max(1e-10,float(abs(delta).max()))
                definitions=[('Temperature (K)','inferno',float(t.min()),max(float(t.max()),float(t.min())+1),None),
                             ('Temperature change (K)','RdBu_r',-change,change,None)]
            elif kind=='porosity':
                phi=series.centers[:,:,2]; change=max(.01,float(abs(100*(phi-phi[0])).max()))
                definitions=[('Porosity / liquid fraction','viridis',0,max(.01,float(phi.max())),None),
                             ('Porosity change (percentage points)','RdBu_r',-change,change,None)]
            else:
                definitions=[('Solid 1 fraction','viridis',0,1,None),('Solid 2 fraction','viridis',0,1,None),
                             ('Liquid fraction','viridis',0,max(.01,float(series.centers[:,:,2].max())),None),
                             ('Phase region',ListedColormap(REGION_COLORS),None,None,BoundaryNorm(np.arange(.5,7),6))]
            for ax,(title,cmap,vmin,vmax,norm) in zip(self.axes.flat,definitions):
                image=ax.imshow(np.zeros(self.grid.xx.shape),extent=series.domain,origin='lower',interpolation='nearest',
                                cmap=cmap,vmin=vmin,vmax=vmax,norm=norm,aspect='equal')
                ax.set_title(title,fontsize=13,pad=12); ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.grid(False)
                bar=self.fig.colorbar(image,ax=ax,fraction=.046,pad=.035)
                if norm:
                    bar.set_ticks(range(1,7),labels=REGIONS); bar.ax.tick_params(labelsize=8)
                self.images.append(image)
        title={'velocity':'Velocity and melt transport','temperature':'Temperature evolution','porosity':'Porosity evolution','phase':'Phase evolution'}[kind]
        self.fig.suptitle(f'{title} · {series.meta["nx"]} × {series.meta["ny"]}',fontsize=22,fontweight='bold',y=.976)
        self.stamp=self.fig.text(.5,.915 if kind=='velocity' else .9,'',ha='center',fontsize=13)
        note='Gauss-point quivers · Constant-color streamlines · No speed color maps' if kind=='velocity' else 'Cell-center samples · Spatial interpolation for display · Fixed color scales'
        self.fig.text(.5,.022,note,ha='center',fontsize=10,color='#555555')
        self.detail=self.fig.text(.5,.055 if kind!='velocity' else .038,'',ha='center',fontsize=10,color='#444444')

    def draw(self, k):
        s=self.s; state=s.centers[k]
        self.stamp.set_text(f'Elapsed after preheat: {s.clock[k,2]-s.clock[0,2]:.2f} years  |  t = {s.clock[k,1]:.5f}  |  Step {int(s.clock[k,0])}')
        if self.kind=='velocity':
            self.velocity.draw(s.gauss(k)); return
        if self.kind=='temperature':
            fields=[state[:,0],state[:,0]-s.centers[0,:,0]]
        elif self.kind=='porosity':
            fields=[state[:,2],100*(state[:,2]-s.centers[0,:,2])]
        else:
            fields=[state[:,3],state[:,4],state[:,2],state[:,5]]
        for j,(image,field) in enumerate(zip(self.images,fields)):
            image.set_data(self.grid.sample(field,categorical=self.kind=='phase' and j==3))
        if self.kind=='temperature':
            for contour in self.contours:
                for collection in contour.collections: collection.remove()
            self.contours=[]
            ax=self.axes[0,0]
            difference=self.grid.sample(state[:,0]-state[:,1])
            plateau=self.grid.sample((state[:,5]==3).astype(float),categorical=True)
            handles=[]
            if difference.min()<0<difference.max():
                self.contours.append(ax.contour(self.grid.x,self.grid.y,difference,levels=[0],colors='white',linewidths=1.25,linestyles='--'))
                handles.append(Line2D([],[],color='white',ls='--',label='$T=T_e(P)$'))
            if plateau.min()<.5<plateau.max():
                self.contours.append(ax.contour(self.grid.x,self.grid.y,plateau,levels=[.5],colors='#00d7dd',linewidths=1.1))
                handles.append(Line2D([],[],color='#00d7dd',label='Eutectic phase boundary'))
            if ax.get_legend(): ax.get_legend().remove()
            if handles: ax.legend(handles=handles,loc='lower left',facecolor='#333333',labelcolor='white',fontsize=8,framealpha=.85)
            self.detail.set_text(f'Eutectic temperature $T_e(P)$: {state[:,1].min():.1f}–{state[:,1].max():.1f} K; reference at P = 0: {s.meta["eutectic_reference_K"]:g} K')


def write_notes(series, output, results):
    solver_rows=[]
    if 'solver' in series.final and 'solve' in series.final:
        configured,measured=series.final['solver'],series.final['solve']
        solver_rows=[('Final flow accepted','Yes' if measured['converged'] else 'No'),
                     ('Method / preconditioner',f"{measured['method']} / {measured['preconditioner']}"),
                     ('Final iterations',str(measured['iterations'])),
                     ('Relative true residual',f"{measured['relative_true_residual']:.3e}"),
                     ('Relative / absolute tolerance',f"{configured['rtol']:.3g} / {configured['atol']:.3g}")]
    notes=rf'''# Evolution videos — {series.meta['nx']} × {series.meta['ny']}

Elapsed time after preheat: {series.clock[0,2]:.4g}–{series.clock[-1,2]:.4g} years.
All {len(series.clock)} accepted states are shown; physical times are quantized to
video frames without interpolating in time. Short steps still receive one frame.

## Velocity labels

- **Solid velocity**: $v_s = u_0 u_s$.
- **Liquid velocity**: $v_l = v_s + q/\phi$, defined only where $\phi > 0$.
- **Darcy segregation flux**: $q = u_0\phi^{{1+\theta}}u_d$. This is liquid volume
  flux relative to the solid per unit area, not the liquid velocity.
- Arrows originate at a spatially thinned subset of the actual **cell Gauss points**.
  Arrow length uses a fixed physical reference labeled in **cm/year** for each field.
  There are no speed/magnitude heatmaps. Color and streamline width are constant.
- Streamlines use linear spatial interpolation of the Gauss-point vector samples.
  They are instantaneous flow lines, not particle trajectories. Interpolation is
  confined to the sample convex hull; liquid streamlines stop at dry samples.

## Temperature, porosity and phase

Temperature in K, porosity and phase fractions are evaluated at **cell centers**
from the bounded H/C reconstruction. Smooth panels interpolate these center samples
for display; they are not additional simulation data. Phase-region colors use
nearest-center categories, preserving integer phase labels. No mesh lines or
center markers are drawn. Coordinates are in km; color limits stay fixed in time.

The temperature movie marks the pressure-dependent **$T_e(P)$**, exported directly
by the phase model, rather than treating the reference {series.meta['eutectic_reference_K']:g} K
as a constant threshold at every depth. A dashed white line marks $T=T_e(P)$ when
it crosses the sampled field; a cyan outline marks an extended eutectic phase region.
The numerical eutectic-temperature range is printed on every frame.

Porosity change is in **percentage points**. Phase evolution shows the two solid
fractions, the liquid fraction, and the categorical phase region.

## Files

**Combined video** shows velocity on the left and temperature, porosity and phase
stacked on the right. All four panels show the same accepted state on every frame.
It preserves every individual video's labels at native resolution (2560 × 2520
pixels overall). The separate videos remain available for viewing each field larger.
These movies cover full evolution after the preheat handoff.

'''
    for kind in results:
        notes+=f'- [{kind.title()} video]({kind}_evolution.mp4)\n'
    conditions_link = ''
    if (output/'conditions.pdf').is_file():
        notes+='- [Initial and boundary conditions PDF](conditions.pdf)\n'
        conditions_link='<p><a href="conditions.pdf">Initial and boundary conditions (PDF)</a></p>'
    if solver_rows:
        notes+='\n## Final Darcy–Stokes solve\n\n| Setting / measurement | Value |\n| --- | --- |\n'
        notes+=''.join(f'| {label} | {value} |\n' for label,value in solver_rows)
        notes+='\nResiduals and solver tolerances refer to the nondimensional algebraic system.\n'
    (output/'report.md').write_text(notes)
    cards=''.join(f'<section><h2>{name.title()}</h2><video controls loop muted playsinline preload="metadata" poster="{name}_evolution_poster.png"><source src="{name}_evolution.mp4" type="video/mp4"></video><p><a href="{name}_evolution.mp4" download>Download MP4</a></p></section>' for name in results)
    solver_table=('<h2>Final Darcy–Stokes solve</h2><table>'+''.join(f'<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>' for label,value in solver_rows)+'</table>') if solver_rows else ''
    (output/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Evolution videos — {series.meta['nx']} × {series.meta['ny']}</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 24px;font:16px/1.6 system-ui;color:#213047;background:#f5f7fa}}video{{width:100%;background:white}}section{{background:white;padding:20px;margin:24px 0;border-radius:12px}}pre{{white-space:pre-wrap}}a{{color:#1760a5}}table{{border-collapse:collapse;background:white}}td,th{{padding:8px 18px;text-align:left;border-bottom:1px solid #dce1e8}}</style>
<h1>Evolution videos · {series.meta['nx']} × {series.meta['ny']}</h1><p>{series.clock[-1,2]-series.clock[0,2]:.2f} years after preheat · {len(series.clock)} recorded states · No mesh overlay</p>
{conditions_link}{solver_table}{cards}<h2>Field definitions and plotting notes</h2><p><a href="report.md">Markdown notes</a> · <a href="movies.json">Frame timing and provenance</a></p><pre>{html.escape(notes)}</pre></html>''')


def render(series, output, seconds=12, fps=24):
    if not FFMpegWriter.isAvailable(): raise ValueError('ffmpeg is required')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    indices=frame_indices(series.clock,seconds,fps)
    results={}
    for kind in MOVIE_KINDS:
        print(f'Rendering {kind}...',flush=True)
        movie=Movie(series,kind)
        path=output/f'{kind}_evolution.mp4'; temporary=output/f'{kind}_evolution.tmp.mp4'
        writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2'])
        previous=None
        with writer.saving(movie.fig,str(temporary),dpi=100):
            for k in indices:
                if k!=previous: movie.draw(k); previous=k
                writer.grab_frame(facecolor='white')
        temporary.replace(path)
        movie.fig.savefig(output/f'{kind}_evolution_poster.png',dpi=100,facecolor='white')
        plt.close(movie.fig)
        results[kind]=path.name
    print('Combining all four synchronized videos...', flush=True)
    results = {'combined': combine_movies(output), **results}
    metadata={'source_directory':str(series.directory),'recorded_states':len(series.clock),'fps':fps,
              'video_frames':len(indices),'duration_seconds':len(indices)/fps,'state_index_per_video_frame':indices.tolist(),
              'scalar_sampling':'cell centers','velocity_sampling':'cell Gauss points','mesh_overlay':False,
              'velocity_display':'quiver and streamlines; fixed arrow scales; no speed color maps','files':results,
              'combined': {'width':2560, 'height':2520, 'layout':COMBINED_LAYOUT,
                           'input_order':list(MOVIE_KINDS), 'synchronized':True}}
    (output/'movies.json').write_text(json.dumps(metadata,indent=2)+'\n')
    write_notes(series,output,results)
    print(f'Video report: {(output/"index.html").resolve()}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--seconds',type=float,default=12)
    parser.add_argument('--fps',type=int,default=24)
    args=parser.parse_args()
    try: render(Series(args.input_dir),args.output_dir or args.input_dir/'visualization',args.seconds,args.fps)
    except (OSError,ValueError,KeyError,AssertionError,subprocess.CalledProcessError) as error: parser.error(str(error))


if __name__=='__main__': main()
