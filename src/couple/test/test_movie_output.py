"""Movie-format and rendering checks using synthetic fixtures, no solver run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from animate_evolution import Series, Movie, render, CENTER_FIELDS, GAUSS_FIELDS
from movie_fields import SpatialGrid, frame_indices, velocity_vectors, YEAR_SECONDS


def fixture(path, ranks=1):
    path.mkdir(parents=True,exist_ok=True)
    nx,ny,nq,frames=4,3,4,3
    cells=np.arange(nx*ny)
    xy=np.c_[cells%nx+.5,cells//nx+.5]*1000
    phase=np.empty((frames,len(cells),8))
    for k in range(frames):
        te=1480+20*(ny-xy[:,1]/1000)
        region=np.where(cells%nx==0,2,np.where(cells%nx==1,3,4))
        phi=np.where(region==2,0,.04+.002*k)
        phase[k]=np.c_[te+np.where(region==2,-100,np.where(region==3,0,100))+k*(region!=3),
                        te,phi,1-phi-.01,np.full(len(cells),.01),region,np.where(phi>0,.5,np.nan),np.full(len(cells),.02)]
    q=np.tile(np.arange(nq),len(cells)); ids=np.repeat(cells,nq)
    pxy=xy[ids]+np.c_[np.where(q%2==0,-1,1),np.where(q//2==0,-1,1)]*(500/np.sqrt(3))
    flow=np.empty((frames,len(ids),5))
    for k in range(frames):
        phi=phase[k,ids,2]
        flow[k]=np.c_[np.full(len(ids),(k+1)*1e-9),np.full(len(ids),1e-9),phi*1e-9,phi*2e-9,phi]
    meta={'schema_version':1,'status':'complete','dtype':'<f8','layout':'frame,point,field','frames':frames,
          'nx':nx,'ny':ny,'cell_count':nx*ny,'cell_points_per_axis':2,'time_end':1,'eutectic_reference_K':1480,
          'times':'evolution_times.csv','center_fields':CENTER_FIELDS,'gauss_fields':GAUSS_FIELDS,'files':[]}
    final={'schema_version':2,'status':'complete','accepted_steps':2,'time':1,'ranks':ranks,'cell_count':nx*ny,
           'domain':[0,nx,0,ny],'scales':{'length_m':1000,'time_s':YEAR_SECONDS},'files':[]}
    np.savetxt(path/'evolution_times.csv',[[0,0,0],[1,.99999,.99999],[2,1,1]],delimiter=',',header='step,time,time_years',comments='',fmt='%.17g')
    for rank in range(ranks):
        ci=cells[cells%ranks==rank][::-1]  # Deliberately noncanonical per-rank ordering.
        gi=np.flatnonzero(np.isin(ids,ci))[::-1]
        entry={'rank':rank,'centers':f'c{rank}.bin','gauss':f'g{rank}.bin','center_points':f'cp{rank}.csv',
               'gauss_points':f'gp{rank}.csv','center_count':len(ci),'gauss_count':len(gi)}
        phase[:,ci,:].astype('<f8').tofile(path/entry['centers']); flow[:,gi,:].astype('<f8').tofile(path/entry['gauss'])
        np.savetxt(path/entry['center_points'],np.c_[ci,xy[ci]],delimiter=',',header='cell_id,x_m,y_m',comments='')
        np.savetxt(path/entry['gauss_points'],np.c_[ids[gi],q[gi],pxy[gi]],delimiter=',',header='cell_id,q,x_m,y_m',comments='')
        np.savetxt(path/f'final_c{rank}.csv',np.c_[ci,phase[-1,ci]],delimiter=',',header='entity_id,'+','.join(CENTER_FIELDS),comments='')
        fields=[('flow_phi' if field=='flow_phi' else field) for field in GAUSS_FIELDS]
        np.savetxt(path/f'final_g{rank}.csv',np.c_[ids[gi],q[gi],flow[-1,gi]],delimiter=',',header='entity_id,q,'+','.join(fields),comments='')
        final['files'].append({'centers':f'final_c{rank}.csv','flow_cells':f'final_g{rank}.csv'})
        meta['files'].append(entry)
    (path/'evolution_series.json').write_text(json.dumps(meta)); (path/'visualization.json').write_text(json.dumps(final))
    return phase,flow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--encode',action='store_true')
    args=parser.parse_args()
    root=args.output_dir or Path(tempfile.mkdtemp(prefix='mantle-movie-test-'))
    expected_c,expected_g=fixture(root/'serial')
    fixture(root/'split',2)
    a,b=Series(root/'serial'),Series(root/'split')
    np.testing.assert_allclose(a.centers,expected_c,equal_nan=True)
    np.testing.assert_allclose(a.centers,b.centers,equal_nan=True)
    for k in range(3):
        np.testing.assert_allclose(a.gauss(k),expected_g[k]); np.testing.assert_allclose(a.gauss(k),b.gauss(k))
    indices=frame_indices(a.clock,.5,2)
    np.testing.assert_array_equal(np.unique(indices),np.arange(3))
    assert (np.diff(indices)>=0).all()
    solid,liquid,flux=velocity_vectors(a.gauss(0)); wet=a.gauss(0)[:,4]>0
    np.testing.assert_allclose(solid[:,0],1e-9*100*YEAR_SECONDS)
    np.testing.assert_allclose(liquid[wet],solid[wet]+flux[wet]/a.gauss(0)[wet,4,None])
    assert np.isnan(liquid[~wet]).all()
    grid=SpatialGrid(a.center_xy,a.domain,resolution=15)
    interpolated=grid.sample(2*a.center_xy[:,0]-3*a.center_xy[:,1]+4)
    np.testing.assert_allclose(interpolated.compressed(),(2*grid.xx-3*grid.yy+4)[~interpolated.mask],atol=1e-12)
    assert interpolated.mask[0,0]
    for kind in ('velocity','temperature','porosity','phase'):
        movie=Movie(a,kind); movie.draw(0); movie.draw(2)
        assert not any(line.get_visible() for ax in movie.axes.flat for line in ax.get_xgridlines()+ax.get_ygridlines())
        if kind=='velocity':
            assert len(movie.fig.axes)==6  # No magnitude color bars.
        else:
            assert all(image.get_extent()[0]==a.domain[0] for image in movie.images)
        movie.fig.savefig(root/f'{kind}_fixture.png')
        import matplotlib.pyplot as plt
        plt.close(movie.fig)
    # Reject partial output, rather than rendering a shortened/misaligned movie.
    damaged=root/'split/g0.bin'; content=damaged.read_bytes(); damaged.write_bytes(content[:-8])
    try: Series(root/'split')
    except ValueError as error: assert 'Incomplete binary' in str(error)
    else: raise AssertionError('Truncated binary file was accepted')
    if args.encode:
        if not shutil.which('ffmpeg'): raise RuntimeError('ffmpeg is required for --encode')
        render(a,root/'movies',seconds=.5,fps=2)
        metadata = json.loads((root/'movies/movies.json').read_text())
        assert metadata['files']['combined'] == 'combined_evolution.mp4'
        for kind in ('velocity','temperature','porosity','phase','combined'):
            subprocess.run(['ffmpeg','-v','error','-i',str(root/f'movies/{kind}_evolution.mp4'),'-f','null','-'],check=True)
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate,duration', '-of', 'json',
                str(root/f'movies/{kind}_evolution.mp4')], text=True))['streams'][0]
            assert int(probe['nb_read_frames']) == metadata['video_frames']
            assert probe['avg_frame_rate'] == '2/1'
            assert abs(float(probe['duration'])-metadata['duration_seconds']) < 1e-8
            if kind == 'combined':
                assert (probe['width'], probe['height']) == (2560, 2520)
        # The composite preserves every source panel, with no scaling/cropping.
        combined = plt.imread(root/'movies/combined_evolution_poster.png')[:,:,:3]
        for kind, (x,y) in zip(('velocity','temperature','porosity','phase'), ((0,540),(1280,0),(1280,720),(1280,1440))):
            panel = plt.imread(root/f'movies/{kind}_evolution_poster.png')[:,:,:3]
            np.testing.assert_array_equal(combined[y:y+panel.shape[0],x:x+panel.shape[1]], panel)
        assert '\\phi' in (root/'movies/report.md').read_text()
        assert 'combined_evolution.mp4' in (root/'movies/index.html').read_text()
    print('Passed: MPI file assembly, physical velocities, dry masking, interpolation, all-state timing, mesh-free figures, truncation checks')


if __name__=='__main__': main()
