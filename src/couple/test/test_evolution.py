"""End-to-end coupled transport checks, independent restart data and RK2 oracle."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--executable',required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--mpiexec')
    ap.add_argument('--numproc-flag',default='-n')
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load(args.input.read_text())
    base['mesh']['cells']=[6,6]
    defs=base['boundary_regions']['definitions']
    defs['left_upper']['selector']={'mode':'boundary_cells','start':1,'count':5}
    for key in ('warm_top','melt_outlet'): defs[key]['selector']={'mode':'boundary_cells','start':0,'count':1}
    base['time']['end']=.02
    base['time']['step'].update(initial=.01,maximum=.01)
    base['flow']['solver']['pressure']['ksp']='cg'

    def restart(name,c,H,C):
        source=args.output_dir/(name+'_source'); source.mkdir(exist_ok=True)
        nx,ny=c['mesh']['cells']; m=c['mesh']; p={k:float(v) for k,v in c['phase']['parameters'].items()}
        l0=(p['mus']*p['k0']/p['mul'])**.5; u0=p['k0']*p['rhor']*p['g']/p['mul']
        data={'schema_version':1,'status':'complete','representation':'cell averages','ordering':'j*nx+i',
              'H':{'file':'cellH1.dat','units':'h/(cp*dT)'},'C':{'file':'cellC1.dat','units':'fraction'},'time':15000,
              'mesh':{'family':m['family'],'nx':nx,'ny':ny,'domain':m['domain']['x']+m['domain']['y'],
                      'perturbation':m['perturbation']['amplitude'],'seed':m['perturbation']['seed']},
              'scales':{'enthalpy_Jkg':p['cp']*(p['Tm0']-p['Te0']),'length_m':l0,'time_s':l0/u0}}
        (source/'transport_state.json').write_text(json.dumps(data))
        np.savetxt(source/'cellH1.dat',H,fmt='%.17g'); np.savetxt(source/'cellC1.dat',C,fmt='%.17g')
        return source

    def run(name,c,source,mpi=False,fail=False,grid=(2,1)):
        c=copy.deepcopy(c); dest=(args.output_dir/name).resolve(); c['output']['directory']=str(dest)
        if mpi:
            c['parallel'].update(ranks=2,process_grid=list(grid))
            c['flow']['solver'].update(ksp='fgmres',preconditioner='schur')
        inp=args.output_dir/(name+'.yaml'); inp.write_text(yaml.safe_dump(c,sort_keys=False))
        cmd=[args.executable,'-input',str(inp),'-preheat',str(source),'-couple_porosity_snapshots','-couple_movie_snapshots']
        if mpi: cmd=[args.mpiexec,args.numproc_flag,'2']+cmd
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
        if fail:
            assert result.returncode!=0,result.stdout
            return result.stdout+result.stderr
        assert result.returncode==0,result.stdout+result.stderr
        m=json.loads((dest/'transport_state.json').read_text())
        v=json.loads((dest/'visualization.json').read_text())
        assert m['time']==c['time']['end'] and v['flow_time']==m['time']
        assert v['evolution_model']=='full phase coupling'
        hist=np.atleast_1d(np.genfromtxt(dest/'evolution_history.csv',delimiter=',',names=True))
        assert len(hist)==m['accepted_steps']+1
        series=json.loads((dest/'porosity_series.json').read_text())
        assert series['status']=='complete' and series['sampling']=='cell centers'
        assert series['frames']==len(hist) and series['time_end']==m['time']
        ids=[]; final_phi=[]
        for name in series['files']:
            frames=np.atleast_1d(np.genfromtxt(dest/name,delimiter=',',names=True))
            np.testing.assert_array_equal(frames['step'],hist['step'])
            np.testing.assert_array_equal(frames['time'],hist['time'])
            np.testing.assert_array_equal(frames['time_years'],hist['time_years'])
            for key in frames.dtype.names[3:]:
                ids.append(int(key.removeprefix('phi_'))); final_phi.append(frames[key][-1])
                assert np.isfinite(frames[key]).all() and (frames[key]>=0).all() and (frames[key]<1).all()
        np.testing.assert_array_equal(np.sort(ids),np.arange(series['cell_count']))
        centers=np.concatenate([np.atleast_1d(np.genfromtxt(dest/f['centers'],delimiter=',',names=True)) for f in v['files']])
        np.testing.assert_allclose(np.array(final_phi)[np.argsort(ids)],centers['phi'][np.argsort(centers['entity_id'])],atol=1e-14)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from animate_evolution import Series
        movies=Series(dest)  # Validates every rank, frame, physical field and final CSV match.
        np.testing.assert_array_equal(movies.clock[:,1],hist['time'])
        assert np.max(np.abs(hist['H_balance_error']))<1e-12
        assert np.max(np.abs(hist['C_balance_error']))<1e-12
        np.testing.assert_allclose(np.diff(hist['H_integral']),hist['dt'][1:]*(hist['adiabatic_H_source'][1:]-hist['outward_H_flux'][1:]),atol=2e-13)
        np.testing.assert_allclose(np.diff(hist['C_integral']),-hist['dt'][1:]*hist['outward_C_flux'][1:],atol=2e-13)
        assert np.max(hist['courant'])<=c['time']['step']['cfl']*(1+1e-10)
        assert np.max(hist['mixture_divergence_max'])<1e-7
        # The limiter must preserve cell averages, not merely produce bounded points.
        cell_data=np.concatenate([np.atleast_1d(np.genfromtxt(dest/f["cells"],delimiter=",",names=True)) for f in v["files"]])
        h_final=np.loadtxt(dest/"cellH1.dat").ravel(); c_final=np.loadtxt(dest/"cellC1.dat").ravel()
        assert cell_data["H"].min()>=0 and cell_data["C"].min()>=0 and cell_data["C"].max()<=float(c["phase"]["parameters"]["Xe"])
        for cell in range(len(h_final)):
            points=cell_data[cell_data["entity_id"]==cell]
            np.testing.assert_allclose(np.average(points["H"],weights=points["weight"]),h_final[cell],atol=2e-12)
            np.testing.assert_allclose(np.average(points["C"],weights=points["weight"]),c_final[cell],atol=2e-12)
        for field in ('H','C'):
            np.testing.assert_array_equal(np.loadtxt(dest/f'starting_cell{field}.dat'),np.loadtxt(source/f'cell{field}1.dat'))
        return dest,np.loadtxt(dest/'cellH1.dat'),np.loadtxt(dest/'cellC1.dat'),hist

    x=np.arange(6)[None,:]; y=np.arange(6)[:,None]
    H=3.3+.005*x-.001*y; C=.04+.002*y+.0001*x
    source=restart('wet',base,H,C)
    serial=run('wet',base,source)
    assert np.max(np.abs(serial[1]-H))>1e-9
    assert np.max(np.abs(serial[2]-C))>1e-9
    assert serial[3]['phi_max'].max()>0
    assert np.any(serial[3]['adiabatic_H_source']!=0)
    if args.mpiexec:
        mpi=run('wet_mpi',base,source,mpi=True)
        np.testing.assert_allclose(mpi[1],serial[1],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(mpi[2],serial[2],rtol=1e-9,atol=1e-10)
        mpi_y=run('wet_mpi_y',base,source,mpi=True,grid=(1,2))
        np.testing.assert_allclose(mpi_y[1],serial[1],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(mpi_y[2],serial[2],rtol=1e-9,atol=1e-10)
        return

    warped=copy.deepcopy(base); warped['mesh']['family']='perturbed_quadrilateral'
    warped_source=restart('warped',warped,H,C)
    run('warped',warped,warped_source)

    # A uniform, dry, motionless 2x2 box cools uniformly. Its semidiscrete
    # thermal mode is exactly exponential, independent of flow/reconstruction.
    ode=copy.deepcopy(base); ode['mesh']['cells']=[2,2]
    ode['boundary_regions']['definitions']={k:v for k,v in defs.items() if k in ('left','right','bottom','top')}
    constant=lambda value:{'name':'constant','parameters':{'value':value}}
    zero={'type':'dirichlet','value':constant(0.)}
    ode['flow']['boundary_conditions']={
        'stokes':{'components':'cartesian','default':{'x':zero,'y':zero}},
        'darcy':{'dirichlet_variable':'assembled_normal_velocity','neumann_variable':'pressure_potential','default':zero}}
    ode['flow']['forcing']['stokes']=constant([0.,0.])
    ode['flow']['linear_system'].update(pressure_nullspace='provided',pressure_modes={'stokes':1.,'darcy':0.})
    ode['flow']['solver']['remove_pressure_nullspace']=True
    ode['flow']['solver'].update(ksp='fgmres',preconditioner='schur')
    ode['transport']['boundary_conditions']={
        'enthalpy':{'default':{'type':'zero_flux'}},'composition':{'default':{'type':'zero_flux'}},
        'temperature':{'default':{'type':'prescribed_temperature','value':constant(2.)}}}
    ode['transport']['thermal_diffusion']['diffusivity']=.01
    ode['time']['end']=.1; ode['time']['step']['control']='fixed'
    src=restart('ode',ode,np.full((2,2),2.1),np.full((2,2),.05))
    decay=2*.01*(25/12)/(.9*.25/4)/.25
    exact=2+.1*np.exp(-decay*.1); errors=[]
    for dt in (.01,.005):
        ode['time']['step'].update(initial=dt,maximum=dt,minimum=1e-12)
        result=run('rk2_'+str(dt),ode,src)
        np.testing.assert_allclose(result[1],result[1][0,0],atol=1e-12)
        errors.append(abs(result[1][0,0]-exact))
    assert 3.8<errors[0]/errors[1]<4.2,errors
    bad=copy.deepcopy(base); bad['mesh']['cells']=[7,6]
    assert 'mesh does not match' in run('bad_mesh',bad,source,fail=True)
    incomplete=json.loads((source/'transport_state.json').read_text()); incomplete['status']='incomplete'
    (source/'transport_state.json').write_text(json.dumps(incomplete))
    assert 'must be complete' in run('bad_manifest',base,source,fail=True)
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'plot_initialization.py'),'--input-dir',str(serial[0])],check=True,timeout=90)
    assert (serial[0]/'visualization/evolution_history.png').exists()
    print('Passed: exact handoff, wet H/C evolution, conservation, flow continuity, SSPRK2 order, rejection and physical plots')

if __name__=='__main__': main()
