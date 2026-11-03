"""Unified baseline fitting, prediction, artifact verification and reporting."""
import argparse
import json
from pathlib import Path
from .config import MODELS, resolve_config


def parser():
    root=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub=root.add_subparsers(dest='command',required=True)
    for name in ('train','predict'):
        p=sub.add_parser(name,allow_abbrev=False)
        p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--model',choices=MODELS,required=name=='train')
        p.add_argument('--config',type=Path)
        p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
        p.add_argument('--seed',type=int)
        p.add_argument('--num-particles',type=int)
        p.add_argument('--views',help='Comma separated camera indices; default all stored cameras')
        p.add_argument('--boundary-projection',choices=('none','known'))
        if name=='train':
            p.add_argument('--output-dir',type=Path,required=True)
            p.add_argument('--epochs',type=int)
            p.add_argument('--resume',type=Path)
        else:
            p.add_argument('--checkpoint',type=Path)
            p.add_argument('--output',type=Path,required=True)
            p.add_argument('--split',choices=('validation','test'),default='validation')
            p.add_argument('--case-index',type=int,default=0)
            p.add_argument('--num-probes',type=int)
            p.add_argument('--num-samples',type=int)
            p.add_argument('--fit-steps',type=int)
            p.add_argument('--viscosity',type=float,help='Explicit known physical kinematic viscosity for PINN')
            p.add_argument('--evaluate',action='store_true')
            p.add_argument('--require-valid-replay',action='store_true')
    p=sub.add_parser('describe')
    p.add_argument('--output-dir',type=Path,default=Path('reports'))
    p=sub.add_parser('verify')
    p.add_argument('path',type=Path)
    p=sub.add_parser('summarize')
    p.add_argument('--inputs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='describe':
        from .reporting import describe
        return describe(args.output_dir)
    if args.command=='verify':
        from .reporting import verify
        print(json.dumps(verify(args.path),indent=2))
        return
    if args.command=='summarize':
        from .reporting import summarize
        return summarize(args.inputs,args.output)
    changes={k:getattr(args,k) for k in ('seed','num_particles','boundary_projection') if getattr(args,k) is not None}
    if args.views is not None:
        changes['views']=[int(x) for x in args.views.split(',')]
    if args.command=='train':
        if args.epochs is not None:
            changes['epochs']=args.epochs
        from .training import train
        result=train(args.model,args.manifest,args.output_dir,resolve_config(args.config,changes),device=args.device,resume=args.resume)
    else:
        for key in ('num_samples','num_probes','fit_steps'):
            if getattr(args,key) is not None:
                changes[key]=getattr(args,key)
        supplied=json.loads(args.config.read_text()) if args.config else {}
        supplied.update(changes)
        if args.viscosity is not None:
            supplied.setdefault('pinn',{})['viscosity']=args.viscosity
        if args.fit_steps is not None and args.model=='pinn':
            supplied.setdefault('pinn',{})['fit_steps']=args.fit_steps
        from .inference import predict
        result=predict(args.manifest,args.output,kind=args.model,checkpoint=args.checkpoint,
            config=supplied or None,split=args.split,case_index=args.case_index,
            device=args.device,evaluate=args.evaluate,require_valid=args.require_valid_replay)
    print(result)
    return result
