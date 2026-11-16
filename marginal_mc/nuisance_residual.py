"""Honest residual conditional density estimation, compatible with nuisance.py.

No DGP parameters, true residuals, evaluation responses, or diagnostic grids
are used. ``density_conditioning='x_only'`` and ``pilot='additive_spline'``
are explicit structural restrictions, NOT generally valid dimension reduction.

The original mean fit can be retained. Density pilots are separate estimators
of the true conditional mean; the nuisance density need not be centered at the
particular fitted mean used in the target estimating equation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import numpy as np
from scipy.sparse import csr_array
from sklearn.preprocessing import SplineTransformer
from .dgp import features
from .nuisance import LocalLinear, WeightResult, _cv_mean_bandwidth

SQRT2PI=math.sqrt(2*math.pi)

class _LocalLinearPilot:
    def __init__(self, v: np.ndarray, y: np.ndarray, settings: dict[str, Any], cv: bool=False):
        n,d=v.shape
        base=(math.log(max(n,3))/n)**(1/(4+d))
        self.y=y
        selection=settings.get('bandwidth_selection',{})
        if cv and selection.get('method','rule')=='inner_cv':
            h, *_ = _cv_mean_bandwidth(v, y, base, int(settings.get("min_neighbors", 12)), selection)
        else:
            h=float(settings.get('pilot_m_bw_mult',settings.get('m_bw_mult',.8)))*base
        self.smoother=LocalLinear(v,h,int(settings.get('min_neighbors',12)))
    def predict(self,v: np.ndarray)->np.ndarray:
        return np.concatenate([np.asarray(self.smoother.weights(v[i:i+256]).matrix@self.y).ravel() for i in range(0,len(v),256)])

class _AdditiveSplinePilot:
    """Growing cubic additive spline sieve; no supplied regression coefficients."""
    def __init__(self,v:np.ndarray,y:np.ndarray,settings:dict[str,Any]):
        n=len(y)
        # This grows with n; a fixed basis would not justify vanishing approximation bias.
        nk=max(4,int(math.ceil(float(settings.get('spline_knot_mult',3.0))*(n/math.log(max(n,3)))**(1/9))))
        self.transform=SplineTransformer(n_knots=nk,degree=3,knots='uniform',include_bias=False,extrapolation='linear')
        self.transform.fit(v)
        a=self.design(v)
        ridge=float(settings.get('spline_ridge',1e-6))/max(n,1)
        if ridge<0 or not math.isfinite(ridge):raise ValueError('spline_ridge must be finite and nonnegative')
        penalty=np.eye(a.shape[1])*ridge;penalty[0,0]=0.
        self.coef=np.linalg.solve(a.T@a+penalty,a.T@y)
        self.n_knots=nk
    def design(self,v:np.ndarray)->np.ndarray:
        return np.column_stack((np.ones(len(v)),self.transform.transform(v)))
    def predict(self,v:np.ndarray)->np.ndarray:return self.design(v)@self.coef

@dataclass
class _ResidualModel:
    residual:np.ndarray
    smoother:LocalLinear
    b:float
    h:float
    pilot:Any
    pilot_indices:np.ndarray
    density_indices:np.ndarray

class _PreparedComponent:
    def __init__(self,weights:csr_array,residual:np.ndarray,b:float,order:int):
        self.weights=weights
        self.rows=np.repeat(np.arange(weights.shape[0]),np.diff(weights.indptr))
        self.residual=residual[weights.indices]
        self.b=b;self.order=order
    def raw_pdf(self,t:float|np.ndarray)->np.ndarray:
        value=np.asarray(t,float)
        if value.ndim and value.shape!=(self.weights.shape[0],):
            raise ValueError('t must be a scalar or one value per query row')
        at=value if value.ndim==0 else value[self.rows]
        u=(at-self.residual)/self.b
        with np.errstate(over='ignore',invalid='ignore'):
            k=np.exp(-.5*u*u)/(SQRT2PI*self.b)
            if self.order==4:k*=(3-u*u)/2
        k=np.nan_to_num(k,nan=0.,posinf=0.,neginf=0.)
        return np.bincount(self.rows,weights=self.weights.data*k,minlength=self.weights.shape[0])

class ResidualPrepared:
    def __init__(self,mean:np.ndarray,components:list[_PreparedComponent],positive_part:bool=False):
        self.mean=np.asarray(mean,float);self.components=components;self.positive_part=positive_part
        self.pdf_calls=0;self.negative_count=0;self.total_count=0
    def raw_pdf(self,t:float|np.ndarray)->np.ndarray:
        out=sum(component.raw_pdf(t) for component in self.components)/len(self.components)
        return np.asarray(out)
    def _record(self,out:np.ndarray)->None:
        self.pdf_calls+=1;self.negative_count+=int(np.sum(out<0));self.total_count+=len(out)
    def pdf(self,t:float|np.ndarray)->np.ndarray:
        out=self.raw_pdf(t);self._record(out)
        return np.maximum(out,0.) if self.positive_part else out
    def A(self,q:float)->np.ndarray:
        # A is signed. Never clip A or impose symmetry.
        left,right=self.raw_pdf(q),self.raw_pdf(-q)
        self._record(left);self._record(right)
        return left-right
    def correction(self,q:float,residual:np.ndarray)->float:
        residual=np.asarray(residual,float)
        if residual.shape!=self.mean.shape:raise ValueError('residual shape mismatch')
        return float(np.mean(residual*self.A(q)))

class ResidualDensityNuisance:
    """Two honest density subfits wholly inside one outer training fold.

    Each pilot is fit on one half; only the other half supplies KDE responses.
    Swapping halves gives two density estimates which are averaged, rather than
    pooling heterogeneous cross-fitted residuals into one allegedly iid sample.

    Defaults make no additive or conditional-independence assumption:
      pilot='local_linear', density_conditioning='full', response_order=4.
    The partially structured numerical-design alternative is explicitly enabled
    by pilot='additive_spline', density_conditioning='x_only'.
    """
    def __init__(self,x:np.ndarray,z:np.ndarray,y:np.ndarray,target:str,
                 settings:dict[str,Any],mean_model:Any|None=None):
        self.target=target;self.settings=dict(settings)
        y=np.asarray(y,float).reshape(-1);v=np.asarray(features(x,z,target),float)
        if len(y)!=len(v) or len(y)<32 or not np.isfinite(y).all() or not np.isfinite(v).all():
            raise ValueError('at least 32 finite, aligned training observations are required')
        self.conditioning=settings.get('density_conditioning','full')
        if self.conditioning not in ('full','x_only'):raise ValueError('invalid density_conditioning')
        self.order=int(settings.get('response_order',4))
        if self.order not in (2,4):raise ValueError('response_order must be 2 or 4')
        self.pilot_kind=settings.get('pilot','local_linear')
        if self.pilot_kind not in ('local_linear','additive_spline'):raise ValueError('invalid pilot')
        self.positive_part=bool(settings.get('density_positive_part',False))
        self.mean_model=mean_model
        self.mean_pilot=None if mean_model is not None else _LocalLinearPilot(v,y,settings,cv=True)
        self.stats:list[WeightResult]=[];self.models:list[_ResidualModel]=[]
        rng=np.random.default_rng(int(settings.get('residual_split_seed',20260911)))
        order=rng.permutation(len(y));halves=np.array_split(order,2)
        for j in range(2):
            pidx,didx=halves[j],halves[1-j]
            pilot=(_LocalLinearPilot(v[pidx],y[pidx],settings) if self.pilot_kind=='local_linear'
                   else _AdditiveSplinePilot(v[pidx],y[pidx],settings))
            r=y[didx]-pilot.predict(v[didx])
            vd=v[didx] if self.conditioning=='full' else v[didx,:1]
            M,d=vd.shape;rho=math.log(max(M,3))/M
            scale=float(np.std(r,ddof=1))
            if not math.isfinite(scale) or scale<1e-10:raise FloatingPointError('degenerate residual scale')
            if self.order==2:ph=pb=1/(5+d)
            else:ph,pb=2/(9+2*d),1/(9+2*d)
            h=float(settings.get('residual_h_mult',.8))*rho**ph
            b=float(settings.get('residual_b_mult',.5))*scale*rho**pb
            if not (math.isfinite(h) and math.isfinite(b) and h>0 and b>0):
                raise ValueError('bandwidth multipliers must be finite and positive')
            self.models.append(_ResidualModel(r,LocalLinear(vd,h,int(settings.get('min_neighbors',12))),b,h,pilot,pidx.copy(),didx.copy()))
        self.feature_dim=v.shape[1]
    def predict_mean(self,x:np.ndarray,z:np.ndarray)->np.ndarray:
        if self.mean_model is not None:return self.mean_model.predict_mean(x,z)
        return self.mean_pilot.predict(features(x,z,self.target))
    def prepare(self,x:np.ndarray,z:np.ndarray)->ResidualPrepared:
        mean=self.predict_mean(x,z);v=features(x,z,self.target)
        if self.conditioning=='x_only':v=v[:,:1]
        components=[]
        for model in self.models:
            wr=model.smoother.weights(v);self.stats.append(wr)
            components.append(_PreparedComponent(wr.matrix,model.residual,model.b,self.order))
        return ResidualPrepared(mean,components,self.positive_part)
    def diagnostics(self)->dict[str,float]:
        stats=self.stats
        return dict(h_m=float(self.mean_pilot.smoother.h) if self.mean_pilot is not None else math.nan,
          h_f=float(np.mean([m.h for m in self.models])),b_y=float(np.mean([m.b for m in self.models])),
          feature_dim=float(self.feature_dim),density_dim=float(self.models[0].smoother.train.shape[1]),
          response_order=float(self.order),honest_residual=1.,density_oracle=0.,direct_contrast=1.,
          additive_pilot=float(self.pilot_kind=='additive_spline'),x_only=float(self.conditioning=='x_only'),
          ll_expanded_rows=float(sum(s.expanded_rows for s in stats)),ll_pinv_rows=float(sum(s.pinv_rows for s in stats)),
          ll_min_neighbors=float(min((s.min_neighbors for s in stats),default=0)),
          ll_max_condition=float(max((s.max_condition for s in stats),default=0)))
