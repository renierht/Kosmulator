#ifndef KOSMULATOR_IDE_DENSITIES_H
#define KOSMULATOR_IDE_DENSITIES_H
#include <math.h>

static double ide_exprel(double x) { return x==0 ? 1. : expm1(x)/x; }

/* Homogeneous backgrounds only. Densities have the units of dm0/de0.
 * Return 0 on success, 1 on invalid domain or nonfinite result.
 * Optional density positivity/future/doom priors live in the Python sampler.
 */
static int ide_density_values(int model, double a, double w, double v,
                              double d, double ddm, double dde,
                              double dm0, double de0,
                              double *dm, double *de, double *qh) {
  double x=log(a), u=w-v, r=dm0/de0;
  if (model<1 || model>10 || a<=0 || dm0<=0 || de0<=0 ||
      !isfinite(a+w+v+d+ddm+dde+dm0+de0)) return 1;
  if (model<=5) {
    double m00,m01,m10,m11,t,s,disc,f,g;
    if(model==2) {ddm=d; dde=d;}
    if(model==3) {ddm=d; dde=-d;}
    if(model==4) {ddm=d; dde=0;}
    if(model==5) {ddm=0; dde=d;}
    m00=-3*(1+v-ddm); m01=3*dde;
    m10=-3*ddm; m11=-3*(1+w+dde);
    t=(m00+m11)/2;
    disc=(ddm+dde+u)*(ddm+dde+u)-4*ddm*dde;
    if(disc<0 || !isfinite(disc)) return 1;
    s=1.5*sqrt(disc);
    if(ddm==0 && dde==0) {
      *dm=dm0*exp(-3*(1+v)*x); *de=de0*exp(-3*(1+w)*x);
    }
    else if(dde==0) {
      *dm=dm0*exp(-3*(1+v-ddm)*x);
      *de=exp(-3*(1+w)*x)*(de0-3*ddm*dm0*x*ide_exprel(3*(u+ddm)*x));
    }
    else if(ddm==0) {
      *de=de0*exp(-3*(1+w+dde)*x);
      *dm=exp(-3*(1+v)*x)*(dm0+3*dde*de0*x*ide_exprel(-3*(u+dde)*x));
    }
    else if(fabs(s*x)<1e-3) {
      f=exp(t*x)*cosh(s*x); g=exp(t*x)*x*ide_exprel(2*s*x)*exp(-s*x);
      *dm=f*dm0+g*((m00-t)*dm0+m01*de0);
      *de=f*de0+g*(m10*dm0+(m11-t)*de0);
    }
    else {
      double ep=exp((t+s)*x), em=exp((t-s)*x);
      double cp=.5*(dm0+((ddm+dde+u)*dm0+2*dde*de0)/sqrt(disc));
      double ce=.5*(de0+(-2*ddm*dm0-(ddm+dde+u)*de0)/sqrt(disc));
      *dm=cp*ep+(dm0-cp)*em; *de=ce*ep+(de0-ce)*em;
    }
    *qh=3*(ddm*(*dm)+dde*(*de));
  }
  else if(model==6) {
    double k=u+d;
    if(k==0) {
      double factor=exp(-3*(1+(v*r+w)/(1+r))*x);
      *dm=dm0*factor; *de=de0*factor;
    }
    else {
      double y=log(r)+3*k*x;
      double lb=3*k*x < 50 ? log1p(r/(1+r)*expm1(3*k*x)) :
                            (y>0 ? y+log1p(exp(-y)) : log1p(exp(y)))-log1p(r);
      *dm=dm0*exp(-3*(1+v-d)*x-d/k*lb);
      *de=de0*exp(-3*(1+w)*x-d/k*lb);
    }
    *qh=3*d*(*dm)*(*de)/(*dm+*de);
  }
  else if(model==7 || model==8) {
    double A,base;
    if(u>=0 || d<=u || d>(model==7 ? -u/r : -u*r)) return 1;
    if(model==7) {
      A=(u+d*r)*exp(-3*u*x);
      base=(A+r*(u-d))/(u*(1+r));
      if(base<=0 || !isfinite(base)) return 1;
      *dm=dm0*exp(-3*(1+v-u*d/(u-d))*x)*pow(base,d/(u-d));
      *de=(*dm)/r*(A-d*r)/u;
      *qh=3*d*(*dm)*(*dm)/(*dm+*de);
    }
    else {
      A=(u*r+d)*exp(3*u*x);
      base=(A+u-d)/(u*(1+r));
      if(base<=0 || !isfinite(base)) return 1;
      *de=de0*exp(-3*(1+w+u*d/(u-d))*x)*pow(base,d/(u-d));
      *dm=(*de)*r*(A-d)/(u*r);
      *qh=3*d*(*de)*(*de)/(*dm+*de);
    }
  }
  else {
    double A=1+3*v,B=1+3*w,proxy=A*r+B;
    double r0,k,b,ratio,I,D;
    if(!(w < -1./3 && v > -1./3 && proxy<0)) return 1;
    if(model==9 && !(d<=-2*u/proxy || d>=-2./3)) return 1;
    if(model==10 && !(d>=-2*u*r/proxy && d<=2./3)) return 1;
    if(model==9) {r0=de0/dm0; k=-1.5*(2*u+d*B); b=-1.5*d*A;}
    else {r0=dm0/de0; k=1.5*(2*u+d*A); b=1.5*d*B;}
    if(k==0) {
      ratio=r0+b*x;
      if(1+ratio<=0) return 1;
      I=b==0 ? x/(1+r0) : log1p(b*x/(1+r0))/b;
    }
    else {
      ratio=r0*exp(k*x)+b*expm1(k*x)/k;
      if(1+ratio<=0) return 1;
      D=(model==9 && d==-2./3) || (model==10 && d==2./3) ? 0 : 1-b/k;
      if(D==0) I=-expm1(-k*x)/(k*(1+r0));
      else I=-log1p(D*expm1(-k*x)/(1+r0))/(k*D);
    }
    if(model==9) {
      *dm=dm0*exp((-3*(1+v)+1.5*d*B)*x-4.5*d*u*I);
      *de=(*dm)*ratio;
    }
    else {
      *de=de0*exp((-3*(1+w)-1.5*d*A)*x-4.5*d*u*I);
      *dm=(*de)*ratio;
    }
    *qh=1.5*d*(model==9 ? *dm : *de)*(A*(*dm)+B*(*de))/(*dm+*de);
  }
  if(!isfinite(*dm) || !isfinite(*de) || !isfinite(*qh)) return 1;
  if(model>=6 && *dm+*de==0) return 1;
  return 0;
}
#endif
