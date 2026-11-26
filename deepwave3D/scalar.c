/* Three-dimensional isotropic scalar wave propagator for CPU. */
#ifdef _MSC_VER
#define restrict __restrict
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdbool.h>
#include <stdint.h>
#include "common.h"

#define CAT_I(name, accuracy, dtype) scalar_iso_3d_##accuracy##_##dtype##_##name
#define CAT(name, accuracy, dtype) CAT_I(name, accuracy, dtype)
#define FUNC(name) CAT(name, DW_ACCURACY, DW_DTYPE)

static void add_sources(DW_DTYPE *restrict wf,
                        DW_DTYPE const *restrict amplitudes,
                        int64_t const *restrict locations,
                        int64_t count) {
  int64_t source;
  for(source=0;source<count;++source) wf[locations[source]]+=amplitudes[source];
}

static void record_receivers(DW_DTYPE *restrict amplitudes,
                             DW_DTYPE const *restrict wf,
                             int64_t const *restrict locations,
                             int64_t count) {
  int64_t receiver;
  for(receiver=0;receiver<count;++receiver)
    amplitudes[receiver]=wf[locations[receiver]];
}

#define WFC(dz, dy, dx) wfc[i + (dz)*nynx + (dy)*nx + (dx)]
#define AZ_PSIZ(dz, dy, dx) (az[z+(dz)]*psiz[i+(dz)*nynx+(dy)*nx+(dx)])
#define AY_PSIY(dz, dy, dx) (ay[y+(dy)]*psiy[i+(dz)*nynx+(dy)*nx+(dx)])
#define AX_PSIX(dz, dy, dx) (ax[x+(dx)]*psix[i+(dz)*nynx+(dy)*nx+(dx)])
#define V2DT2_WFC(dz, dy, dx) \
  (v2dt2[j+(dz)*nynx+(dy)*nx+(dx)]*wfc[i+(dz)*nynx+(dy)*nx+(dx)])
#define UT_TERMZ1(dz, dy, dx) \
  (dbzdz[z+(dz)]*((1+bz[z+(dz)])*v2dt2[j+(dz)*nynx]* \
   wfc[i+(dz)*nynx]+bz[z+(dz)]*zetaz[i+(dz)*nynx])+ \
   bz[z+(dz)]*psiz[i+(dz)*nynx])
#define UT_TERMY1(dz, dy, dx) \
  (dbydy[y+(dy)]*((1+by[y+(dy)])*v2dt2[j+(dy)*nx]* \
   wfc[i+(dy)*nx]+by[y+(dy)]*zetay[i+(dy)*nx])+ \
   by[y+(dy)]*psiy[i+(dy)*nx])
#define UT_TERMX1(dz, dy, dx) \
  (dbxdx[x+(dx)]*((1+bx[x+(dx)])*v2dt2[j+(dx)]*wfc[i+(dx)]+ \
   bx[x+(dx)]*zetax[i+(dx)])+bx[x+(dx)]*psix[i+(dx)])
#define UT_TERMZ2(dz, dy, dx) \
  ((1+bz[z+(dz)])*((1+bz[z+(dz)])*v2dt2[j+(dz)*nynx]* \
   wfc[i+(dz)*nynx]+bz[z+(dz)]*zetaz[i+(dz)*nynx]))
#define UT_TERMY2(dz, dy, dx) \
  ((1+by[y+(dy)])*((1+by[y+(dy)])*v2dt2[j+(dy)*nx]* \
   wfc[i+(dy)*nx]+by[y+(dy)]*zetay[i+(dy)*nx]))
#define UT_TERMX2(dz, dy, dx) \
  ((1+bx[x+(dx)])*((1+bx[x+(dx)])*v2dt2[j+(dx)]*wfc[i+(dx)]+ \
   bx[x+(dx)]*zetax[i+(dx)]))
#define PSIZ_TERM(dz, dy, dx) \
  ((1+bz[z+(dz)])*v2dt2[j+(dz)*nynx]*wfc[i+(dz)*nynx]+ \
   bz[z+(dz)]*zetaz[i+(dz)*nynx])
#define PSIY_TERM(dz, dy, dx) \
  ((1+by[y+(dy)])*v2dt2[j+(dy)*nx]*wfc[i+(dy)*nx]+ \
   by[y+(dy)]*zetay[i+(dy)*nx])
#define PSIX_TERM(dz, dy, dx) \
  ((1+bx[x+(dx)])*v2dt2[j+(dx)]*wfc[i+(dx)]+bx[x+(dx)]*zetax[i+(dx)])

static void forward_kernel(
    DW_DTYPE const *restrict v, DW_DTYPE const *restrict wfc,
    DW_DTYPE *restrict wfp, DW_DTYPE const *restrict psiz,
    DW_DTYPE const *restrict psiy, DW_DTYPE const *restrict psix,
    DW_DTYPE *restrict psizn, DW_DTYPE *restrict psiyn,
    DW_DTYPE *restrict psixn, DW_DTYPE *restrict zetaz,
    DW_DTYPE *restrict zetay, DW_DTYPE *restrict zetax,
    DW_DTYPE *restrict dwdv, DW_DTYPE const *restrict az,
    DW_DTYPE const *restrict ay, DW_DTYPE const *restrict ax,
    DW_DTYPE const *restrict bz, DW_DTYPE const *restrict by,
    DW_DTYPE const *restrict bx, DW_DTYPE const *restrict dbzdz,
    DW_DTYPE const *restrict dbydy, DW_DTYPE const *restrict dbxdx,
    DW_DTYPE rdz, DW_DTYPE rdy, DW_DTYPE rdx,
    DW_DTYPE rdz2, DW_DTYPE rdy2, DW_DTYPE rdx2, DW_DTYPE dt2,
    int64_t shot, int64_t nz, int64_t ny, int64_t nx,
    bool save_dwdv, int64_t pml_z0, int64_t pml_z1,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1) {
  int64_t const nynx=ny*nx, nxyz=nz*nynx;
  int64_t z,y,x;
  for(z=FD_PAD;z<nz-FD_PAD;++z) for(y=FD_PAD;y<ny-FD_PAD;++y)
    for(x=FD_PAD;x<nx-FD_PAD;++x) {
      int64_t const j=z*nynx+y*nx+x, i=shot*nxyz+j;
      bool const pz=z<pml_z0||z>=pml_z1;
      bool const py=y<pml_y0||y>=pml_y1;
      bool const px=x<pml_x0||x>=pml_x1;
      DW_DTYPE sum;
      if(!pz) sum=DIFFZ2(WFC); else {
        DW_DTYPE const d=DIFFZ1(WFC);
        DW_DTYPE const tmp=(1+bz[z])*DIFFZ2(WFC)+dbzdz[z]*d+DIFFZ1(AZ_PSIZ);
        sum=(1+bz[z])*tmp+az[z]*zetaz[i];
        psizn[i]=bz[z]*d+az[z]*psiz[i];
        zetaz[i]=bz[z]*tmp+az[z]*zetaz[i];
      }
      if(!py) sum+=DIFFY2(WFC); else {
        DW_DTYPE const d=DIFFY1(WFC);
        DW_DTYPE const tmp=(1+by[y])*DIFFY2(WFC)+dbydy[y]*d+DIFFY1(AY_PSIY);
        sum+=(1+by[y])*tmp+ay[y]*zetay[i];
        psiyn[i]=by[y]*d+ay[y]*psiy[i];
        zetay[i]=by[y]*tmp+ay[y]*zetay[i];
      }
      if(!px) sum+=DIFFX2(WFC); else {
        DW_DTYPE const d=DIFFX1(WFC);
        DW_DTYPE const tmp=(1+bx[x])*DIFFX2(WFC)+dbxdx[x]*d+DIFFX1(AX_PSIX);
        sum+=(1+bx[x])*tmp+ax[x]*zetax[i];
        psixn[i]=bx[x]*d+ax[x]*psix[i];
        zetax[i]=bx[x]*tmp+ax[x]*zetax[i];
      }
      if(save_dwdv) dwdv[j]=2*v[j]*dt2*sum;
      wfp[i]=v[j]*v[j]*dt2*sum+2*wfc[i]-wfp[i];
    }
}

static void backward_kernel(
    DW_DTYPE const *restrict v2dt2, DW_DTYPE const *restrict wfc,
    DW_DTYPE *restrict wfp, DW_DTYPE const *restrict psiz,
    DW_DTYPE const *restrict psiy, DW_DTYPE const *restrict psix,
    DW_DTYPE *restrict psizn, DW_DTYPE *restrict psiyn,
    DW_DTYPE *restrict psixn, DW_DTYPE *restrict zetaz,
    DW_DTYPE *restrict zetay, DW_DTYPE *restrict zetax,
    DW_DTYPE *restrict zetazn, DW_DTYPE *restrict zetayn,
    DW_DTYPE *restrict zetaxn, DW_DTYPE const *restrict dwdv,
    DW_DTYPE *restrict grad_v, DW_DTYPE const *restrict az,
    DW_DTYPE const *restrict ay, DW_DTYPE const *restrict ax,
    DW_DTYPE const *restrict bz, DW_DTYPE const *restrict by,
    DW_DTYPE const *restrict bx, DW_DTYPE const *restrict dbzdz,
    DW_DTYPE const *restrict dbydy, DW_DTYPE const *restrict dbxdx,
    DW_DTYPE rdz, DW_DTYPE rdy, DW_DTYPE rdx,
    DW_DTYPE rdz2, DW_DTYPE rdy2, DW_DTYPE rdx2,
    int64_t shot, int64_t nz, int64_t ny, int64_t nx,
    int64_t step_ratio, bool accumulate_grad,
    int64_t pml_z0, int64_t pml_z1, int64_t pml_y0, int64_t pml_y1,
    int64_t pml_x0, int64_t pml_x1) {
  int64_t const nynx=ny*nx, nxyz=nz*nynx;
  int64_t z,y,x;
  for(z=FD_PAD;z<nz-FD_PAD;++z) for(y=FD_PAD;y<ny-FD_PAD;++y)
    for(x=FD_PAD;x<nx-FD_PAD;++x) {
      int64_t const j=z*nynx+y*nx+x, i=shot*nxyz+j;
      bool const pz=z<pml_z0||z>=pml_z1;
      bool const py=y<pml_y0||y>=pml_y1;
      bool const px=x<pml_x0||x>=pml_x1;
      wfp[i]=(pz ? -DIFFZ1(UT_TERMZ1)+DIFFZ2(UT_TERMZ2) : DIFFZ2(V2DT2_WFC))+
             (py ? -DIFFY1(UT_TERMY1)+DIFFY2(UT_TERMY2) : DIFFY2(V2DT2_WFC))+
             (px ? -DIFFX1(UT_TERMX1)+DIFFX2(UT_TERMX2) : DIFFX2(V2DT2_WFC))+
             2*wfc[i]-wfp[i];
      if(pz) { psizn[i]=-az[z]*DIFFZ1(PSIZ_TERM)+az[z]*psiz[i];
               zetazn[i]=az[z]*v2dt2[j]*wfc[i]+az[z]*zetaz[i]; }
      if(py) { psiyn[i]=-ay[y]*DIFFY1(PSIY_TERM)+ay[y]*psiy[i];
               zetayn[i]=ay[y]*v2dt2[j]*wfc[i]+ay[y]*zetay[i]; }
      if(px) { psixn[i]=-ax[x]*DIFFX1(PSIX_TERM)+ax[x]*psix[i];
               zetaxn[i]=ax[x]*v2dt2[j]*wfc[i]+ax[x]*zetax[i]; }
      if(accumulate_grad) grad_v[j]+=wfc[i]*dwdv[j]*(DW_DTYPE)step_ratio;
    }
}

static void combine_grad_3d(DW_DTYPE *restrict grad,
    DW_DTYPE const *restrict per_thread, int64_t n_threads,
    int64_t nz, int64_t ny, int64_t nx) {
  int64_t const nxyz=nz*ny*nx; int64_t z,y,x,t;
  for(z=FD_PAD;z<nz-FD_PAD;++z) for(y=FD_PAD;y<ny-FD_PAD;++y)
    for(x=FD_PAD;x<nx-FD_PAD;++x) {
      int64_t const i=z*ny*nx+y*nx+x;
      for(t=0;t<n_threads;++t) grad[i]+=per_thread[t*nxyz+i];
    }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void FUNC(forward)(
    DW_DTYPE const *restrict v, DW_DTYPE const *restrict f,
    DW_DTYPE *restrict wfc, DW_DTYPE *restrict wfp,
    DW_DTYPE *restrict psiz, DW_DTYPE *restrict psiy, DW_DTYPE *restrict psix,
    DW_DTYPE *restrict psizn, DW_DTYPE *restrict psiyn, DW_DTYPE *restrict psixn,
    DW_DTYPE *restrict zetaz, DW_DTYPE *restrict zetay, DW_DTYPE *restrict zetax,
    DW_DTYPE *restrict dwdv, DW_DTYPE *restrict r,
    DW_DTYPE const *restrict az, DW_DTYPE const *restrict ay,
    DW_DTYPE const *restrict ax, DW_DTYPE const *restrict bz,
    DW_DTYPE const *restrict by, DW_DTYPE const *restrict bx,
    DW_DTYPE const *restrict dbzdz, DW_DTYPE const *restrict dbydy,
    DW_DTYPE const *restrict dbxdx, int64_t const *restrict sources_i,
    int64_t const *restrict receivers_i,
    DW_DTYPE rdz, DW_DTYPE rdy, DW_DTYPE rdx,
    DW_DTYPE rdz2, DW_DTYPE rdy2, DW_DTYPE rdx2, DW_DTYPE dt2,
    int64_t nt, int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t n_sources, int64_t n_receivers, int64_t step_ratio,
    bool v_requires_grad, int64_t pml_z0, int64_t pml_z1,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t n_threads) {
  int64_t const nxyz=nz*ny*nx; int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(shot=0;shot<n_shots;++shot) {
    int64_t const wi=shot*nxyz, si=shot*n_sources, ri=shot*n_receivers;
    int64_t t;
    for(t=0;t<nt;++t) {
      DW_DTYPE *const dw=v_requires_grad
          ? dwdv+shot*(nt/step_ratio)*nxyz+(t/step_ratio)*nxyz : dwdv;
      if(t&1) {
        forward_kernel(v,wfp,wfc,psizn,psiyn,psixn,psiz,psiy,psix,
          zetaz,zetay,zetax,dw,az,ay,ax,bz,by,bx,dbzdz,dbydy,dbxdx,
          rdz,rdy,rdx,rdz2,rdy2,rdx2,dt2,shot,nz,ny,nx,
          v_requires_grad&&t%step_ratio==0,pml_z0,pml_z1,pml_y0,pml_y1,pml_x0,pml_x1);
        if(n_sources) add_sources(wfc+wi,f+si*nt+t*n_sources,sources_i+si,n_sources);
        if(n_receivers) record_receivers(r+ri*nt+t*n_receivers,wfp+wi,receivers_i+ri,n_receivers);
      } else {
        forward_kernel(v,wfc,wfp,psiz,psiy,psix,psizn,psiyn,psixn,
          zetaz,zetay,zetax,dw,az,ay,ax,bz,by,bx,dbzdz,dbydy,dbxdx,
          rdz,rdy,rdx,rdz2,rdy2,rdx2,dt2,shot,nz,ny,nx,
          v_requires_grad&&t%step_ratio==0,pml_z0,pml_z1,pml_y0,pml_y1,pml_x0,pml_x1);
        if(n_sources) add_sources(wfp+wi,f+si*nt+t*n_sources,sources_i+si,n_sources);
        if(n_receivers) record_receivers(r+ri*nt+t*n_receivers,wfc+wi,receivers_i+ri,n_receivers);
      }
    }
  }
}

#ifdef _WIN32
__declspec(dllexport)
#endif
void FUNC(backward)(
    DW_DTYPE const *restrict v2dt2, DW_DTYPE const *restrict grad_r,
    DW_DTYPE *restrict wfc, DW_DTYPE *restrict wfp,
    DW_DTYPE *restrict psiz, DW_DTYPE *restrict psiy, DW_DTYPE *restrict psix,
    DW_DTYPE *restrict psizn, DW_DTYPE *restrict psiyn, DW_DTYPE *restrict psixn,
    DW_DTYPE *restrict zetaz, DW_DTYPE *restrict zetay, DW_DTYPE *restrict zetax,
    DW_DTYPE *restrict zetazn, DW_DTYPE *restrict zetayn, DW_DTYPE *restrict zetaxn,
    DW_DTYPE const *restrict dwdv, DW_DTYPE *restrict grad_f,
    DW_DTYPE *restrict grad_v, DW_DTYPE *restrict grad_v_thread,
    DW_DTYPE const *restrict az, DW_DTYPE const *restrict ay,
    DW_DTYPE const *restrict ax, DW_DTYPE const *restrict bz,
    DW_DTYPE const *restrict by, DW_DTYPE const *restrict bx,
    DW_DTYPE const *restrict dbzdz, DW_DTYPE const *restrict dbydy,
    DW_DTYPE const *restrict dbxdx, int64_t const *restrict sources_i,
    int64_t const *restrict receivers_i,
    DW_DTYPE rdz, DW_DTYPE rdy, DW_DTYPE rdx,
    DW_DTYPE rdz2, DW_DTYPE rdy2, DW_DTYPE rdx2,
    int64_t nt, int64_t n_shots, int64_t nz, int64_t ny, int64_t nx,
    int64_t n_sources, int64_t n_receivers, int64_t step_ratio,
    bool v_requires_grad, int64_t pml_z0, int64_t pml_z1,
    int64_t pml_y0, int64_t pml_y1, int64_t pml_x0, int64_t pml_x1,
    int64_t n_threads) {
  int64_t const nxyz=nz*ny*nx; int64_t shot;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads)
#endif
  for(shot=0;shot<n_shots;++shot) {
    int64_t const wi=shot*nxyz, si=shot*n_sources, ri=shot*n_receivers;
#ifdef _OPENMP
    int64_t const gi=omp_get_thread_num()*nxyz;
#else
    int64_t const gi=0;
#endif
    int64_t t;
    for(t=nt-1;t>=0;--t) {
      DW_DTYPE const *const dw=v_requires_grad
          ? dwdv+shot*(nt/step_ratio)*nxyz+(t/step_ratio)*nxyz : dwdv;
      if((nt-1-t)&1) {
        if(n_sources) record_receivers(grad_f+si*nt+t*n_sources,wfp+wi,sources_i+si,n_sources);
        backward_kernel(v2dt2,wfp,wfc,psizn,psiyn,psixn,psiz,psiy,psix,
          zetazn,zetayn,zetaxn,zetaz,zetay,zetax,dw,grad_v_thread+gi,
          az,ay,ax,bz,by,bx,dbzdz,dbydy,dbxdx,rdz,rdy,rdx,rdz2,rdy2,rdx2,
          shot,nz,ny,nx,step_ratio,v_requires_grad&&t%step_ratio==0,
          pml_z0,pml_z1,pml_y0,pml_y1,pml_x0,pml_x1);
        if(n_receivers) add_sources(wfc+wi,grad_r+ri*nt+t*n_receivers,receivers_i+ri,n_receivers);
      } else {
        if(n_sources) record_receivers(grad_f+si*nt+t*n_sources,wfc+wi,sources_i+si,n_sources);
        backward_kernel(v2dt2,wfc,wfp,psiz,psiy,psix,psizn,psiyn,psixn,
          zetaz,zetay,zetax,zetazn,zetayn,zetaxn,dw,grad_v_thread+gi,
          az,ay,ax,bz,by,bx,dbzdz,dbydy,dbxdx,rdz,rdy,rdx,rdz2,rdy2,rdx2,
          shot,nz,ny,nx,step_ratio,v_requires_grad&&t%step_ratio==0,
          pml_z0,pml_z1,pml_y0,pml_y1,pml_x0,pml_x1);
        if(n_receivers) add_sources(wfp+wi,grad_r+ri*nt+t*n_receivers,receivers_i+ri,n_receivers);
      }
    }
  }
#ifdef _OPENMP
  if(v_requires_grad&&n_threads>1)
    combine_grad_3d(grad_v,grad_v_thread,n_threads,nz,ny,nx);
#endif
}
