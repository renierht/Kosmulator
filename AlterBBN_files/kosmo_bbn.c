#include <math.h>
#include <string.h>
#include "src/include.h"

#if defined(_WIN32)
  #define KOSMO_EXPORT __declspec(dllexport)
#else
  #define KOSMO_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    double eta;     // baryon-to-photon ratio
    double Yp;      // 4He mass fraction
    double D_H;     // 2H/H
    double He3_H;   // 3He/H
    double Li7_H;   // 7Li/H
    double Li6_Li7; // 6Li/7Li
    int    status;  // 1=OK, 0=fail
} bbn_result;

// eta_10 = 273.9 * (Ω_b h^2)  =>  eta = 2.739e-8 * (Ω_b h^2)
static inline double eta_from_omegabh2(double Obh2){ return 2.739e-8 * Obh2; }

KOSMO_EXPORT int kosmo_bbn_run(double Omega_b_h2, double N_eff, double tau_n, bbn_result *out){
    if(!out) return -1;

    relicparam p;
    Init_cosmomodel(&p);                  // defaults (standard BBN)

    // Set key physics knobs
    p.eta0 = eta_from_omegabh2(Omega_b_h2);

    // These fields exist in this espensem fork; if your compiler complains, comment out the next two lines.
    p.Nnu  = N_eff;        // effective neutrino number
    p.dNnu = 0.0;          // extra beyond N_eff (keep 0 unless exploring ΔN)

    if(tau_n > 0.0) p.life_neutron = tau_n;

    // Compute central values: err=0. Results in ratioH[].
    double ratioH[64]; memset(ratioH, 0, sizeof(ratioH));
    int ret = nucl(0, p, ratioH);   // NOTE: this fork returns 0 on success

    // Map ratioH[] → our compact struct
    // Indices per AlterBBN tutorial: eta=0, D/H=3, He3/H=5, Yp=6, Li6/H=7, Li7/H=8
    out->eta     = ratioH[0];
    out->Yp      = ratioH[6];
    out->D_H     = ratioH[3];
    out->He3_H   = ratioH[5];
    out->Li7_H   = ratioH[8];
    out->Li6_Li7 = (ratioH[8] > 0.0 ? ratioH[7]/ratioH[8] : 0.0);

    out->status  = (ret == 0) ? 1 : 0;
    return (ret == 0) ? 0 : 1;
}
