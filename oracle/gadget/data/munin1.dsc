belief network "unknown"
node R_LNLT1_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLT1_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_LNLT1_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_LNLT1_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNLT1_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_LNLT1_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node DIFFN_M_SEV_PROX {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node DIFFN_TYPE {
  type : discrete [ 3 ] = { "MOTOR", "MIXED", "SENS" };
}
node DIFFN_SEV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node DIFFN_MOT_SEV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node DIFFN_DISTR {
  type : discrete [ 3 ] = { "DIST", "PROX", "RANDOM" };
}
node DIFFN_SENS_SEV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLLP_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLLP_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_LNLLP_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_LNLLP_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNLW_MED_TIME {
  type : discrete [ 4 ] = { "ACUTE", "SUBACUTE", "CHRONIC", "OLD" };
}
node DIFFN_TIME {
  type : discrete [ 4 ] = { "ACUTE", "SUBACUTE", "CHRONIC", "OLD" };
}
node DIFFN_M_SEV_DIST {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLBE_MED_TIME {
  type : discrete [ 4 ] = { "ACUTE", "SUBACUTE", "CHRONIC", "OLD" };
}
node R_LNLLP_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_LNLLP_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node DIFFN_PATHO {
  type : discrete [ 5 ] = { "DEMY", "BLOCK", "AXONAL", "V_E_REIN", "E_REIN" };
}
node DIFFN_S_SEV_DIST {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MYDY_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MYOP_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLW_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_DIFFN_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLBE_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLT1_LP_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLW_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_DIFFN_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_LNLBE_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_LNLT1_LP_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_MYDY_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_MYOP_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_LNLW_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_DIFFN_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_LNLBE_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_LNLT1_LP_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_MYDY_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_MYOP_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNLW_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_DIFFN_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNLBE_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNLT1_LP_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_MYOP_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_MYDY_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_LNLW_MEDD2_DISP_WD {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_DIFFN_MEDD2_SALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLW_MEDD2_SALOSS_WD {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_NMT_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MYOP_MYDY_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_DIFFN_LNLW_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLT1_LP_BE_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MYAS_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_MYOP_MYDY_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_DIFFN_LNLW_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_LNLT1_LP_BE_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_DIFFN_LNLW_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNLT1_LP_BE_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_DIFFN_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_LNLW_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_DIFFN_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLW_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLT1_LP_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_LNLBE_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_LNLBE_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLT1_LP_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLT1_LP_BE_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_DIFFN_LNLW_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_LNL_DIFFN_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MUSCLE_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_DIFFN_LNLW_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_LNLT1_LP_BE_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_MUSCLE_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_LNL_DIFFN_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_MYOP_MYDY_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNL_DIFFN_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_LNLW_MED_SEV {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLW_MED_PATHO {
  type : discrete [ 5 ] = { "DEMY", "BLOCK", "AXONAL", "V_E_REIN", "E_REIN" };
}
node R_LNLBE_MED_DIFSLOW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLW_MED_BLOCK {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_MYAS_APB_NMT {
  type : discrete [ 7 ] = { "NO", "MOD_PRE", "SEV_PRE", "MLD_POST", "MOD_POST", "SEV_POST", "MIXED" };
}
node R_DE_REGEN_APB_NMT {
  type : discrete [ 7 ] = { "NO", "MOD_PRE", "SEV_PRE", "MLD_POST", "MOD_POST", "SEV_POST", "MIXED" };
}
node R_LNL_DIFFN_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_MYOP_MYDY_APB_MUSIZE {
  type : discrete [ 6 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE" };
}
node R_DIFFN_LNLW_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLT1_LP_BE_APB_MALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_DIFFN_MED_DIFSLOW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLBE_MED_SEV {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLBE_MED_PATHO {
  type : discrete [ 5 ] = { "DEMY", "BLOCK", "AXONAL", "V_E_REIN", "E_REIN" };
}
node R_LNLBE_MED_BLOCK {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_DIFFN_MED_BLOCK {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLW_MEDD2_RD_WD {
  type : discrete [ 3 ] = { "NO", "MOD", "SEV" };
}
node R_LNLW_MEDD2_LD_WD {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLBE_MEDD2_DIFSLOW_WD {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLW_MEDD2_BLOCK_WD {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_DIFFN_LNLW_MEDD2_DISP_WD {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLBE_MEDD2_SALOSS_EW {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_DIFFN_LNLW_MEDD2_SALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_DIFFN_MEDD2_DIFSLOW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLBE_MEDD2_RD_EW {
  type : discrete [ 3 ] = { "NO", "MOD", "SEV" };
}
node R_LNLBE_MEDD2_LD_EW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_LNLBE_MEDD2_BLOCK_EW {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_DIFFN_MEDD2_BLOCK {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_LNLBE_MEDD2_DISP_EW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_DIFFN_MEDD2_DISP {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MED_RD_WA {
  type : discrete [ 3 ] = { "NO", "MOD", "SEV" };
}
node R_MED_LD_WA {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MED_DIFSLOW_WA {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MED_BLOCK_WA {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_MED_DIFSLOW_EW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MED_RD_EW {
  type : discrete [ 3 ] = { "NO", "MOD", "SEV" };
}
node R_MED_LD_EW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MEDD2_RD_WD {
  type : discrete [ 3 ] = { "NO", "MOD", "SEV" };
}
node R_MEDD2_LD_WD {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MEDD2_DIFSLOW_WD {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MEDD2_BLOCK_WD {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_MEDD2_DIFSLOW_EW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MEDD2_SALOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_MEDD2_RD_EW {
  type : discrete [ 3 ] = { "NO", "MOD", "SEV" };
}
node R_MEDD2_LD_EW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MED_DCV_WA {
  type : discrete [ 9 ] = { "M_S60", "M_S52", "M_S44", "M_S36", "M_S28", "M_S20", "M_S14", "M_S08", "M_S00" };
}
node R_MED_RDLDDEL {
  type : discrete [ 5 ] = { "MS3_1", "MS3_9", "MS4_7", "MS10_1", "MS20_1" };
}
node R_MED_RDLDCV_EW {
  type : discrete [ 6 ] = { "M_S60", "M_S52", "M_S44", "M_S27", "M_S15", "M_S07" };
}
node R_MED_DCV_EW {
  type : discrete [ 10 ] = { "M_S60", "M_S56", "M_S52", "M_S44", "M_S36", "M_S28", "M_S20", "M_S14", "M_S08", "M_S00" };
}
node R_MEDD2_DSLOW_EW {
  type : discrete [ 9 ] = { "M_S60", "M_S52", "M_S44", "M_S36", "M_S28", "M_S20", "M_S14", "M_S08", "M_S00" };
}
node R_MEDD2_LSLOW_EW {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "V_SEV" };
}
node R_MEDD2_DSLOW_WD {
  type : discrete [ 9 ] = { "M_S60", "M_S52", "M_S44", "M_S36", "M_S28", "M_S20", "M_S14", "M_S08", "M_S00" };
}
node R_MEDD2_LSLOW_WD {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "V_SEV" };
}
node R_MEDD2_EFFAXLOSS {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_MEDD2_DISP_EW {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_MEDD2_DISP_WD {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_APB_SPONT_INS_ACT {
  type : discrete [ 2 ] = { "NORMAL", "INCR" };
}
node R_APB_SPONT_HF_DISCH {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_APB_DENERV {
  type : discrete [ 4 ] = { "NO", "MILD", "MOD", "SEV" };
}
node R_APB_SPONT_DENERV_ACT {
  type : discrete [ 4 ] = { "NO", "SOME", "MOD", "ABUNDANT" };
}
node R_APB_NEUR_ACT {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_APB_SPONT_NEUR_DISCH {
  type : discrete [ 6 ] = { "NO", "FASCIC", "NEUROMYO", "MYOKYMIA", "TETANUS", "OTHER" };
}
node R_APB_MUDENS {
  type : discrete [ 3 ] = { "NORMAL", "INCR", "V_INCR" };
}
node R_APB_SF_DENSITY {
  type : discrete [ 3 ] = { "__2SD", "2_4SD", "__4SD" };
}
node R_APB_SF_JITTER {
  type : discrete [ 4 ] = { "NORMAL", "2_5", "5_10", "__10" };
}
node R_APB_REPSTIM_POST_DECR {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "INCON" };
}
node R_APB_REPSTIM_FACILI {
  type : discrete [ 4 ] = { "NO", "MOD", "SEV", "REDUCED" };
}
node R_APB_REPSTIM_DECR {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "INCON" };
}
node R_APB_REPSTIM_CMAPAMP {
  type : discrete [ 21 ] = { "MV_000", "MV_032", "MV_044", "MV_063", "MV_088", "MV_13", "MV_18", "MV_25", "MV_35", "MV_5", "MV_71", "MV1", "MV1_4", "MV2", "MV2_8", "MV4", "MV5_6", "MV8", "MV11_3", "MV16", "MV22_6" };
}
node R_APB_NMT {
  type : discrete [ 7 ] = { "NO", "MOD_PRE", "SEV_PRE", "MLD_POST", "MOD_POST", "SEV_POST", "MIXED" };
}
node R_APB_MUPINSTAB {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_APB_DE_REGEN {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_APB_MUPSATEL {
  type : discrete [ 2 ] = { "NO", "YES" };
}
node R_APB_QUAN_MUPPOLY {
  type : discrete [ 3 ] = { "__12_", "12_24_", "__24_" };
}
node R_APB_QUAL_MUPPOLY {
  type : discrete [ 2 ] = { "NORMAL", "INCR" };
}
node R_APB_QUAL_MUPDUR {
  type : discrete [ 3 ] = { "SMALL", "NORMAL", "INCR" };
}
node R_APB_MUPDUR {
  type : discrete [ 7 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE", "OTHER" };
}
node R_APB_QUAN_MUPDUR {
  type : discrete [ 19 ] = { "MS3", "MS4", "MS5", "MS6", "MS7", "MS8", "MS9", "MS10", "MS11", "MS12", "MS13", "MS14", "MS15", "MS16", "MS17", "MS18", "MS19", "MS20", "MS_20" };
}
node R_APB_QUAL_MUPAMP {
  type : discrete [ 5 ] = { "V_RED", "REDUCED", "NORMAL", "INCR", "V_INCR" };
}
node R_APB_MUPAMP {
  type : discrete [ 7 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE", "OTHER" };
}
node R_APB_QUAN_MUPAMP {
  type : discrete [ 20 ] = { "UV34", "UV44", "UV58", "UV74", "UV94", "UV122", "UV156", "UV200", "UV260", "UV330", "UV420", "UV540", "UV700", "UV900", "UV1150", "UV1480", "UV1900", "UV2440", "UV3130", "UV4020" };
}
node R_APB_TA_CONCL {
  type : discrete [ 6 ] = { "__5ABOVE", "2_5ABOVE", "NORMAL", "2_5BELOW", "__5BELOW", "OTHER" };
}
node R_APB_EFFMUS {
  type : discrete [ 7 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE", "OTHER" };
}
node R_APB_MVA_AMP {
  type : discrete [ 3 ] = { "INCR", "NORMAL", "REDUCED" };
}
node R_APB_MULOSS {
  type : discrete [ 6 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL", "OTHER" };
}
node R_APB_MVA_RECRUIT {
  type : discrete [ 4 ] = { "FULL", "REDUCED", "DISCRETE", "NO_UNITS" };
}
node R_APB_MALOSS {
  type : discrete [ 6 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL", "OTHER" };
}
node R_APB_MUSIZE {
  type : discrete [ 7 ] = { "V_SMALL", "SMALL", "NORMAL", "INCR", "LARGE", "V_LARGE", "OTHER" };
}
node R_APB_MUSCLE_VOL {
  type : discrete [ 2 ] = { "ATROPHIC", "NORMAL" };
}
node R_APB_VOL_ACT {
  type : discrete [ 4 ] = { "NORMAL", "REDUCED", "V_RED", "ABSENT" };
}
node R_APB_FORCE {
  type : discrete [ 6 ] = { "5", "4", "3", "2", "1", "0" };
}
node R_MED_ALLDEL_WA {
  type : discrete [ 9 ] = { "MS0_0", "MS0_4", "MS0_8", "MS1_6", "MS3_2", "MS6_4", "MS12_8", "MS25_6", "INFIN" };
}
node R_MED_LAT_WA {
  type : discrete [ 19 ] = { "MS2_3", "MS2_7", "MS3_1", "MS3_5", "MS3_9", "MS4_3", "MS4_7", "MS5_3", "MS5_9", "MS6_5", "MS7_1", "MS8_0", "MS9_0", "MS10_0", "MS12_0", "MS14_0", "MS16_0", "MS18_0", "INFIN" };
}
node R_APB_ALLAMP_WA {
  type : discrete [ 9 ] = { "ZERO", "A0_01", "A0_10", "A0_30", "A0_70", "A1_00", "A2_00", "A4_00", "A8_00" };
}
node R_MED_AMP_WA {
  type : discrete [ 17 ] = { "MV_000", "MV_13", "MV_18", "MV_25", "MV_35", "MV_5", "MV_71", "MV1", "MV1_4", "MV2", "MV2_8", "MV4", "MV5_6", "MV8", "MV11_3", "MV16", "MV22_6" };
}
node R_MED_ALLCV_EW {
  type : discrete [ 10 ] = { "M_S60", "M_S56", "M_S52", "M_S44", "M_S36", "M_S28", "M_S20", "M_S14", "M_S08", "M_S00" };
}
node R_MED_CV_EW {
  type : discrete [ 19 ] = { "M_S00", "M_S04", "M_S08", "M_S12", "M_S16", "M_S20", "M_S24", "M_S28", "M_S32", "M_S36", "M_S40", "M_S44", "M_S48", "M_S52", "M_S56", "M_S60", "M_S64", "M_S68", "M_S72" };
}
node R_MED_BLOCK_EW {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_MED_AMPR_EW {
  type : discrete [ 12 ] = { "R_1_1", "R1_0", "R0_9", "R0_8", "R0_7", "R0_6", "R0_5", "R0_4", "R0_3", "R0_2", "R0_1", "R0_0" };
}
node R_MEDD2_ALLCV_WD {
  type : discrete [ 9 ] = { "M_S60", "M_S52", "M_S44", "M_S36", "M_S28", "M_S20", "M_S14", "M_S08", "M_S00" };
}
node R_MEDD2_CV_WD {
  type : discrete [ 19 ] = { "M_S00", "M_S04", "M_S08", "M_S12", "M_S16", "M_S20", "M_S24", "M_S28", "M_S32", "M_S36", "M_S40", "M_S44", "M_S48", "M_S52", "M_S56", "M_S60", "M_S64", "M_S68", "M_S_72" };
}
node R_MEDD2_ALLAMP_WD {
  type : discrete [ 6 ] = { "ZERO", "A0_01", "A0_10", "A0_30", "A0_70", "A1_00" };
}
node R_MEDD2_AMP_WD {
  type : discrete [ 15 ] = { "UV_0_63", "UV0_88", "UV1_25", "UV1_77", "UV2_50", "UV3_50", "UV5_00", "UV7_10", "UV10_0", "UV14_0", "UV20_0", "UV28_0", "UV40_0", "UV57_0", "UV_80_0" };
}
node R_MEDD2_ALLCV_EW {
  type : discrete [ 9 ] = { "M_S60", "M_S52", "M_S44", "M_S36", "M_S28", "M_S20", "M_S14", "M_S08", "M_S00" };
}
node R_MEDD2_CV_EW {
  type : discrete [ 20 ] = { "M_S00", "M_S04", "M_S08", "M_S12", "M_S16", "M_S20", "M_S24", "M_S28", "M_S32", "M_S36", "M_S40", "M_S44", "M_S48", "M_S52", "M_S56", "M_S60", "M_S64", "M_S68", "M_S72", "M_S_76" };
}
node R_MEDD2_BLOCK_EW {
  type : discrete [ 5 ] = { "NO", "MILD", "MOD", "SEV", "TOTAL" };
}
node R_MEDD2_DISP_EWD {
  type : discrete [ 9 ] = { "R0_15", "R0_25", "R0_35", "R0_45", "R0_55", "R0_65", "R0_75", "R0_85", "R0_95" };
}
node R_MEDD2_AMPR_EW {
  type : discrete [ 12 ] = { "R0_0", "R0_1", "R0_2", "R0_3", "R0_4", "R0_5", "R0_6", "R0_7", "R0_8", "R0_9", "R1_0", "R_1_1" };
}
probability ( R_LNLT1_APB_DENERV ) {
   1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLT1_APB_NEUR_ACT ) {
   1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLT1_APB_MUDENS ) {
   1.0, 0.0, 0.0;
}
probability ( R_LNLT1_APB_DE_REGEN ) {
   1.0, 0.0;
}
probability ( R_LNLT1_APB_MUSIZE ) {
   0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLT1_APB_MALOSS ) {
   1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( DIFFN_M_SEV_PROX | DIFFN_MOT_SEV, DIFFN_DISTR ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.5, 0.5, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0;
  (3, 1) : 0.0, 0.5, 0.5, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.25, 0.45, 0.25, 0.05;
  (2, 2) : 0.1, 0.4, 0.4, 0.1;
  (3, 2) : 0.05, 0.15, 0.40, 0.40;
}
probability ( DIFFN_TYPE ) {
   0.060, 0.935, 0.005;
}
probability ( DIFFN_SEV ) {
   0.78, 0.10, 0.07, 0.05;
}
probability ( DIFFN_MOT_SEV | DIFFN_SEV, DIFFN_TYPE ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0;
}
probability ( DIFFN_DISTR ) {
   0.93, 0.02, 0.05;
}
probability ( DIFFN_SENS_SEV | DIFFN_SEV, DIFFN_TYPE ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0;
  (2, 0) : 1.0, 0.0, 0.0, 0.0;
  (3, 0) : 1.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLLP_APB_DENERV ) {
   1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLLP_APB_NEUR_ACT ) {
   1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLLP_APB_MUDENS ) {
   1.0, 0.0, 0.0;
}
probability ( R_LNLLP_APB_DE_REGEN ) {
   1.0, 0.0;
}
probability ( R_LNLW_MED_TIME ) {
   0.05, 0.33, 0.60, 0.02;
}
probability ( DIFFN_TIME ) {
   0.01, 0.25, 0.65, 0.09;
}
probability ( DIFFN_M_SEV_DIST | DIFFN_MOT_SEV, DIFFN_DISTR ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 1.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.5, 0.5, 0.0, 0.0;
  (3, 1) : 0.0, 0.5, 0.5, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.25, 0.45, 0.25, 0.05;
  (2, 2) : 0.1, 0.4, 0.4, 0.1;
  (3, 2) : 0.05, 0.15, 0.40, 0.40;
}
probability ( R_LNLBE_MED_TIME ) {
   0.05, 0.60, 0.30, 0.05;
}
probability ( R_LNLLP_APB_MUSIZE ) {
   0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLLP_APB_MALOSS ) {
   1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( DIFFN_PATHO ) {
   0.086, 0.010, 0.900, 0.002, 0.002;
}
probability ( DIFFN_S_SEV_DIST | DIFFN_SENS_SEV, DIFFN_DISTR ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 1.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.5, 0.5, 0.0, 0.0;
  (3, 1) : 0.0, 0.5, 0.5, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.25, 0.45, 0.25, 0.05;
  (2, 2) : 0.1, 0.4, 0.4, 0.1;
  (3, 2) : 0.05, 0.15, 0.40, 0.40;
}
probability ( R_MYDY_APB_DENERV ) {
   1.0, 0.0, 0.0, 0.0;
}
probability ( R_MYOP_APB_DENERV ) {
   1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLW_APB_DENERV | R_LNLW_MED_SEV, R_LNLW_MED_TIME, R_LNLW_MED_PATHO ) {
  (0, 0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (4, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 1, 0) : 0.3, 0.5, 0.2, 0.0;
  (3, 1, 0) : 0.1, 0.5, 0.4, 0.0;
  (4, 1, 0) : 0.0, 0.4, 0.4, 0.2;
  (0, 2, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 2, 0) : 0.3, 0.5, 0.2, 0.0;
  (3, 2, 0) : 0.1, 0.5, 0.4, 0.0;
  (4, 2, 0) : 0.0, 0.4, 0.4, 0.2;
  (0, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (3, 3, 0) : 0.5, 0.4, 0.1, 0.0;
  (4, 3, 0) : 0.10, 0.60, 0.25, 0.05;
  (0, 0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (4, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 1) : 0.8, 0.2, 0.0, 0.0;
  (2, 1, 1) : 0.6, 0.4, 0.0, 0.0;
  (3, 1, 1) : 0.4, 0.5, 0.1, 0.0;
  (4, 1, 1) : 0.3, 0.4, 0.3, 0.0;
  (0, 2, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 1) : 0.9, 0.1, 0.0, 0.0;
  (2, 2, 1) : 0.6, 0.4, 0.0, 0.0;
  (3, 2, 1) : 0.4, 0.5, 0.1, 0.0;
  (4, 2, 1) : 0.3, 0.4, 0.3, 0.0;
  (0, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (3, 3, 1) : 0.5, 0.5, 0.0, 0.0;
  (4, 3, 1) : 0.5, 0.5, 0.0, 0.0;
  (0, 0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (4, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 2) : 0.0, 1.0, 0.0, 0.0;
  (2, 1, 2) : 0.0, 0.0, 1.0, 0.0;
  (3, 1, 2) : 0.0, 0.0, 0.5, 0.5;
  (4, 1, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 2, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 2) : 0.0, 1.0, 0.0, 0.0;
  (2, 2, 2) : 0.0, 0.0, 1.0, 0.0;
  (3, 2, 2) : 0.0, 0.0, 0.5, 0.5;
  (4, 2, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 2) : 0.9, 0.1, 0.0, 0.0;
  (3, 3, 2) : 0.6, 0.3, 0.1, 0.0;
  (4, 3, 2) : 0.45, 0.45, 0.10, 0.00;
  (0, 0, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 1, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 2, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 3, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 0, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 1, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 2, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 3, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 3, 4) : 0.05, 0.40, 0.50, 0.05;
}
probability ( R_DIFFN_APB_DENERV | DIFFN_M_SEV_DIST, DIFFN_TIME, DIFFN_PATHO ) {
  (0, 0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 1, 0) : 0.3, 0.5, 0.2, 0.0;
  (3, 1, 0) : 0.1, 0.5, 0.4, 0.0;
  (0, 2, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 2, 0) : 0.3, 0.5, 0.2, 0.0;
  (3, 2, 0) : 0.1, 0.5, 0.4, 0.0;
  (0, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (3, 3, 0) : 0.5, 0.4, 0.1, 0.0;
  (0, 0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 1) : 0.9, 0.1, 0.0, 0.0;
  (2, 1, 1) : 0.6, 0.4, 0.0, 0.0;
  (3, 1, 1) : 0.4, 0.5, 0.1, 0.0;
  (0, 2, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 1) : 0.9, 0.1, 0.0, 0.0;
  (2, 2, 1) : 0.6, 0.4, 0.0, 0.0;
  (3, 2, 1) : 0.4, 0.5, 0.1, 0.0;
  (0, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (3, 3, 1) : 0.5, 0.5, 0.0, 0.0;
  (0, 0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 2) : 0.0, 1.0, 0.0, 0.0;
  (2, 1, 2) : 0.0, 0.0, 1.0, 0.0;
  (3, 1, 2) : 0.0, 0.0, 0.5, 0.5;
  (0, 2, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 2) : 0.0, 1.0, 0.0, 0.0;
  (2, 2, 2) : 0.0, 0.0, 1.0, 0.0;
  (3, 2, 2) : 0.0, 0.0, 0.5, 0.5;
  (0, 3, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 2) : 0.9, 0.1, 0.0, 0.0;
  (3, 3, 2) : 0.6, 0.3, 0.1, 0.0;
  (0, 0, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 1, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 2, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 3, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 0, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 1, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 2, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 3, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 3, 4) : 0.05, 0.40, 0.50, 0.05;
}
probability ( R_LNLBE_APB_DENERV | R_LNLBE_MED_SEV, R_LNLBE_MED_TIME, R_LNLBE_MED_PATHO ) {
  (0, 0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (4, 0, 0) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 1, 0) : 0.3, 0.5, 0.2, 0.0;
  (3, 1, 0) : 0.1, 0.5, 0.4, 0.0;
  (4, 1, 0) : 0.0, 0.4, 0.4, 0.2;
  (0, 2, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 0) : 0.8, 0.2, 0.0, 0.0;
  (2, 2, 0) : 0.3, 0.5, 0.2, 0.0;
  (3, 2, 0) : 0.1, 0.5, 0.4, 0.0;
  (4, 2, 0) : 0.0, 0.4, 0.4, 0.2;
  (0, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 0) : 1.0, 0.0, 0.0, 0.0;
  (3, 3, 0) : 0.5, 0.4, 0.1, 0.0;
  (4, 3, 0) : 0.10, 0.60, 0.25, 0.05;
  (0, 0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (4, 0, 1) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 1) : 0.8, 0.2, 0.0, 0.0;
  (2, 1, 1) : 0.6, 0.4, 0.0, 0.0;
  (3, 1, 1) : 0.4, 0.5, 0.1, 0.0;
  (4, 1, 1) : 0.3, 0.4, 0.3, 0.0;
  (0, 2, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 1) : 0.9, 0.1, 0.0, 0.0;
  (2, 2, 1) : 0.6, 0.4, 0.0, 0.0;
  (3, 2, 1) : 0.4, 0.5, 0.1, 0.0;
  (4, 2, 1) : 0.3, 0.4, 0.3, 0.0;
  (0, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 1) : 1.0, 0.0, 0.0, 0.0;
  (3, 3, 1) : 0.5, 0.5, 0.0, 0.0;
  (4, 3, 1) : 0.5, 0.5, 0.0, 0.0;
  (0, 0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (2, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (3, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (4, 0, 2) : 0.8, 0.2, 0.0, 0.0;
  (0, 1, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 2) : 0.0, 1.0, 0.0, 0.0;
  (2, 1, 2) : 0.0, 0.0, 1.0, 0.0;
  (3, 1, 2) : 0.0, 0.0, 0.5, 0.5;
  (4, 1, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 2, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 2) : 0.0, 1.0, 0.0, 0.0;
  (2, 2, 2) : 0.0, 0.0, 1.0, 0.0;
  (3, 2, 2) : 0.0, 0.0, 0.5, 0.5;
  (4, 2, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 3, 2) : 0.9, 0.1, 0.0, 0.0;
  (3, 3, 2) : 0.6, 0.3, 0.1, 0.0;
  (4, 3, 2) : 0.45, 0.45, 0.10, 0.00;
  (0, 0, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 0, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 1, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 1, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 2, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 2, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 3, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (2, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (3, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (4, 3, 3) : 0.0, 0.0, 0.5, 0.5;
  (0, 0, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 0, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 1, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 1, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 2, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 2, 4) : 0.05, 0.40, 0.50, 0.05;
  (0, 3, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (2, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (3, 3, 4) : 0.05, 0.40, 0.50, 0.05;
  (4, 3, 4) : 0.05, 0.40, 0.50, 0.05;
}
probability ( R_LNLT1_LP_APB_DENERV | R_LNLT1_APB_DENERV, R_LNLLP_APB_DENERV ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.0, 1.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0;
  (1, 2) : 0.0, 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 0.0, 1.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLW_APB_NEUR_ACT | R_LNLW_MED_SEV, R_LNLW_MED_TIME ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (4, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (4, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.3, 0.7, 0.0, 0.0, 0.0, 0.0;
  (4, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.1, 0.9, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 0.3, 0.7, 0.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_DIFFN_APB_NEUR_ACT | DIFFN_M_SEV_DIST, DIFFN_TIME ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.3, 0.7, 0.0, 0.0, 0.0, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.1, 0.9, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 0.3, 0.7, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLBE_APB_NEUR_ACT | R_LNLBE_MED_SEV, R_LNLBE_MED_TIME ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (4, 0) : 0.9, 0.1, 0.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (4, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.3, 0.7, 0.0, 0.0, 0.0, 0.0;
  (4, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.1, 0.9, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 0.7, 0.3, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 0.3, 0.7, 0.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLT1_LP_APB_NEUR_ACT | R_LNLT1_APB_NEUR_ACT, R_LNLLP_APB_NEUR_ACT ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MYDY_APB_MUDENS ) {
   1.0, 0.0, 0.0;
}
probability ( R_MYOP_APB_MUDENS ) {
   1.0, 0.0, 0.0;
}
probability ( R_LNLW_APB_MUDENS | R_LNLW_MED_SEV, R_LNLW_MED_TIME, R_LNLW_MED_PATHO ) {
  (0, 0, 0) : 1.0, 0.0, 0.0;
  (1, 0, 0) : 1.0, 0.0, 0.0;
  (2, 0, 0) : 1.0, 0.0, 0.0;
  (3, 0, 0) : 1.0, 0.0, 0.0;
  (4, 0, 0) : 1.0, 0.0, 0.0;
  (0, 1, 0) : 1.0, 0.0, 0.0;
  (1, 1, 0) : 0.9, 0.1, 0.0;
  (2, 1, 0) : 0.8, 0.2, 0.0;
  (3, 1, 0) : 0.6, 0.4, 0.0;
  (4, 1, 0) : 0.6, 0.4, 0.0;
  (0, 2, 0) : 1.0, 0.0, 0.0;
  (1, 2, 0) : 0.8, 0.2, 0.0;
  (2, 2, 0) : 0.7, 0.3, 0.0;
  (3, 2, 0) : 0.6, 0.4, 0.0;
  (4, 2, 0) : 0.6, 0.4, 0.0;
  (0, 3, 0) : 1.0, 0.0, 0.0;
  (1, 3, 0) : 0.8, 0.2, 0.0;
  (2, 3, 0) : 0.7, 0.3, 0.0;
  (3, 3, 0) : 0.6, 0.4, 0.0;
  (4, 3, 0) : 0.6, 0.4, 0.0;
  (0, 0, 1) : 1.0, 0.0, 0.0;
  (1, 0, 1) : 1.0, 0.0, 0.0;
  (2, 0, 1) : 1.0, 0.0, 0.0;
  (3, 0, 1) : 1.0, 0.0, 0.0;
  (4, 0, 1) : 1.0, 0.0, 0.0;
  (0, 1, 1) : 1.0, 0.0, 0.0;
  (1, 1, 1) : 0.9, 0.1, 0.0;
  (2, 1, 1) : 0.8, 0.2, 0.0;
  (3, 1, 1) : 0.6, 0.4, 0.0;
  (4, 1, 1) : 0.6, 0.4, 0.0;
  (0, 2, 1) : 1.0, 0.0, 0.0;
  (1, 2, 1) : 0.8, 0.2, 0.0;
  (2, 2, 1) : 0.7, 0.3, 0.0;
  (3, 2, 1) : 0.6, 0.4, 0.0;
  (4, 2, 1) : 0.6, 0.4, 0.0;
  (0, 3, 1) : 1.0, 0.0, 0.0;
  (1, 3, 1) : 0.8, 0.2, 0.0;
  (2, 3, 1) : 0.7, 0.3, 0.0;
  (3, 3, 1) : 0.6, 0.4, 0.0;
  (4, 3, 1) : 0.6, 0.4, 0.0;
  (0, 0, 2) : 1.0, 0.0, 0.0;
  (1, 0, 2) : 1.0, 0.0, 0.0;
  (2, 0, 2) : 1.0, 0.0, 0.0;
  (3, 0, 2) : 1.0, 0.0, 0.0;
  (4, 0, 2) : 1.0, 0.0, 0.0;
  (0, 1, 2) : 1.0, 0.0, 0.0;
  (1, 1, 2) : 0.6, 0.4, 0.0;
  (2, 1, 2) : 0.5, 0.5, 0.0;
  (3, 1, 2) : 0.5, 0.4, 0.1;
  (4, 1, 2) : 0.3, 0.6, 0.1;
  (0, 2, 2) : 1.0, 0.0, 0.0;
  (1, 2, 2) : 0.7, 0.3, 0.0;
  (2, 2, 2) : 0.1, 0.6, 0.3;
  (3, 2, 2) : 0.0, 0.5, 0.5;
  (4, 2, 2) : 0.0, 0.5, 0.5;
  (0, 3, 2) : 1.0, 0.0, 0.0;
  (1, 3, 2) : 0.5, 0.5, 0.0;
  (2, 3, 2) : 0.15, 0.70, 0.15;
  (3, 3, 2) : 0.0, 0.5, 0.5;
  (4, 3, 2) : 0.0, 0.5, 0.5;
  (0, 0, 3) : 1.0, 0.0, 0.0;
  (1, 0, 3) : 1.0, 0.0, 0.0;
  (2, 0, 3) : 1.0, 0.0, 0.0;
  (3, 0, 3) : 1.0, 0.0, 0.0;
  (4, 0, 3) : 1.0, 0.0, 0.0;
  (0, 1, 3) : 1.0, 0.0, 0.0;
  (1, 1, 3) : 0.05, 0.50, 0.45;
  (2, 1, 3) : 0.05, 0.50, 0.45;
  (3, 1, 3) : 0.05, 0.50, 0.45;
  (4, 1, 3) : 0.05, 0.50, 0.45;
  (0, 2, 3) : 1.0, 0.0, 0.0;
  (1, 2, 3) : 0.05, 0.50, 0.45;
  (2, 2, 3) : 0.05, 0.50, 0.45;
  (3, 2, 3) : 0.05, 0.50, 0.45;
  (4, 2, 3) : 0.05, 0.50, 0.45;
  (0, 3, 3) : 1.0, 0.0, 0.0;
  (1, 3, 3) : 0.05, 0.50, 0.45;
  (2, 3, 3) : 0.05, 0.50, 0.45;
  (3, 3, 3) : 0.05, 0.50, 0.45;
  (4, 3, 3) : 0.05, 0.50, 0.45;
  (0, 0, 4) : 1.0, 0.0, 0.0;
  (1, 0, 4) : 1.0, 0.0, 0.0;
  (2, 0, 4) : 1.0, 0.0, 0.0;
  (3, 0, 4) : 1.0, 0.0, 0.0;
  (4, 0, 4) : 1.0, 0.0, 0.0;
  (0, 1, 4) : 1.0, 0.0, 0.0;
  (1, 1, 4) : 0.2, 0.5, 0.3;
  (2, 1, 4) : 0.2, 0.5, 0.3;
  (3, 1, 4) : 0.2, 0.5, 0.3;
  (4, 1, 4) : 0.2, 0.5, 0.3;
  (0, 2, 4) : 1.0, 0.0, 0.0;
  (1, 2, 4) : 0.2, 0.5, 0.3;
  (2, 2, 4) : 0.2, 0.5, 0.3;
  (3, 2, 4) : 0.2, 0.5, 0.3;
  (4, 2, 4) : 0.2, 0.5, 0.3;
  (0, 3, 4) : 1.0, 0.0, 0.0;
  (1, 3, 4) : 0.2, 0.5, 0.3;
  (2, 3, 4) : 0.2, 0.5, 0.3;
  (3, 3, 4) : 0.2, 0.5, 0.3;
  (4, 3, 4) : 0.2, 0.5, 0.3;
}
probability ( R_DIFFN_APB_MUDENS | DIFFN_M_SEV_DIST, DIFFN_TIME, DIFFN_PATHO ) {
  (0, 0, 0) : 1.0, 0.0, 0.0;
  (1, 0, 0) : 1.0, 0.0, 0.0;
  (2, 0, 0) : 1.0, 0.0, 0.0;
  (3, 0, 0) : 1.0, 0.0, 0.0;
  (0, 1, 0) : 1.0, 0.0, 0.0;
  (1, 1, 0) : 0.9, 0.1, 0.0;
  (2, 1, 0) : 0.8, 0.2, 0.0;
  (3, 1, 0) : 0.6, 0.4, 0.0;
  (0, 2, 0) : 1.0, 0.0, 0.0;
  (1, 2, 0) : 0.8, 0.2, 0.0;
  (2, 2, 0) : 0.7, 0.3, 0.0;
  (3, 2, 0) : 0.6, 0.4, 0.0;
  (0, 3, 0) : 1.0, 0.0, 0.0;
  (1, 3, 0) : 0.8, 0.2, 0.0;
  (2, 3, 0) : 0.7, 0.3, 0.0;
  (3, 3, 0) : 0.6, 0.4, 0.0;
  (0, 0, 1) : 1.0, 0.0, 0.0;
  (1, 0, 1) : 1.0, 0.0, 0.0;
  (2, 0, 1) : 1.0, 0.0, 0.0;
  (3, 0, 1) : 1.0, 0.0, 0.0;
  (0, 1, 1) : 1.0, 0.0, 0.0;
  (1, 1, 1) : 0.9, 0.1, 0.0;
  (2, 1, 1) : 0.8, 0.2, 0.0;
  (3, 1, 1) : 0.6, 0.4, 0.0;
  (0, 2, 1) : 1.0, 0.0, 0.0;
  (1, 2, 1) : 0.8, 0.2, 0.0;
  (2, 2, 1) : 0.7, 0.3, 0.0;
  (3, 2, 1) : 0.6, 0.4, 0.0;
  (0, 3, 1) : 1.0, 0.0, 0.0;
  (1, 3, 1) : 0.8, 0.2, 0.0;
  (2, 3, 1) : 0.7, 0.3, 0.0;
  (3, 3, 1) : 0.6, 0.4, 0.0;
  (0, 0, 2) : 1.0, 0.0, 0.0;
  (1, 0, 2) : 1.0, 0.0, 0.0;
  (2, 0, 2) : 1.0, 0.0, 0.0;
  (3, 0, 2) : 1.0, 0.0, 0.0;
  (0, 1, 2) : 1.0, 0.0, 0.0;
  (1, 1, 2) : 0.6, 0.4, 0.0;
  (2, 1, 2) : 0.5, 0.5, 0.0;
  (3, 1, 2) : 0.5, 0.4, 0.1;
  (0, 2, 2) : 1.0, 0.0, 0.0;
  (1, 2, 2) : 0.7, 0.3, 0.0;
  (2, 2, 2) : 0.1, 0.6, 0.3;
  (3, 2, 2) : 0.0, 0.5, 0.5;
  (0, 3, 2) : 1.0, 0.0, 0.0;
  (1, 3, 2) : 0.5, 0.5, 0.0;
  (2, 3, 2) : 0.15, 0.70, 0.15;
  (3, 3, 2) : 0.0, 0.5, 0.5;
  (0, 0, 3) : 1.0, 0.0, 0.0;
  (1, 0, 3) : 1.0, 0.0, 0.0;
  (2, 0, 3) : 1.0, 0.0, 0.0;
  (3, 0, 3) : 1.0, 0.0, 0.0;
  (0, 1, 3) : 1.0, 0.0, 0.0;
  (1, 1, 3) : 0.05, 0.50, 0.45;
  (2, 1, 3) : 0.05, 0.50, 0.45;
  (3, 1, 3) : 0.05, 0.50, 0.45;
  (0, 2, 3) : 1.0, 0.0, 0.0;
  (1, 2, 3) : 0.05, 0.50, 0.45;
  (2, 2, 3) : 0.05, 0.50, 0.45;
  (3, 2, 3) : 0.05, 0.50, 0.45;
  (0, 3, 3) : 1.0, 0.0, 0.0;
  (1, 3, 3) : 0.05, 0.50, 0.45;
  (2, 3, 3) : 0.05, 0.50, 0.45;
  (3, 3, 3) : 0.05, 0.50, 0.45;
  (0, 0, 4) : 1.0, 0.0, 0.0;
  (1, 0, 4) : 1.0, 0.0, 0.0;
  (2, 0, 4) : 1.0, 0.0, 0.0;
  (3, 0, 4) : 1.0, 0.0, 0.0;
  (0, 1, 4) : 1.0, 0.0, 0.0;
  (1, 1, 4) : 0.2, 0.5, 0.3;
  (2, 1, 4) : 0.2, 0.5, 0.3;
  (3, 1, 4) : 0.2, 0.5, 0.3;
  (0, 2, 4) : 1.0, 0.0, 0.0;
  (1, 2, 4) : 0.2, 0.5, 0.3;
  (2, 2, 4) : 0.2, 0.5, 0.3;
  (3, 2, 4) : 0.2, 0.5, 0.3;
  (0, 3, 4) : 1.0, 0.0, 0.0;
  (1, 3, 4) : 0.2, 0.5, 0.3;
  (2, 3, 4) : 0.2, 0.5, 0.3;
  (3, 3, 4) : 0.2, 0.5, 0.3;
}
probability ( R_LNLBE_APB_MUDENS | R_LNLBE_MED_SEV, R_LNLBE_MED_TIME, R_LNLBE_MED_PATHO ) {
  (0, 0, 0) : 1.0, 0.0, 0.0;
  (1, 0, 0) : 1.0, 0.0, 0.0;
  (2, 0, 0) : 1.0, 0.0, 0.0;
  (3, 0, 0) : 1.0, 0.0, 0.0;
  (4, 0, 0) : 1.0, 0.0, 0.0;
  (0, 1, 0) : 1.0, 0.0, 0.0;
  (1, 1, 0) : 0.9, 0.1, 0.0;
  (2, 1, 0) : 0.8, 0.2, 0.0;
  (3, 1, 0) : 0.6, 0.4, 0.0;
  (4, 1, 0) : 0.6, 0.4, 0.0;
  (0, 2, 0) : 1.0, 0.0, 0.0;
  (1, 2, 0) : 0.8, 0.2, 0.0;
  (2, 2, 0) : 0.7, 0.3, 0.0;
  (3, 2, 0) : 0.6, 0.4, 0.0;
  (4, 2, 0) : 0.6, 0.4, 0.0;
  (0, 3, 0) : 1.0, 0.0, 0.0;
  (1, 3, 0) : 0.8, 0.2, 0.0;
  (2, 3, 0) : 0.7, 0.3, 0.0;
  (3, 3, 0) : 0.6, 0.4, 0.0;
  (4, 3, 0) : 0.6, 0.4, 0.0;
  (0, 0, 1) : 1.0, 0.0, 0.0;
  (1, 0, 1) : 1.0, 0.0, 0.0;
  (2, 0, 1) : 1.0, 0.0, 0.0;
  (3, 0, 1) : 1.0, 0.0, 0.0;
  (4, 0, 1) : 1.0, 0.0, 0.0;
  (0, 1, 1) : 1.0, 0.0, 0.0;
  (1, 1, 1) : 0.9, 0.1, 0.0;
  (2, 1, 1) : 0.8, 0.2, 0.0;
  (3, 1, 1) : 0.6, 0.4, 0.0;
  (4, 1, 1) : 0.6, 0.4, 0.0;
  (0, 2, 1) : 1.0, 0.0, 0.0;
  (1, 2, 1) : 0.8, 0.2, 0.0;
  (2, 2, 1) : 0.7, 0.3, 0.0;
  (3, 2, 1) : 0.6, 0.4, 0.0;
  (4, 2, 1) : 0.6, 0.4, 0.0;
  (0, 3, 1) : 1.0, 0.0, 0.0;
  (1, 3, 1) : 0.8, 0.2, 0.0;
  (2, 3, 1) : 0.7, 0.3, 0.0;
  (3, 3, 1) : 0.6, 0.4, 0.0;
  (4, 3, 1) : 0.6, 0.4, 0.0;
  (0, 0, 2) : 1.0, 0.0, 0.0;
  (1, 0, 2) : 1.0, 0.0, 0.0;
  (2, 0, 2) : 1.0, 0.0, 0.0;
  (3, 0, 2) : 1.0, 0.0, 0.0;
  (4, 0, 2) : 1.0, 0.0, 0.0;
  (0, 1, 2) : 1.0, 0.0, 0.0;
  (1, 1, 2) : 0.6, 0.4, 0.0;
  (2, 1, 2) : 0.5, 0.5, 0.0;
  (3, 1, 2) : 0.5, 0.4, 0.1;
  (4, 1, 2) : 0.3, 0.6, 0.1;
  (0, 2, 2) : 1.0, 0.0, 0.0;
  (1, 2, 2) : 0.7, 0.3, 0.0;
  (2, 2, 2) : 0.1, 0.6, 0.3;
  (3, 2, 2) : 0.0, 0.5, 0.5;
  (4, 2, 2) : 0.0, 0.5, 0.5;
  (0, 3, 2) : 1.0, 0.0, 0.0;
  (1, 3, 2) : 0.5, 0.5, 0.0;
  (2, 3, 2) : 0.15, 0.70, 0.15;
  (3, 3, 2) : 0.0, 0.5, 0.5;
  (4, 3, 2) : 0.0, 0.5, 0.5;
  (0, 0, 3) : 1.0, 0.0, 0.0;
  (1, 0, 3) : 1.0, 0.0, 0.0;
  (2, 0, 3) : 1.0, 0.0, 0.0;
  (3, 0, 3) : 1.0, 0.0, 0.0;
  (4, 0, 3) : 1.0, 0.0, 0.0;
  (0, 1, 3) : 1.0, 0.0, 0.0;
  (1, 1, 3) : 0.05, 0.50, 0.45;
  (2, 1, 3) : 0.05, 0.50, 0.45;
  (3, 1, 3) : 0.05, 0.50, 0.45;
  (4, 1, 3) : 0.05, 0.50, 0.45;
  (0, 2, 3) : 1.0, 0.0, 0.0;
  (1, 2, 3) : 0.05, 0.50, 0.45;
  (2, 2, 3) : 0.05, 0.50, 0.45;
  (3, 2, 3) : 0.05, 0.50, 0.45;
  (4, 2, 3) : 0.05, 0.50, 0.45;
  (0, 3, 3) : 1.0, 0.0, 0.0;
  (1, 3, 3) : 0.05, 0.50, 0.45;
  (2, 3, 3) : 0.05, 0.50, 0.45;
  (3, 3, 3) : 0.05, 0.50, 0.45;
  (4, 3, 3) : 0.05, 0.50, 0.45;
  (0, 0, 4) : 1.0, 0.0, 0.0;
  (1, 0, 4) : 1.0, 0.0, 0.0;
  (2, 0, 4) : 1.0, 0.0, 0.0;
  (3, 0, 4) : 1.0, 0.0, 0.0;
  (4, 0, 4) : 1.0, 0.0, 0.0;
  (0, 1, 4) : 1.0, 0.0, 0.0;
  (1, 1, 4) : 0.2, 0.5, 0.3;
  (2, 1, 4) : 0.2, 0.5, 0.3;
  (3, 1, 4) : 0.2, 0.5, 0.3;
  (4, 1, 4) : 0.2, 0.5, 0.3;
  (0, 2, 4) : 1.0, 0.0, 0.0;
  (1, 2, 4) : 0.2, 0.5, 0.3;
  (2, 2, 4) : 0.2, 0.5, 0.3;
  (3, 2, 4) : 0.2, 0.5, 0.3;
  (4, 2, 4) : 0.2, 0.5, 0.3;
  (0, 3, 4) : 1.0, 0.0, 0.0;
  (1, 3, 4) : 0.2, 0.5, 0.3;
  (2, 3, 4) : 0.2, 0.5, 0.3;
  (3, 3, 4) : 0.2, 0.5, 0.3;
  (4, 3, 4) : 0.2, 0.5, 0.3;
}
probability ( R_LNLT1_LP_APB_MUDENS | R_LNLT1_APB_MUDENS, R_LNLLP_APB_MUDENS ) {
  (0, 0) : 1.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0;
  (2, 1) : 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0;
  (1, 2) : 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 1.0;
}
probability ( R_MYDY_APB_DE_REGEN ) {
   1.0, 0.0;
}
probability ( R_MYOP_APB_DE_REGEN ) {
   1.0, 0.0;
}
probability ( R_LNLW_APB_DE_REGEN | R_LNLW_MED_SEV, R_LNLW_MED_TIME, R_LNLW_MED_PATHO ) {
  (0, 0, 0) : 1.0, 0.0;
  (1, 0, 0) : 1.0, 0.0;
  (2, 0, 0) : 1.0, 0.0;
  (3, 0, 0) : 1.0, 0.0;
  (4, 0, 0) : 1.0, 0.0;
  (0, 1, 0) : 1.0, 0.0;
  (1, 1, 0) : 0.8, 0.2;
  (2, 1, 0) : 0.2, 0.8;
  (3, 1, 0) : 0.4, 0.6;
  (4, 1, 0) : 1.0, 0.0;
  (0, 2, 0) : 1.0, 0.0;
  (1, 2, 0) : 0.8, 0.2;
  (2, 2, 0) : 0.2, 0.8;
  (3, 2, 0) : 0.4, 0.6;
  (4, 2, 0) : 1.0, 0.0;
  (0, 3, 0) : 1.0, 0.0;
  (1, 3, 0) : 1.0, 0.0;
  (2, 3, 0) : 0.8, 0.2;
  (3, 3, 0) : 0.4, 0.6;
  (4, 3, 0) : 1.0, 0.0;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 1.0, 0.0;
  (2, 0, 1) : 1.0, 0.0;
  (3, 0, 1) : 1.0, 0.0;
  (4, 0, 1) : 1.0, 0.0;
  (0, 1, 1) : 1.0, 0.0;
  (1, 1, 1) : 0.8, 0.2;
  (2, 1, 1) : 0.2, 0.8;
  (3, 1, 1) : 0.4, 0.6;
  (4, 1, 1) : 1.0, 0.0;
  (0, 2, 1) : 1.0, 0.0;
  (1, 2, 1) : 0.8, 0.2;
  (2, 2, 1) : 0.2, 0.8;
  (3, 2, 1) : 0.4, 0.6;
  (4, 2, 1) : 1.0, 0.0;
  (0, 3, 1) : 1.0, 0.0;
  (1, 3, 1) : 1.0, 0.0;
  (2, 3, 1) : 0.8, 0.2;
  (3, 3, 1) : 0.4, 0.6;
  (4, 3, 1) : 1.0, 0.0;
  (0, 0, 2) : 1.0, 0.0;
  (1, 0, 2) : 1.0, 0.0;
  (2, 0, 2) : 1.0, 0.0;
  (3, 0, 2) : 1.0, 0.0;
  (4, 0, 2) : 1.0, 0.0;
  (0, 1, 2) : 1.0, 0.0;
  (1, 1, 2) : 0.5, 0.5;
  (2, 1, 2) : 0.2, 0.8;
  (3, 1, 2) : 0.1, 0.9;
  (4, 1, 2) : 1.0, 0.0;
  (0, 2, 2) : 1.0, 0.0;
  (1, 2, 2) : 0.5, 0.5;
  (2, 2, 2) : 0.2, 0.8;
  (3, 2, 2) : 0.1, 0.9;
  (4, 2, 2) : 1.0, 0.0;
  (0, 3, 2) : 1.0, 0.0;
  (1, 3, 2) : 1.0, 0.0;
  (2, 3, 2) : 0.8, 0.2;
  (3, 3, 2) : 0.4, 0.6;
  (4, 3, 2) : 1.0, 0.0;
  (0, 0, 3) : 0.0, 1.0;
  (1, 0, 3) : 0.0, 1.0;
  (2, 0, 3) : 0.0, 1.0;
  (3, 0, 3) : 0.0, 1.0;
  (4, 0, 3) : 0.0, 1.0;
  (0, 1, 3) : 0.0, 1.0;
  (1, 1, 3) : 0.0, 1.0;
  (2, 1, 3) : 0.0, 1.0;
  (3, 1, 3) : 0.0, 1.0;
  (4, 1, 3) : 0.0, 1.0;
  (0, 2, 3) : 0.0, 1.0;
  (1, 2, 3) : 0.0, 1.0;
  (2, 2, 3) : 0.0, 1.0;
  (3, 2, 3) : 0.0, 1.0;
  (4, 2, 3) : 0.0, 1.0;
  (0, 3, 3) : 0.0, 1.0;
  (1, 3, 3) : 0.0, 1.0;
  (2, 3, 3) : 0.0, 1.0;
  (3, 3, 3) : 0.0, 1.0;
  (4, 3, 3) : 0.0, 1.0;
  (0, 0, 4) : 0.0, 1.0;
  (1, 0, 4) : 0.0, 1.0;
  (2, 0, 4) : 0.0, 1.0;
  (3, 0, 4) : 0.0, 1.0;
  (4, 0, 4) : 0.0, 1.0;
  (0, 1, 4) : 0.0, 1.0;
  (1, 1, 4) : 0.0, 1.0;
  (2, 1, 4) : 0.0, 1.0;
  (3, 1, 4) : 0.0, 1.0;
  (4, 1, 4) : 0.0, 1.0;
  (0, 2, 4) : 0.0, 1.0;
  (1, 2, 4) : 0.0, 1.0;
  (2, 2, 4) : 0.0, 1.0;
  (3, 2, 4) : 0.0, 1.0;
  (4, 2, 4) : 0.0, 1.0;
  (0, 3, 4) : 0.0, 1.0;
  (1, 3, 4) : 0.0, 1.0;
  (2, 3, 4) : 0.0, 1.0;
  (3, 3, 4) : 0.0, 1.0;
  (4, 3, 4) : 0.0, 1.0;
}
probability ( R_DIFFN_APB_DE_REGEN | DIFFN_M_SEV_DIST, DIFFN_TIME, DIFFN_PATHO ) {
  (0, 0, 0) : 1.0, 0.0;
  (1, 0, 0) : 1.0, 0.0;
  (2, 0, 0) : 1.0, 0.0;
  (3, 0, 0) : 1.0, 0.0;
  (0, 1, 0) : 1.0, 0.0;
  (1, 1, 0) : 0.8, 0.2;
  (2, 1, 0) : 0.2, 0.8;
  (3, 1, 0) : 0.4, 0.6;
  (0, 2, 0) : 1.0, 0.0;
  (1, 2, 0) : 0.8, 0.2;
  (2, 2, 0) : 0.2, 0.8;
  (3, 2, 0) : 0.4, 0.6;
  (0, 3, 0) : 1.0, 0.0;
  (1, 3, 0) : 1.0, 0.0;
  (2, 3, 0) : 0.8, 0.2;
  (3, 3, 0) : 0.4, 0.6;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 1.0, 0.0;
  (2, 0, 1) : 1.0, 0.0;
  (3, 0, 1) : 1.0, 0.0;
  (0, 1, 1) : 1.0, 0.0;
  (1, 1, 1) : 0.8, 0.2;
  (2, 1, 1) : 0.2, 0.8;
  (3, 1, 1) : 0.4, 0.6;
  (0, 2, 1) : 1.0, 0.0;
  (1, 2, 1) : 0.8, 0.2;
  (2, 2, 1) : 0.2, 0.8;
  (3, 2, 1) : 0.4, 0.6;
  (0, 3, 1) : 1.0, 0.0;
  (1, 3, 1) : 1.0, 0.0;
  (2, 3, 1) : 0.8, 0.2;
  (3, 3, 1) : 0.4, 0.6;
  (0, 0, 2) : 1.0, 0.0;
  (1, 0, 2) : 1.0, 0.0;
  (2, 0, 2) : 1.0, 0.0;
  (3, 0, 2) : 1.0, 0.0;
  (0, 1, 2) : 1.0, 0.0;
  (1, 1, 2) : 0.5, 0.5;
  (2, 1, 2) : 0.2, 0.8;
  (3, 1, 2) : 0.1, 0.9;
  (0, 2, 2) : 1.0, 0.0;
  (1, 2, 2) : 0.5, 0.5;
  (2, 2, 2) : 0.2, 0.8;
  (3, 2, 2) : 0.1, 0.9;
  (0, 3, 2) : 1.0, 0.0;
  (1, 3, 2) : 1.0, 0.0;
  (2, 3, 2) : 0.8, 0.2;
  (3, 3, 2) : 0.4, 0.6;
  (0, 0, 3) : 0.0, 1.0;
  (1, 0, 3) : 0.0, 1.0;
  (2, 0, 3) : 0.0, 1.0;
  (3, 0, 3) : 0.0, 1.0;
  (0, 1, 3) : 0.0, 1.0;
  (1, 1, 3) : 0.0, 1.0;
  (2, 1, 3) : 0.0, 1.0;
  (3, 1, 3) : 0.0, 1.0;
  (0, 2, 3) : 0.0, 1.0;
  (1, 2, 3) : 0.0, 1.0;
  (2, 2, 3) : 0.0, 1.0;
  (3, 2, 3) : 0.0, 1.0;
  (0, 3, 3) : 0.0, 1.0;
  (1, 3, 3) : 0.0, 1.0;
  (2, 3, 3) : 0.0, 1.0;
  (3, 3, 3) : 0.0, 1.0;
  (0, 0, 4) : 0.0, 1.0;
  (1, 0, 4) : 0.0, 1.0;
  (2, 0, 4) : 0.0, 1.0;
  (3, 0, 4) : 0.0, 1.0;
  (0, 1, 4) : 0.0, 1.0;
  (1, 1, 4) : 0.0, 1.0;
  (2, 1, 4) : 0.0, 1.0;
  (3, 1, 4) : 0.0, 1.0;
  (0, 2, 4) : 0.0, 1.0;
  (1, 2, 4) : 0.0, 1.0;
  (2, 2, 4) : 0.0, 1.0;
  (3, 2, 4) : 0.0, 1.0;
  (0, 3, 4) : 0.0, 1.0;
  (1, 3, 4) : 0.0, 1.0;
  (2, 3, 4) : 0.0, 1.0;
  (3, 3, 4) : 0.0, 1.0;
}
probability ( R_LNLBE_APB_DE_REGEN | R_LNLBE_MED_SEV, R_LNLBE_MED_TIME, R_LNLBE_MED_PATHO ) {
  (0, 0, 0) : 1.0, 0.0;
  (1, 0, 0) : 1.0, 0.0;
  (2, 0, 0) : 1.0, 0.0;
  (3, 0, 0) : 1.0, 0.0;
  (4, 0, 0) : 1.0, 0.0;
  (0, 1, 0) : 1.0, 0.0;
  (1, 1, 0) : 0.8, 0.2;
  (2, 1, 0) : 0.2, 0.8;
  (3, 1, 0) : 0.4, 0.6;
  (4, 1, 0) : 1.0, 0.0;
  (0, 2, 0) : 1.0, 0.0;
  (1, 2, 0) : 0.8, 0.2;
  (2, 2, 0) : 0.2, 0.8;
  (3, 2, 0) : 0.4, 0.6;
  (4, 2, 0) : 1.0, 0.0;
  (0, 3, 0) : 1.0, 0.0;
  (1, 3, 0) : 1.0, 0.0;
  (2, 3, 0) : 0.8, 0.2;
  (3, 3, 0) : 0.4, 0.6;
  (4, 3, 0) : 1.0, 0.0;
  (0, 0, 1) : 1.0, 0.0;
  (1, 0, 1) : 1.0, 0.0;
  (2, 0, 1) : 1.0, 0.0;
  (3, 0, 1) : 1.0, 0.0;
  (4, 0, 1) : 1.0, 0.0;
  (0, 1, 1) : 1.0, 0.0;
  (1, 1, 1) : 0.8, 0.2;
  (2, 1, 1) : 0.2, 0.8;
  (3, 1, 1) : 0.4, 0.6;
  (4, 1, 1) : 1.0, 0.0;
  (0, 2, 1) : 1.0, 0.0;
  (1, 2, 1) : 0.8, 0.2;
  (2, 2, 1) : 0.2, 0.8;
  (3, 2, 1) : 0.4, 0.6;
  (4, 2, 1) : 1.0, 0.0;
  (0, 3, 1) : 1.0, 0.0;
  (1, 3, 1) : 1.0, 0.0;
  (2, 3, 1) : 0.8, 0.2;
  (3, 3, 1) : 0.4, 0.6;
  (4, 3, 1) : 1.0, 0.0;
  (0, 0, 2) : 1.0, 0.0;
  (1, 0, 2) : 1.0, 0.0;
  (2, 0, 2) : 1.0, 0.0;
  (3, 0, 2) : 1.0, 0.0;
  (4, 0, 2) : 1.0, 0.0;
  (0, 1, 2) : 1.0, 0.0;
  (1, 1, 2) : 0.5, 0.5;
  (2, 1, 2) : 0.2, 0.8;
  (3, 1, 2) : 0.1, 0.9;
  (4, 1, 2) : 1.0, 0.0;
  (0, 2, 2) : 1.0, 0.0;
  (1, 2, 2) : 0.5, 0.5;
  (2, 2, 2) : 0.2, 0.8;
  (3, 2, 2) : 0.1, 0.9;
  (4, 2, 2) : 1.0, 0.0;
  (0, 3, 2) : 1.0, 0.0;
  (1, 3, 2) : 1.0, 0.0;
  (2, 3, 2) : 0.8, 0.2;
  (3, 3, 2) : 0.4, 0.6;
  (4, 3, 2) : 1.0, 0.0;
  (0, 0, 3) : 0.0, 1.0;
  (1, 0, 3) : 0.0, 1.0;
  (2, 0, 3) : 0.0, 1.0;
  (3, 0, 3) : 0.0, 1.0;
  (4, 0, 3) : 0.0, 1.0;
  (0, 1, 3) : 0.0, 1.0;
  (1, 1, 3) : 0.0, 1.0;
  (2, 1, 3) : 0.0, 1.0;
  (3, 1, 3) : 0.0, 1.0;
  (4, 1, 3) : 0.0, 1.0;
  (0, 2, 3) : 0.0, 1.0;
  (1, 2, 3) : 0.0, 1.0;
  (2, 2, 3) : 0.0, 1.0;
  (3, 2, 3) : 0.0, 1.0;
  (4, 2, 3) : 0.0, 1.0;
  (0, 3, 3) : 0.0, 1.0;
  (1, 3, 3) : 0.0, 1.0;
  (2, 3, 3) : 0.0, 1.0;
  (3, 3, 3) : 0.0, 1.0;
  (4, 3, 3) : 0.0, 1.0;
  (0, 0, 4) : 0.0, 1.0;
  (1, 0, 4) : 0.0, 1.0;
  (2, 0, 4) : 0.0, 1.0;
  (3, 0, 4) : 0.0, 1.0;
  (4, 0, 4) : 0.0, 1.0;
  (0, 1, 4) : 0.0, 1.0;
  (1, 1, 4) : 0.0, 1.0;
  (2, 1, 4) : 0.0, 1.0;
  (3, 1, 4) : 0.0, 1.0;
  (4, 1, 4) : 0.0, 1.0;
  (0, 2, 4) : 0.0, 1.0;
  (1, 2, 4) : 0.0, 1.0;
  (2, 2, 4) : 0.0, 1.0;
  (3, 2, 4) : 0.0, 1.0;
  (4, 2, 4) : 0.0, 1.0;
  (0, 3, 4) : 0.0, 1.0;
  (1, 3, 4) : 0.0, 1.0;
  (2, 3, 4) : 0.0, 1.0;
  (3, 3, 4) : 0.0, 1.0;
  (4, 3, 4) : 0.0, 1.0;
}
probability ( R_LNLT1_LP_APB_DE_REGEN | R_LNLT1_APB_DE_REGEN, R_LNLLP_APB_DE_REGEN ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.0, 1.0;
}
probability ( R_MYOP_APB_MUSIZE ) {
   0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
}
probability ( R_MYDY_APB_MUSIZE ) {
   0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLW_MEDD2_DISP_WD | R_LNLW_MED_SEV, R_LNLW_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (4, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.5, 0.5, 0.0;
  (3, 1) : 0.0, 0.1, 0.5, 0.4;
  (4, 1) : 0.0, 0.0, 0.5, 0.5;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0;
  (3, 2) : 0.3, 0.5, 0.2, 0.0;
  (4, 2) : 0.0, 0.5, 0.5, 0.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
  (4, 3) : 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_MEDD2_SALOSS | DIFFN_S_SEV_DIST, DIFFN_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 0.7, 0.3;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 1) : 0.4, 0.3, 0.3, 0.0, 0.0;
  (3, 1) : 0.0, 0.3, 0.5, 0.2, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 0.7, 0.3;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (0, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (1, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (2, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
}
probability ( R_LNLW_MEDD2_SALOSS_WD | R_LNLW_MED_SEV, R_LNLW_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.4, 0.6;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 1) : 0.2, 0.5, 0.3, 0.0, 0.0;
  (3, 1) : 0.0, 0.2, 0.5, 0.3, 0.0;
  (4, 1) : 0.0, 0.1, 0.4, 0.4, 0.1;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (0, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (1, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (2, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
}
probability ( R_NMT_APB_DENERV | R_APB_NMT ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 0.40, 0.45, 0.15, 0.00;
  (2) : 0.15, 0.35, 0.35, 0.15;
  (3) : 0.85, 0.15, 0.00, 0.00;
  (4) : 0.30, 0.45, 0.20, 0.05;
  (5) : 0.15, 0.35, 0.35, 0.15;
  (6) : 0.25, 0.25, 0.25, 0.25;
}
probability ( R_MYOP_MYDY_APB_DENERV | R_MYOP_APB_DENERV, R_MYDY_APB_DENERV ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.0, 1.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0;
  (1, 2) : 0.0, 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 0.0, 1.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_LNLW_APB_DENERV | R_DIFFN_APB_DENERV, R_LNLW_APB_DENERV ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.0, 1.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0;
  (1, 2) : 0.0, 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 0.0, 1.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLT1_LP_BE_APB_DENERV | R_LNLT1_LP_APB_DENERV, R_LNLBE_APB_DENERV ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.0, 1.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0;
  (1, 2) : 0.0, 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 0.0, 1.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MYAS_APB_MUDENS ) {
   1.0, 0.0, 0.0;
}
probability ( R_MYOP_MYDY_APB_MUDENS | R_MYOP_APB_MUDENS, R_MYDY_APB_MUDENS ) {
  (0, 0) : 1.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0;
  (2, 1) : 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0;
  (1, 2) : 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_LNLW_APB_MUDENS | R_DIFFN_APB_MUDENS, R_LNLW_APB_MUDENS ) {
  (0, 0) : 1.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0;
  (2, 1) : 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0;
  (1, 2) : 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 1.0;
}
probability ( R_LNLT1_LP_BE_APB_MUDENS | R_LNLT1_LP_APB_MUDENS, R_LNLBE_APB_MUDENS ) {
  (0, 0) : 1.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0;
  (2, 1) : 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0;
  (1, 2) : 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_LNLW_APB_DE_REGEN | R_DIFFN_APB_DE_REGEN, R_LNLW_APB_DE_REGEN ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.0, 1.0;
}
probability ( R_LNLT1_LP_BE_APB_DE_REGEN | R_LNLT1_LP_APB_DE_REGEN, R_LNLBE_APB_DE_REGEN ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.0, 1.0;
}
probability ( R_DIFFN_APB_MUSIZE | DIFFN_M_SEV_DIST, DIFFN_TIME, DIFFN_PATHO ) {
  (0, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 0) : 0.0, 0.0, 0.7, 0.3, 0.0, 0.0;
  (2, 2, 0) : 0.0, 0.0, 0.0, 0.8, 0.2, 0.0;
  (3, 2, 0) : 0.0, 0.0, 0.0, 0.0, 0.5, 0.5;
  (0, 3, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 0) : 0.0, 0.0, 0.7, 0.3, 0.0, 0.0;
  (2, 3, 0) : 0.0, 0.0, 0.0, 0.8, 0.2, 0.0;
  (3, 3, 0) : 0.0, 0.0, 0.0, 0.0, 0.5, 0.5;
  (0, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 1) : 0.0, 0.0, 0.9, 0.1, 0.0, 0.0;
  (2, 2, 1) : 0.0, 0.0, 0.7, 0.2, 0.1, 0.0;
  (3, 2, 1) : 0.00, 0.00, 0.00, 0.25, 0.50, 0.25;
  (0, 3, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 1) : 0.0, 0.0, 0.9, 0.1, 0.0, 0.0;
  (2, 3, 1) : 0.0, 0.0, 0.7, 0.2, 0.1, 0.0;
  (3, 3, 1) : 0.00, 0.00, 0.00, 0.25, 0.50, 0.25;
  (0, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 2) : 0.0, 0.0, 0.7, 0.3, 0.0, 0.0;
  (2, 2, 2) : 0.0, 0.0, 0.0, 0.7, 0.3, 0.0;
  (3, 2, 2) : 0.0, 0.0, 0.0, 0.0, 0.5, 0.5;
  (0, 3, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 2) : 0.0, 0.0, 0.7, 0.3, 0.0, 0.0;
  (2, 3, 2) : 0.0, 0.0, 0.0, 0.7, 0.3, 0.0;
  (3, 3, 2) : 0.0, 0.0, 0.0, 0.0, 0.5, 0.5;
  (0, 0, 3) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1, 3) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2, 3) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3, 3) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 0, 4) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1, 4) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2, 4) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3, 4) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLW_APB_MUSIZE | R_LNLW_MED_SEV, R_LNLW_MED_TIME, R_LNLW_MED_PATHO ) {
  (0, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 0) : 0.0, 0.0, 0.9, 0.1, 0.0, 0.0;
  (2, 2, 0) : 0.0, 0.0, 0.2, 0.7, 0.1, 0.0;
  (3, 2, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (4, 2, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (0, 3, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 0) : 0.0, 0.0, 0.9, 0.1, 0.0, 0.0;
  (2, 3, 0) : 0.0, 0.0, 0.2, 0.7, 0.1, 0.0;
  (3, 3, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (4, 3, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (0, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 1) : 0.00, 0.00, 0.95, 0.05, 0.00, 0.00;
  (2, 2, 1) : 0.00, 0.00, 0.70, 0.25, 0.05, 0.00;
  (3, 2, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (4, 2, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (0, 3, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 1) : 0.00, 0.00, 0.95, 0.05, 0.00, 0.00;
  (2, 3, 1) : 0.00, 0.00, 0.70, 0.25, 0.05, 0.00;
  (3, 3, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (4, 3, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (0, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 2) : 0.0, 0.0, 0.8, 0.2, 0.0, 0.0;
  (2, 2, 2) : 0.0, 0.0, 0.0, 0.8, 0.2, 0.0;
  (3, 2, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (4, 2, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (0, 3, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 2) : 0.0, 0.0, 0.8, 0.2, 0.0, 0.0;
  (2, 3, 2) : 0.0, 0.0, 0.0, 0.8, 0.2, 0.0;
  (3, 3, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (4, 3, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (0, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_DIFFN_APB_MALOSS | DIFFN_M_SEV_DIST, DIFFN_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.4, 0.6, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.4, 0.6, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.4, 0.6, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 1) : 0.4, 0.3, 0.3, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 0.8, 0.2;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (0, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (1, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (2, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
}
probability ( R_LNLW_APB_MALOSS | R_LNLW_MED_SEV, R_LNLW_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.1, 0.9;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 1) : 0.4, 0.3, 0.3, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 1) : 0.25, 0.25, 0.25, 0.25, 0.00;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (0, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (1, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (2, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
}
probability ( R_LNLT1_LP_APB_MUSIZE | R_LNLLP_APB_MUSIZE, R_LNLT1_APB_MUSIZE ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (5, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (5, 1) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 5) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLBE_APB_MUSIZE | R_LNLBE_MED_SEV, R_LNLBE_MED_TIME, R_LNLBE_MED_PATHO ) {
  (0, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 0, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 1, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 0) : 0.0, 0.0, 0.9, 0.1, 0.0, 0.0;
  (2, 2, 0) : 0.0, 0.0, 0.2, 0.7, 0.1, 0.0;
  (3, 2, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (4, 2, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (0, 3, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 0) : 0.0, 0.0, 0.9, 0.1, 0.0, 0.0;
  (2, 3, 0) : 0.0, 0.0, 0.2, 0.7, 0.1, 0.0;
  (3, 3, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (4, 3, 0) : 0.0, 0.0, 0.0, 0.2, 0.7, 0.1;
  (0, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 0, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 1) : 0.00, 0.00, 0.95, 0.05, 0.00, 0.00;
  (2, 2, 1) : 0.00, 0.00, 0.70, 0.25, 0.05, 0.00;
  (3, 2, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (4, 2, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (0, 3, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 1) : 0.00, 0.00, 0.95, 0.05, 0.00, 0.00;
  (2, 3, 1) : 0.00, 0.00, 0.70, 0.25, 0.05, 0.00;
  (3, 3, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (4, 3, 1) : 0.00, 0.00, 0.00, 0.25, 0.70, 0.05;
  (0, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (0, 2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2, 2) : 0.0, 0.0, 0.8, 0.2, 0.0, 0.0;
  (2, 2, 2) : 0.0, 0.0, 0.0, 0.8, 0.2, 0.0;
  (3, 2, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (4, 2, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (0, 3, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3, 2) : 0.0, 0.0, 0.8, 0.2, 0.0, 0.0;
  (2, 3, 2) : 0.0, 0.0, 0.0, 0.8, 0.2, 0.0;
  (3, 3, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (4, 3, 2) : 0.0, 0.0, 0.0, 0.1, 0.6, 0.3;
  (0, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLBE_APB_MALOSS | R_LNLBE_MED_SEV, R_LNLBE_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.1, 0.9;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 1) : 0.4, 0.3, 0.3, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 1) : 0.25, 0.25, 0.25, 0.25, 0.00;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (0, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (1, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (2, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
}
probability ( R_LNLT1_LP_APB_MALOSS | R_LNLT1_APB_MALOSS, R_LNLLP_APB_MALOSS ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0022, 0.9977, 0.0001, 0.0000, 0.0000;
  (2, 0) : 0.000199980, 0.036896305, 0.958804124, 0.004099591, 0.000000000;
  (3, 0) : 0.0000, 0.0002, 0.0329, 0.9669, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0022, 0.9977, 0.0001, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0282, 0.9718, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0009, 0.3409, 0.6582, 0.0000;
  (3, 1) : 0.0000, 0.0000, 0.0038, 0.9962, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.000199980, 0.036896305, 0.958804124, 0.004099591, 0.000000000;
  (1, 2) : 0.0000, 0.0009, 0.3409, 0.6582, 0.0000;
  (2, 2) : 0.00, 0.00, 0.01, 0.99, 0.00;
  (3, 2) : 0.0000, 0.0000, 0.0011, 0.9989, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0000, 0.0002, 0.0329, 0.9669, 0.0000;
  (1, 3) : 0.0000, 0.0000, 0.0038, 0.9962, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0011, 0.9989, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0004, 0.9996, 0.0000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLT1_LP_BE_APB_MUSIZE | R_LNLBE_APB_MUSIZE, R_LNLT1_LP_APB_MUSIZE ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (5, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (5, 1) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 5) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_LNLW_APB_MUSIZE | R_LNLW_APB_MUSIZE, R_DIFFN_APB_MUSIZE ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (5, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (5, 1) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0000, 0.0000, 0.9981, 0.0019, 0.0000, 0.0000;
  (3, 2) : 0.0000, 0.0000, 0.0019, 0.9981, 0.0000, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 0.0000, 0.0000, 0.0019, 0.9981, 0.0000, 0.0000;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 5) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNL_DIFFN_APB_DENERV | R_LNLT1_LP_BE_APB_DENERV, R_DIFFN_LNLW_APB_DENERV ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.0, 1.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0;
  (1, 2) : 0.0, 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 0.0, 1.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MUSCLE_APB_DENERV | R_MYOP_MYDY_APB_DENERV, R_NMT_APB_DENERV ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.0, 1.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0;
  (1, 2) : 0.0, 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 0.0, 1.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_LNLW_APB_NEUR_ACT | R_DIFFN_APB_NEUR_ACT, R_LNLW_APB_NEUR_ACT ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLT1_LP_BE_APB_NEUR_ACT | R_LNLT1_LP_APB_NEUR_ACT, R_LNLBE_APB_NEUR_ACT ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MUSCLE_APB_MUDENS | R_MYOP_MYDY_APB_MUDENS, R_MYAS_APB_MUDENS ) {
  (0, 0) : 1.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0;
  (2, 1) : 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0;
  (1, 2) : 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 1.0;
}
probability ( R_LNL_DIFFN_APB_MUDENS | R_LNLT1_LP_BE_APB_MUDENS, R_DIFFN_LNLW_APB_MUDENS ) {
  (0, 0) : 1.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0;
  (2, 1) : 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0;
  (1, 2) : 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 1.0;
}
probability ( R_MYOP_MYDY_APB_DE_REGEN | R_MYOP_APB_DE_REGEN, R_MYDY_APB_DE_REGEN ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.0, 1.0;
}
probability ( R_LNL_DIFFN_APB_DE_REGEN | R_LNLT1_LP_BE_APB_DE_REGEN, R_DIFFN_LNLW_APB_DE_REGEN ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.0, 1.0;
}
probability ( R_LNLW_MED_SEV ) {
   0.73, 0.16, 0.07, 0.03, 0.01;
}
probability ( R_LNLW_MED_PATHO ) {
   0.800, 0.120, 0.070, 0.005, 0.005;
}
probability ( R_LNLBE_MED_DIFSLOW | R_LNLBE_MED_PATHO ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 1.0, 0.0, 0.0, 0.0;
  (2) : 1.0, 0.0, 0.0, 0.0;
  (3) : 0.0, 0.0, 0.0, 1.0;
  (4) : 0.0, 0.0, 1.0, 0.0;
}
probability ( R_LNLW_MED_BLOCK | R_LNLW_MED_SEV, R_LNLW_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.6, 0.4, 0.0, 0.0, 0.0;
  (3, 0) : 0.25, 0.50, 0.25, 0.00, 0.00;
  (4, 0) : 0.2, 0.2, 0.2, 0.2, 0.2;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.4, 0.5, 0.1;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2) : 0.2, 0.2, 0.2, 0.2, 0.2;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_MYAS_APB_NMT ) {
   1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_DE_REGEN_APB_NMT | R_APB_DE_REGEN ) {
  (0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1) : 0.65, 0.02, 0.01, 0.27, 0.02, 0.01, 0.02;
}
probability ( R_LNL_DIFFN_APB_MUSIZE | R_DIFFN_LNLW_APB_MUSIZE, R_LNLT1_LP_BE_APB_MUSIZE ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (5, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (5, 1) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0000, 0.0000, 0.9981, 0.0019, 0.0000, 0.0000;
  (3, 2) : 0.0000, 0.0000, 0.0019, 0.9981, 0.0000, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 0.0000, 0.0000, 0.0019, 0.9981, 0.0000, 0.0000;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 5) : 0.00000000, 0.99939994, 0.00050005, 0.00010001, 0.00000000, 0.00000000;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MYOP_MYDY_APB_MUSIZE | R_MYDY_APB_MUSIZE, R_MYOP_APB_MUSIZE ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.9983, 0.0017, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.9857, 0.0143, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 0) : 0.3673, 0.6298, 0.0029, 0.0000, 0.0000, 0.0000;
  (4, 0) : 0.011501207, 0.861686526, 0.124912076, 0.001900191, 0.000000000, 0.000000000;
  (5, 0) : 0.0000, 0.1596, 0.7368, 0.1016, 0.0020, 0.0000;
  (0, 1) : 0.9983, 0.0017, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.8667, 0.1329, 0.0004, 0.0000, 0.0000, 0.0000;
  (2, 1) : 0.01390141, 0.96369638, 0.02240221, 0.00000000, 0.00000000, 0.00000000;
  (3, 1) : 0.0003, 0.3514, 0.6035, 0.0443, 0.0005, 0.0000;
  (4, 1) : 0.0000, 0.0105, 0.5726, 0.3806, 0.0359, 0.0004;
  (5, 1) : 0.0000, 0.0000, 0.0792, 0.4758, 0.4066, 0.0384;
  (0, 2) : 0.9857, 0.0143, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 2) : 0.01390141, 0.96369638, 0.02240221, 0.00000000, 0.00000000, 0.00000000;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.00000000, 0.00000000, 0.04060409, 0.92779272, 0.03160319, 0.00000000;
  (4, 2) : 0.0000, 0.0000, 0.0000, 0.0319, 0.9362, 0.0319;
  (5, 2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0329, 0.9671;
  (0, 3) : 0.3673, 0.6298, 0.0029, 0.0000, 0.0000, 0.0000;
  (1, 3) : 0.0003, 0.3514, 0.6035, 0.0443, 0.0005, 0.0000;
  (2, 3) : 0.00000000, 0.00000000, 0.04060409, 0.92779272, 0.03160319, 0.00000000;
  (3, 3) : 0.00000000, 0.00000000, 0.00039996, 0.10988900, 0.77982204, 0.10988900;
  (4, 3) : 0.00000000, 0.00000000, 0.00000000, 0.00030003, 0.12341200, 0.87628797;
  (5, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0028, 0.9972;
  (0, 4) : 0.011501207, 0.861686526, 0.124912076, 0.001900191, 0.000000000, 0.000000000;
  (1, 4) : 0.0000, 0.0105, 0.5726, 0.3806, 0.0359, 0.0004;
  (2, 4) : 0.0000, 0.0000, 0.0000, 0.0319, 0.9362, 0.0319;
  (3, 4) : 0.00000000, 0.00000000, 0.00000000, 0.00030003, 0.12341200, 0.87628797;
  (4, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0028, 0.9972;
  (5, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999;
  (0, 5) : 0.0000, 0.1596, 0.7368, 0.1016, 0.0020, 0.0000;
  (1, 5) : 0.0000, 0.0000, 0.0792, 0.4758, 0.4066, 0.0384;
  (2, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0329, 0.9671;
  (3, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0028, 0.9972;
  (4, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_LNLW_APB_MALOSS | R_LNLW_APB_MALOSS, R_DIFFN_APB_MALOSS ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0022, 0.9977, 0.0001, 0.0000, 0.0000;
  (2, 0) : 0.000199980, 0.036896305, 0.958804124, 0.004099591, 0.000000000;
  (3, 0) : 0.0000, 0.0002, 0.0329, 0.9669, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0022, 0.9977, 0.0001, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0282, 0.9718, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0009, 0.3409, 0.6582, 0.0000;
  (3, 1) : 0.0000, 0.0000, 0.0038, 0.9962, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.000199980, 0.036896305, 0.958804124, 0.004099591, 0.000000000;
  (1, 2) : 0.0000, 0.0009, 0.3409, 0.6582, 0.0000;
  (2, 2) : 0.00, 0.00, 0.01, 0.99, 0.00;
  (3, 2) : 0.0000, 0.0000, 0.0011, 0.9989, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0000, 0.0002, 0.0329, 0.9669, 0.0000;
  (1, 3) : 0.0000, 0.0000, 0.0038, 0.9962, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0011, 0.9989, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0004, 0.9996, 0.0000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLT1_LP_BE_APB_MALOSS | R_LNLT1_LP_APB_MALOSS, R_LNLBE_APB_MALOSS ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0022, 0.9977, 0.0001, 0.0000, 0.0000;
  (2, 0) : 0.000199980, 0.036896305, 0.958804124, 0.004099591, 0.000000000;
  (3, 0) : 0.0000, 0.0002, 0.0329, 0.9669, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0022, 0.9977, 0.0001, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0282, 0.9718, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0009, 0.3409, 0.6582, 0.0000;
  (3, 1) : 0.0000, 0.0000, 0.0038, 0.9962, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.000199980, 0.036896305, 0.958804124, 0.004099591, 0.000000000;
  (1, 2) : 0.0000, 0.0009, 0.3409, 0.6582, 0.0000;
  (2, 2) : 0.00, 0.00, 0.01, 0.99, 0.00;
  (3, 2) : 0.0000, 0.0000, 0.0011, 0.9989, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0000, 0.0002, 0.0329, 0.9669, 0.0000;
  (1, 3) : 0.0000, 0.0000, 0.0038, 0.9962, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0011, 0.9989, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0004, 0.9996, 0.0000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_MED_DIFSLOW | DIFFN_M_SEV_DIST, DIFFN_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 0.5, 0.5, 0.0;
  (2, 0) : 0.0, 0.2, 0.6, 0.2;
  (3, 0) : 0.0, 0.2, 0.6, 0.2;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.4, 0.5, 0.1, 0.0;
  (2, 1) : 0.2, 0.4, 0.3, 0.1;
  (3, 1) : 0.2, 0.4, 0.3, 0.1;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.7, 0.3;
  (1, 4) : 0.0, 0.0, 0.7, 0.3;
  (2, 4) : 0.0, 0.0, 0.7, 0.3;
  (3, 4) : 0.0, 0.0, 0.7, 0.3;
}
probability ( R_LNLBE_MED_SEV ) {
   0.981, 0.010, 0.005, 0.003, 0.001;
}
probability ( R_LNLBE_MED_PATHO ) {
   0.600, 0.190, 0.200, 0.005, 0.005;
}
probability ( R_LNLBE_MED_BLOCK | R_LNLBE_MED_SEV, R_LNLBE_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.6, 0.4, 0.0, 0.0, 0.0;
  (3, 0) : 0.25, 0.50, 0.25, 0.00, 0.00;
  (4, 0) : 0.2, 0.2, 0.2, 0.2, 0.2;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.4, 0.5, 0.1;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2) : 0.2, 0.2, 0.2, 0.2, 0.2;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_DIFFN_MED_BLOCK | DIFFN_M_SEV_DIST, DIFFN_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.6, 0.4, 0.0, 0.0, 0.0;
  (3, 0) : 0.25, 0.50, 0.25, 0.00, 0.00;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.4, 0.5, 0.1;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLW_MEDD2_RD_WD | R_LNLW_MED_PATHO ) {
  (0) : 1.0, 0.0, 0.0;
  (1) : 1.0, 0.0, 0.0;
  (2) : 1.0, 0.0, 0.0;
  (3) : 0.0, 0.0, 1.0;
  (4) : 0.0, 1.0, 0.0;
}
probability ( R_LNLW_MEDD2_LD_WD | R_LNLW_MED_SEV, R_LNLW_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (4, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.25, 0.50, 0.25, 0.00;
  (3, 1) : 0.05, 0.30, 0.50, 0.15;
  (4, 1) : 0.25, 0.25, 0.25, 0.25;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0;
  (3, 2) : 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.5, 0.5, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0;
  (4, 4) : 1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLBE_MEDD2_DIFSLOW_WD | R_LNLBE_MED_PATHO ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 1.0, 0.0, 0.0, 0.0;
  (2) : 1.0, 0.0, 0.0, 0.0;
  (3) : 0.0, 0.0, 0.0, 1.0;
  (4) : 0.0, 0.0, 1.0, 0.0;
}
probability ( R_LNLW_MEDD2_BLOCK_WD | R_LNLW_MED_SEV, R_LNLW_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.8, 0.2, 0.0, 0.0, 0.0;
  (2, 0) : 0.3, 0.6, 0.1, 0.0, 0.0;
  (3, 0) : 0.1, 0.5, 0.3, 0.1, 0.0;
  (4, 0) : 0.00, 0.05, 0.20, 0.55, 0.20;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 1) : 0.0, 0.0, 0.2, 0.6, 0.2;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_DIFFN_LNLW_MEDD2_DISP_WD | R_LNLW_MEDD2_DISP_WD, R_DIFFN_MEDD2_DISP ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0749, 0.8229, 0.1022, 0.0000;
  (2, 0) : 0.00000000, 0.06260628, 0.93739372, 0.00000000;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0749, 0.8229, 0.1022, 0.0000;
  (1, 1) : 0.0047, 0.1786, 0.8167, 0.0000;
  (2, 1) : 0.0000, 0.0190, 0.9795, 0.0015;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.00000000, 0.06260628, 0.93739372, 0.00000000;
  (1, 2) : 0.0000, 0.0190, 0.9795, 0.0015;
  (2, 2) : 0.0000, 0.0001, 0.0833, 0.9166;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_LNLBE_MEDD2_SALOSS_EW | R_LNLBE_MED_SEV, R_LNLBE_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.4, 0.6;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0, 0.0;
  (2, 1) : 0.2, 0.5, 0.3, 0.0, 0.0;
  (3, 1) : 0.0, 0.2, 0.5, 0.3, 0.0;
  (4, 1) : 0.0, 0.1, 0.4, 0.4, 0.1;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 1.0, 0.0;
  (0, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (1, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (2, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (4, 4) : 0.0, 0.0, 0.5, 0.5, 0.0;
}
probability ( R_DIFFN_LNLW_MEDD2_SALOSS | R_LNLW_MEDD2_SALOSS_WD, R_DIFFN_MEDD2_SALOSS ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0073, 0.9812, 0.0115, 0.0000, 0.0000;
  (2, 0) : 0.0017, 0.1306, 0.8670, 0.0007, 0.0000;
  (3, 0) : 0.0000, 0.0003, 0.0212, 0.9785, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0073, 0.9812, 0.0115, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0289, 0.9711, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0097, 0.5989, 0.3914, 0.0000;
  (3, 1) : 0.0000, 0.0000, 0.0017, 0.9983, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0017, 0.1306, 0.8670, 0.0007, 0.0000;
  (1, 2) : 0.0000, 0.0097, 0.5989, 0.3914, 0.0000;
  (2, 2) : 0.0000000000, 0.0004999502, 0.0330967116, 0.9664033382, 0.0000000000;
  (3, 2) : 0.0000, 0.0000, 0.0008, 0.9992, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0000, 0.0003, 0.0212, 0.9785, 0.0000;
  (1, 3) : 0.0000, 0.0000, 0.0017, 0.9983, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0008, 0.9992, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0001, 0.9999, 0.0000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_MEDD2_DIFSLOW | DIFFN_S_SEV_DIST, DIFFN_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 0.5, 0.5, 0.0;
  (2, 0) : 0.0, 0.2, 0.6, 0.2;
  (3, 0) : 0.0, 0.1, 0.6, 0.3;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.2, 0.4, 0.3, 0.1;
  (3, 1) : 0.2, 0.4, 0.3, 0.1;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.7, 0.3;
  (1, 4) : 0.0, 0.0, 0.7, 0.3;
  (2, 4) : 0.0, 0.0, 0.7, 0.3;
  (3, 4) : 0.0, 0.0, 0.7, 0.3;
}
probability ( R_LNLBE_MEDD2_RD_EW | R_LNLBE_MED_PATHO ) {
  (0) : 1.0, 0.0, 0.0;
  (1) : 1.0, 0.0, 0.0;
  (2) : 1.0, 0.0, 0.0;
  (3) : 0.0, 0.0, 1.0;
  (4) : 0.0, 1.0, 0.0;
}
probability ( R_LNLBE_MEDD2_LD_EW | R_LNLBE_MED_SEV, R_LNLBE_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (4, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.25, 0.50, 0.25, 0.00;
  (3, 1) : 0.05, 0.30, 0.50, 0.15;
  (4, 1) : 0.25, 0.25, 0.25, 0.25;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0;
  (3, 2) : 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.5, 0.5, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0;
  (4, 4) : 1.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLBE_MEDD2_BLOCK_EW | R_LNLBE_MED_SEV, R_LNLBE_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.8, 0.2, 0.0, 0.0, 0.0;
  (2, 0) : 0.3, 0.6, 0.1, 0.0, 0.0;
  (3, 0) : 0.1, 0.5, 0.3, 0.1, 0.0;
  (4, 0) : 0.00, 0.05, 0.20, 0.55, 0.20;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.5, 0.5, 0.0;
  (3, 1) : 0.0, 0.0, 0.2, 0.6, 0.2;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (4, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_DIFFN_MEDD2_BLOCK | DIFFN_S_SEV_DIST, DIFFN_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.8, 0.2, 0.0, 0.0, 0.0;
  (2, 0) : 0.6, 0.4, 0.0, 0.0, 0.0;
  (3, 0) : 0.25, 0.50, 0.25, 0.00, 0.00;
  (0, 1) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.2, 0.5, 0.3, 0.0, 0.0;
  (2, 1) : 0.0, 0.2, 0.5, 0.3, 0.0;
  (3, 1) : 0.0, 0.0, 0.4, 0.5, 0.1;
  (0, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_LNLBE_MEDD2_DISP_EW | R_LNLBE_MED_SEV, R_LNLBE_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (4, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.5, 0.5, 0.0;
  (3, 1) : 0.0, 0.1, 0.5, 0.4;
  (4, 1) : 0.0, 0.0, 0.5, 0.5;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0;
  (3, 2) : 0.3, 0.5, 0.2, 0.0;
  (4, 2) : 0.0, 0.5, 0.5, 0.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
  (4, 3) : 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_DIFFN_MEDD2_DISP | DIFFN_S_SEV_DIST, DIFFN_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.0, 0.5, 0.5, 0.0;
  (3, 1) : 0.0, 0.1, 0.5, 0.4;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0;
  (3, 2) : 0.0, 0.5, 0.5, 0.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MED_RD_WA | R_LNLW_MED_PATHO ) {
  (0) : 1.0, 0.0, 0.0;
  (1) : 1.0, 0.0, 0.0;
  (2) : 1.0, 0.0, 0.0;
  (3) : 0.0, 0.0, 1.0;
  (4) : 0.0, 1.0, 0.0;
}
probability ( R_MED_LD_WA | R_LNLW_MED_SEV, R_LNLW_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (4, 0) : 0.25, 0.25, 0.25, 0.25;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.25, 0.50, 0.25, 0.00;
  (3, 1) : 0.05, 0.30, 0.50, 0.15;
  (4, 1) : 0.25, 0.25, 0.25, 0.25;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0;
  (3, 2) : 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.25, 0.25, 0.25, 0.25;
  (0, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0;
  (4, 3) : 0.25, 0.25, 0.25, 0.25;
  (0, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0;
  (4, 4) : 0.25, 0.25, 0.25, 0.25;
}
probability ( R_MED_DIFSLOW_WA | R_LNLBE_MED_DIFSLOW, R_DIFFN_MED_DIFSLOW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0131987048, 0.9862013550, 0.0005999402, 0.0000000000;
  (2, 0) : 0.0000000, 0.0181018, 0.9818982, 0.0000000;
  (3, 0) : 0.0000, 0.0003, 0.0252, 0.9745;
  (0, 1) : 0.0131987048, 0.9862013550, 0.0005999402, 0.0000000000;
  (1, 1) : 0.0001, 0.0952, 0.9047, 0.0000;
  (2, 1) : 0.0000, 0.0009, 0.5880, 0.4111;
  (3, 1) : 0.000000000, 0.000000000, 0.004400438, 0.995599562;
  (0, 2) : 0.0000000, 0.0181018, 0.9818982, 0.0000000;
  (1, 2) : 0.0000, 0.0009, 0.5880, 0.4111;
  (2, 2) : 0.000, 0.000, 0.002, 0.998;
  (3, 2) : 0.0000, 0.0000, 0.0006, 0.9994;
  (0, 3) : 0.0000, 0.0003, 0.0252, 0.9745;
  (1, 3) : 0.000000000, 0.000000000, 0.004400438, 0.995599562;
  (2, 3) : 0.0000, 0.0000, 0.0006, 0.9994;
  (3, 3) : 0.0000, 0.0000, 0.0005, 0.9995;
}
probability ( R_MED_BLOCK_WA | R_DIFFN_MED_BLOCK, R_LNLW_MED_BLOCK ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (3, 0) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0001, 0.4980, 0.5019, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (3, 1) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (1, 2) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0007, 0.4328, 0.5665, 0.0000;
  (3, 2) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (1, 3) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (2, 3) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (3, 3) : 0.0003000299, 0.0011001095, 0.0125012945, 0.9860985661, 0.0000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MED_DIFSLOW_EW | R_DIFFN_MED_DIFSLOW ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 0.0126, 0.9869, 0.0005, 0.0000;
  (2) : 0.0000, 0.0179, 0.9821, 0.0000;
  (3) : 0.0000, 0.0003, 0.0252, 0.9745;
}
probability ( R_MED_RD_EW | R_LNLBE_MED_PATHO ) {
  (0) : 1.0, 0.0, 0.0;
  (1) : 1.0, 0.0, 0.0;
  (2) : 1.0, 0.0, 0.0;
  (3) : 0.0, 0.0, 1.0;
  (4) : 0.0, 1.0, 0.0;
}
probability ( R_MED_LD_EW | R_LNLBE_MED_SEV, R_LNLBE_MED_PATHO ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (4, 0) : 0.25, 0.25, 0.25, 0.25;
  (0, 1) : 1.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.5, 0.5, 0.0, 0.0;
  (2, 1) : 0.25, 0.50, 0.25, 0.00;
  (3, 1) : 0.05, 0.30, 0.50, 0.15;
  (4, 1) : 0.25, 0.25, 0.25, 0.25;
  (0, 2) : 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.5, 0.5, 0.0, 0.0;
  (3, 2) : 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.25, 0.25, 0.25, 0.25;
  (0, 3) : 1.0, 0.0, 0.0, 0.0;
  (1, 3) : 1.0, 0.0, 0.0, 0.0;
  (2, 3) : 1.0, 0.0, 0.0, 0.0;
  (3, 3) : 1.0, 0.0, 0.0, 0.0;
  (4, 3) : 0.25, 0.25, 0.25, 0.25;
  (0, 4) : 1.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0;
  (4, 4) : 0.25, 0.25, 0.25, 0.25;
}
probability ( R_MEDD2_RD_WD | R_LNLW_MEDD2_RD_WD ) {
  (0) : 1.0, 0.0, 0.0;
  (1) : 0.0, 1.0, 0.0;
  (2) : 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_LD_WD | R_LNLW_MEDD2_LD_WD ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 0.0, 1.0, 0.0, 0.0;
  (2) : 0.0, 0.0, 1.0, 0.0;
  (3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_DIFSLOW_WD | R_LNLBE_MEDD2_DIFSLOW_WD, R_DIFFN_MEDD2_DIFSLOW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0127, 0.9867, 0.0006, 0.0000;
  (2, 0) : 0.0000000, 0.0181018, 0.9818982, 0.0000000;
  (3, 0) : 0.0000, 0.0006, 0.0492, 0.9502;
  (0, 1) : 0.0127, 0.9867, 0.0006, 0.0000;
  (1, 1) : 0.0001, 0.0952, 0.9047, 0.0000;
  (2, 1) : 0.0000, 0.0011, 0.7402, 0.2587;
  (3, 1) : 0.000000000, 0.000000000, 0.008800881, 0.991199119;
  (0, 2) : 0.0000000, 0.0181018, 0.9818982, 0.0000000;
  (1, 2) : 0.0000, 0.0011, 0.7402, 0.2587;
  (2, 2) : 0.000, 0.000, 0.004, 0.996;
  (3, 2) : 0.0000, 0.0000, 0.0012, 0.9988;
  (0, 3) : 0.0000, 0.0006, 0.0492, 0.9502;
  (1, 3) : 0.000000000, 0.000000000, 0.008800881, 0.991199119;
  (2, 3) : 0.0000, 0.0000, 0.0012, 0.9988;
  (3, 3) : 0.0000, 0.0000, 0.0009, 0.9991;
}
probability ( R_MEDD2_BLOCK_WD | R_LNLW_MEDD2_BLOCK_WD, R_DIFFN_MEDD2_BLOCK ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (3, 0) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0001, 0.4980, 0.5019, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (3, 1) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (1, 2) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0007, 0.4328, 0.5665, 0.0000;
  (3, 2) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (1, 3) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (2, 3) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (3, 3) : 0.0003000299, 0.0011001095, 0.0125012945, 0.9860985661, 0.0000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_DIFSLOW_EW | R_DIFFN_MEDD2_DIFSLOW ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 0.0, 1.0, 0.0, 0.0;
  (2) : 0.0, 0.0, 1.0, 0.0;
  (3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_SALOSS | R_DIFFN_LNLW_MEDD2_SALOSS, R_LNLBE_MEDD2_SALOSS_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0073, 0.9812, 0.0115, 0.0000, 0.0000;
  (2, 0) : 0.0017, 0.1306, 0.8670, 0.0007, 0.0000;
  (3, 0) : 0.0000, 0.0003, 0.0212, 0.9785, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0073, 0.9812, 0.0115, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0289, 0.9711, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0097, 0.5989, 0.3914, 0.0000;
  (3, 1) : 0.0000, 0.0000, 0.0017, 0.9983, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0017, 0.1306, 0.8670, 0.0007, 0.0000;
  (1, 2) : 0.0000, 0.0097, 0.5989, 0.3914, 0.0000;
  (2, 2) : 0.0000000000, 0.0004999502, 0.0330967116, 0.9664033382, 0.0000000000;
  (3, 2) : 0.0000, 0.0000, 0.0008, 0.9992, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0000, 0.0003, 0.0212, 0.9785, 0.0000;
  (1, 3) : 0.0000, 0.0000, 0.0017, 0.9983, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0008, 0.9992, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0001, 0.9999, 0.0000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_RD_EW | R_LNLBE_MEDD2_RD_EW ) {
  (0) : 1.0, 0.0, 0.0;
  (1) : 0.0, 1.0, 0.0;
  (2) : 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_LD_EW | R_LNLBE_MEDD2_LD_EW ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 0.0, 1.0, 0.0, 0.0;
  (2) : 0.0, 0.0, 1.0, 0.0;
  (3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MED_DCV_WA | R_APB_MALOSS, R_MED_DIFSLOW_WA ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.1136, 0.8864, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0006, 0.0764, 0.8866, 0.0364, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 0) : 0.0000, 0.0000, 0.0655, 0.9299, 0.0046, 0.0000, 0.0000, 0.0000, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 0) : 0.1523, 0.2904, 0.3678, 0.1726, 0.0169, 0.0000, 0.0000, 0.0000, 0.0000;
  (0, 1) : 0.0080, 0.1680, 0.7407, 0.0833, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.00089991, 0.03679630, 0.55764401, 0.40246000, 0.00219978, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (2, 1) : 0.0000000000, 0.0007000697, 0.0525052790, 0.5712567715, 0.3752378499, 0.0003000299, 0.0000000000, 0.0000000000, 0.0000000000;
  (3, 1) : 0.0000, 0.0000, 0.0007, 0.0745, 0.8859, 0.0389, 0.0000, 0.0000, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 1) : 0.0168, 0.0618, 0.2223, 0.4015, 0.2803, 0.0172, 0.0001, 0.0000, 0.0000;
  (0, 2) : 0.0006998602, 0.0058988219, 0.0614877197, 0.3006400962, 0.5525891768, 0.0781844250, 0.0004999002, 0.0000000000, 0.0000000000;
  (1, 2) : 0.0002, 0.0018, 0.0263, 0.1887, 0.5884, 0.1916, 0.0030, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0001, 0.0024, 0.0358, 0.3160, 0.5741, 0.0716, 0.0000, 0.0000;
  (3, 2) : 0.000000e+00, 0.000000e+00, 9.998992e-05, 3.199677e-03, 7.809214e-02, 5.946405e-01, 3.235677e-01, 3.999597e-04, 0.000000e+00;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 2) : 0.0009998995, 0.0046995277, 0.0279971860, 0.1183879408, 0.3646638176, 0.3922608038, 0.0906908547, 0.0002999699, 0.0000000000;
  (0, 3) : 0.0001, 0.0003, 0.0018, 0.0110, 0.0670, 0.2945, 0.5047, 0.1206, 0.0000;
  (1, 3) : 0.0000000000, 0.0001000100, 0.0009000901, 0.0059005906, 0.0422042046, 0.2356240259, 0.5225520576, 0.1927190212, 0.0000000000;
  (2, 3) : 0.0000, 0.0000, 0.0001, 0.0012, 0.0121, 0.1131, 0.4415, 0.4320, 0.0000;
  (3, 3) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0002000201, 0.0028002808, 0.0439044132, 0.2917290875, 0.6613661984, 0.0000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 3) : 0.0000, 0.0002, 0.0011, 0.0057, 0.0332, 0.1704, 0.4424, 0.3470, 0.0000;
}
probability ( R_MED_RDLDDEL | R_MED_LD_WA, R_MED_RD_WA ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0964, 0.7981, 0.1055, 0.0000, 0.0000;
  (2, 0) : 0.0032, 0.1270, 0.8698, 0.0000, 0.0000;
  (3, 0) : 0.0009000901, 0.0028002804, 0.0147015019, 0.9815981276, 0.0000000000;
  (0, 1) : 0.0019, 0.5257, 0.4724, 0.0000, 0.0000;
  (1, 1) : 0.00019998, 0.04149590, 0.95830412, 0.00000000, 0.00000000;
  (2, 1) : 0.0001, 0.0144, 0.9855, 0.0000, 0.0000;
  (3, 1) : 0.0001999801, 0.0005999403, 0.0036996317, 0.9955004479, 0.0000000000;
  (0, 2) : 0.0002, 0.0304, 0.9694, 0.0000, 0.0000;
  (1, 2) : 0.0001, 0.0142, 0.9857, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0090, 0.9808, 0.0102, 0.0000;
  (3, 2) : 0.0000, 0.0002, 0.0012, 0.9984, 0.0002;
}
probability ( R_MED_RDLDCV_EW | R_MED_LD_EW, R_MED_RD_EW ) {
  (0, 0) : 0.9044904342, 0.0953095457, 0.0002000201, 0.0000000000, 0.0000000000, 0.0000000000;
  (1, 0) : 0.1320, 0.6039, 0.2641, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0139, 0.1839, 0.8022, 0.0000, 0.0000, 0.0000;
  (3, 0) : 0.00120012, 0.00670067, 0.05440540, 0.86008601, 0.07760780, 0.00000000;
  (0, 1) : 0.0115, 0.0333, 0.1509, 0.7319, 0.0724, 0.0000;
  (1, 1) : 0.0034003384, 0.0122011941, 0.0690068669, 0.7196716546, 0.1953199062, 0.0004000398;
  (2, 1) : 0.0011, 0.0045, 0.0299, 0.5742, 0.3876, 0.0027;
  (3, 1) : 0.0001, 0.0002, 0.0018, 0.0914, 0.6093, 0.2972;
  (0, 2) : 0.000000e+00, 9.999007e-05, 1.099891e-03, 1.461851e-01, 8.070196e-01, 4.559543e-02;
  (1, 2) : 0.0000, 0.0000, 0.0002, 0.0581, 0.7950, 0.1467;
  (2, 2) : 0.0000, 0.0000, 0.0001, 0.0228, 0.6344, 0.3427;
  (3, 2) : 0.0000, 0.0000, 0.0000, 0.0014, 0.1063, 0.8923;
}
probability ( R_MED_DCV_EW | R_APB_MALOSS, R_MED_DIFSLOW_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.1090, 0.8903, 0.0007, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.003999597, 0.114388920, 0.862113397, 0.019498086, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (3, 0) : 0.0001, 0.0028, 0.0640, 0.9243, 0.0088, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 0) : 0.0835, 0.1153, 0.2417, 0.3746, 0.1682, 0.0167, 0.0000, 0.0000, 0.0000, 0.0000;
  (0, 1) : 0.004100411, 0.024702507, 0.154615045, 0.738974214, 0.077607823, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (1, 1) : 0.0011, 0.0082, 0.0683, 0.7087, 0.2135, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 1) : 9.999002e-05, 7.999202e-04, 8.999102e-03, 3.029701e-01, 6.654331e-01, 2.169780e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (3, 1) : 0.0000, 0.0000, 0.0006, 0.0547, 0.6199, 0.3247, 0.0001, 0.0000, 0.0000, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 1) : 9.299071e-03, 1.809820e-02, 5.459451e-02, 2.276770e-01, 3.933611e-01, 2.775720e-01, 1.929810e-02, 9.999001e-05, 0.000000e+00, 0.000000e+00;
  (0, 2) : 0.0004, 0.0012, 0.0055, 0.0628, 0.2950, 0.5485, 0.0861, 0.0005, 0.0000, 0.0000;
  (1, 2) : 0.0002000201, 0.0005000503, 0.0028002817, 0.0389039233, 0.2309231386, 0.5829583498, 0.1422140853, 0.0015001509, 0.0000000000, 0.0000000000;
  (2, 2) : 0.0000000000, 0.0001000100, 0.0006000598, 0.0123011954, 0.1129109582, 0.5272528049, 0.3355338759, 0.0113010958, 0.0000000000, 0.0000000000;
  (3, 2) : 0.000000e+00, 0.000000e+00, 9.998995e-05, 2.499749e-03, 3.599638e-02, 3.240678e-01, 5.710427e-01, 6.629336e-02, 0.000000e+00, 0.000000e+00;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 2) : 0.00059994, 0.00119988, 0.00419958, 0.02839720, 0.11488900, 0.35756401, 0.39636002, 0.09639040, 0.00039996, 0.00000000;
  (0, 3) : 0.000000e+00, 9.998998e-05, 1.999800e-04, 1.899810e-03, 1.069890e-02, 6.579339e-02, 2.845720e-01, 5.054490e-01, 1.312870e-01, 0.000000e+00;
  (1, 3) : 0.00000000, 0.00000000, 0.00010001, 0.00120012, 0.00750075, 0.05100510, 0.25242501, 0.51855201, 0.16921700, 0.00000000;
  (2, 3) : 0.0000, 0.0000, 0.0000, 0.0005, 0.0034, 0.0280, 0.1822, 0.5098, 0.2761, 0.0000;
  (3, 3) : 0.000000000, 0.000000000, 0.000000000, 0.000100010, 0.001200119, 0.012501295, 0.111810952, 0.445244809, 0.429142815, 0.000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 3) : 0.0000, 0.0000, 0.0002, 0.0011, 0.0056, 0.0330, 0.1653, 0.4414, 0.3534, 0.0000;
}
probability ( R_MEDD2_DSLOW_EW | R_MEDD2_SALOSS, R_MEDD2_DIFSLOW_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.1036, 0.8964, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0005999402, 0.9973002692, 0.0020997906, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (3, 0) : 0.0001, 0.0629, 0.9278, 0.0092, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0532, 0.2387, 0.4950, 0.2072, 0.0059, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.01780179, 0.11901190, 0.44814464, 0.38543869, 0.02960298, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (2, 1) : 0.0048, 0.0476, 0.3148, 0.5311, 0.1016, 0.0001, 0.0000, 0.0000, 0.0000;
  (3, 1) : 0.0006000597, 0.0092009157, 0.1160119455, 0.4857487717, 0.3835378197, 0.0049004877, 0.0000000000, 0.0000000000, 0.0000000000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0007, 0.0063, 0.0614, 0.3006, 0.5524, 0.0781, 0.0005, 0.0000, 0.0000;
  (1, 2) : 0.0002, 0.0021, 0.0283, 0.1995, 0.5939, 0.1737, 0.0023, 0.0000, 0.0000;
  (2, 2) : 0.0000000000, 0.0006000606, 0.0114011106, 0.1147111067, 0.5445545063, 0.3196322973, 0.0091009185, 0.0000000000, 0.0000000000;
  (3, 2) : 0.00000000, 0.00010001, 0.00240024, 0.03740370, 0.33273302, 0.56705703, 0.06030600, 0.00000000, 0.00000000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0003, 0.0011, 0.0050, 0.0205, 0.0870, 0.2822, 0.4514, 0.1525, 0.0000;
  (1, 3) : 0.0002, 0.0006, 0.0029, 0.0133, 0.0634, 0.2436, 0.4657, 0.2103, 0.0000;
  (2, 3) : 0.0001000100, 0.0003000301, 0.0017001708, 0.0083008338, 0.0447045206, 0.2031200934, 0.4632462131, 0.2785281281, 0.0000000000;
  (3, 3) : 0.0000000000, 0.0001000100, 0.0007000701, 0.0038003805, 0.0243024034, 0.1416140198, 0.4240420594, 0.4054410568, 0.0000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_LSLOW_EW | R_MEDD2_LD_EW, R_MEDD2_RD_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0185, 0.9561, 0.0254, 0.0000, 0.0000;
  (2, 0) : 0.0000, 0.0166, 0.9834, 0.0000, 0.0000;
  (3, 0) : 0.0007, 0.0020, 0.0219, 0.9754, 0.0000;
  (0, 1) : 0.0084008376, 0.0119011967, 0.0619061827, 0.9173917431, 0.0004000399;
  (1, 1) : 0.0069006888, 0.0100009982, 0.0535053904, 0.9286928328, 0.0009000898;
  (2, 1) : 0.005599441, 0.008299171, 0.045195504, 0.938906084, 0.001999800;
  (3, 1) : 0.0023, 0.0036, 0.0217, 0.8326, 0.1398;
  (0, 2) : 0.005299469, 0.006199378, 0.026397393, 0.208178948, 0.753924812;
  (1, 2) : 0.0049, 0.0057, 0.0244, 0.1966, 0.7684;
  (2, 2) : 0.004400444, 0.005200524, 0.022402219, 0.183918154, 0.784078659;
  (3, 2) : 0.0028, 0.0033, 0.0145, 0.1304, 0.8490;
}
probability ( R_MEDD2_DSLOW_WD | R_MEDD2_SALOSS, R_MEDD2_DIFSLOW_WD ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.1036, 0.8964, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0005999402, 0.9973002692, 0.0020997906, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (3, 0) : 0.0001, 0.0629, 0.9278, 0.0092, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0532, 0.2387, 0.4950, 0.2072, 0.0059, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.01780179, 0.11901190, 0.44814464, 0.38543869, 0.02960298, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (2, 1) : 0.0048, 0.0476, 0.3148, 0.5311, 0.1016, 0.0001, 0.0000, 0.0000, 0.0000;
  (3, 1) : 0.0006000597, 0.0092009157, 0.1160119455, 0.4857487717, 0.3835378197, 0.0049004877, 0.0000000000, 0.0000000000, 0.0000000000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0007, 0.0063, 0.0614, 0.3006, 0.5524, 0.0781, 0.0005, 0.0000, 0.0000;
  (1, 2) : 0.0002, 0.0021, 0.0283, 0.1995, 0.5939, 0.1737, 0.0023, 0.0000, 0.0000;
  (2, 2) : 0.0000000000, 0.0006000606, 0.0114011106, 0.1147111067, 0.5445545063, 0.3196322973, 0.0091009185, 0.0000000000, 0.0000000000;
  (3, 2) : 0.00000000, 0.00010001, 0.00240024, 0.03740370, 0.33273302, 0.56705703, 0.06030600, 0.00000000, 0.00000000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0003, 0.0011, 0.0050, 0.0205, 0.0870, 0.2822, 0.4514, 0.1525, 0.0000;
  (1, 3) : 0.0002, 0.0006, 0.0029, 0.0133, 0.0634, 0.2436, 0.4657, 0.2103, 0.0000;
  (2, 3) : 0.0001000100, 0.0003000301, 0.0017001708, 0.0083008338, 0.0447045206, 0.2031200934, 0.4632462131, 0.2785281281, 0.0000000000;
  (3, 3) : 0.0000000000, 0.0001000100, 0.0007000701, 0.0038003805, 0.0243024034, 0.1416140198, 0.4240420594, 0.4054410568, 0.0000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_LSLOW_WD | R_MEDD2_LD_WD, R_MEDD2_RD_WD ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0185, 0.9561, 0.0254, 0.0000, 0.0000;
  (2, 0) : 0.0000, 0.0166, 0.9834, 0.0000, 0.0000;
  (3, 0) : 0.0007, 0.0020, 0.0219, 0.9754, 0.0000;
  (0, 1) : 0.0021, 0.0042, 0.0295, 0.9642, 0.0000;
  (1, 1) : 0.0012, 0.0025, 0.0194, 0.9769, 0.0000;
  (2, 1) : 0.0006999303, 0.0014998506, 0.0119988050, 0.9858014141, 0.0000000000;
  (3, 1) : 0.0002000201, 0.0005000502, 0.0046004621, 0.9944994475, 0.0002000201;
  (0, 2) : 0.0001, 0.0002, 0.0014, 0.0830, 0.9153;
  (1, 2) : 0.0001, 0.0001, 0.0008, 0.0533, 0.9457;
  (2, 2) : 0.0000, 0.0001, 0.0004, 0.0319, 0.9676;
  (3, 2) : 0.000000000, 0.000000000, 0.000000000, 0.003500349, 0.996499651;
}
probability ( R_MEDD2_EFFAXLOSS | R_MEDD2_BLOCK_WD, R_MEDD2_SALOSS ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0073, 0.9812, 0.0115, 0.0000, 0.0000;
  (2, 0) : 0.0017, 0.1306, 0.8670, 0.0007, 0.0000;
  (3, 0) : 0.0000, 0.0003, 0.0212, 0.9785, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0073, 0.9812, 0.0115, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0289, 0.9711, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0097, 0.5989, 0.3914, 0.0000;
  (3, 1) : 0.0000, 0.0000, 0.0017, 0.9983, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0017, 0.1306, 0.8670, 0.0007, 0.0000;
  (1, 2) : 0.0000, 0.0097, 0.5989, 0.3914, 0.0000;
  (2, 2) : 0.0000000000, 0.0004999502, 0.0330967116, 0.9664033382, 0.0000000000;
  (3, 2) : 0.0000, 0.0000, 0.0008, 0.9992, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0000, 0.0003, 0.0212, 0.9785, 0.0000;
  (1, 3) : 0.0000, 0.0000, 0.0017, 0.9983, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0008, 0.9992, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0001, 0.9999, 0.0000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_DISP_EW | R_DIFFN_MEDD2_DISP, R_LNLBE_MEDD2_DISP_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0749, 0.8229, 0.1022, 0.0000;
  (2, 0) : 0.00000000, 0.06260628, 0.93739372, 0.00000000;
  (3, 0) : 0.000000000, 0.007900792, 0.602860126, 0.389239082;
  (0, 1) : 0.0749, 0.8229, 0.1022, 0.0000;
  (1, 1) : 0.0047, 0.1786, 0.8167, 0.0000;
  (2, 1) : 0.0000, 0.0190, 0.9795, 0.0015;
  (3, 1) : 0.0000, 0.0001, 0.0069, 0.9930;
  (0, 2) : 0.00000000, 0.06260628, 0.93739372, 0.00000000;
  (1, 2) : 0.0000, 0.0190, 0.9795, 0.0015;
  (2, 2) : 0.0000, 0.0001, 0.0833, 0.9166;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.000000000, 0.007900792, 0.602860126, 0.389239082;
  (1, 3) : 0.0000, 0.0001, 0.0069, 0.9930;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_DISP_WD | R_DIFFN_LNLW_MEDD2_DISP_WD ) {
  (0) : 1.0, 0.0, 0.0, 0.0;
  (1) : 0.0, 1.0, 0.0, 0.0;
  (2) : 0.0, 0.0, 1.0, 0.0;
  (3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_APB_SPONT_INS_ACT | R_APB_DENERV ) {
  (0) : 0.98, 0.02;
  (1) : 0.1, 0.9;
  (2) : 0.05, 0.95;
  (3) : 0.05, 0.95;
}
probability ( R_APB_SPONT_HF_DISCH | R_APB_DENERV ) {
  (0) : 0.99, 0.01;
  (1) : 0.97, 0.03;
  (2) : 0.95, 0.05;
  (3) : 0.93, 0.07;
}
probability ( R_APB_DENERV | R_MUSCLE_APB_DENERV, R_LNL_DIFFN_APB_DENERV ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0;
  (2, 1) : 0.0, 0.0, 0.0, 1.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0;
  (1, 2) : 0.0, 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 0.0, 1.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_APB_SPONT_DENERV_ACT | R_APB_DENERV ) {
  (0) : 0.98, 0.02, 0.00, 0.00;
  (1) : 0.07, 0.85, 0.08, 0.00;
  (2) : 0.01, 0.07, 0.85, 0.07;
  (3) : 0.00, 0.01, 0.07, 0.92;
}
probability ( R_APB_NEUR_ACT | R_LNLT1_LP_BE_APB_NEUR_ACT, R_DIFFN_LNLW_APB_NEUR_ACT ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_APB_SPONT_NEUR_DISCH | R_APB_NEUR_ACT ) {
  (0) : 0.98, 0.02, 0.00, 0.00, 0.00, 0.00;
  (1) : 0.1, 0.9, 0.0, 0.0, 0.0, 0.0;
  (2) : 0.01, 0.04, 0.75, 0.05, 0.05, 0.10;
  (3) : 0.01, 0.04, 0.05, 0.75, 0.05, 0.10;
  (4) : 0.01, 0.04, 0.05, 0.05, 0.75, 0.10;
  (5) : 0.01, 0.05, 0.05, 0.05, 0.05, 0.79;
}
probability ( R_APB_MUDENS | R_LNL_DIFFN_APB_MUDENS, R_MUSCLE_APB_MUDENS ) {
  (0, 0) : 1.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0;
  (2, 1) : 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0;
  (1, 2) : 0.0, 0.0, 1.0;
  (2, 2) : 0.0, 0.0, 1.0;
}
probability ( R_APB_SF_DENSITY | R_APB_MUDENS ) {
  (0) : 0.97, 0.03, 0.00;
  (1) : 0.05, 0.90, 0.05;
  (2) : 0.01, 0.04, 0.95;
}
probability ( R_APB_SF_JITTER | R_APB_NMT ) {
  (0) : 0.95, 0.05, 0.00, 0.00;
  (1) : 0.02, 0.20, 0.70, 0.08;
  (2) : 0.0, 0.1, 0.4, 0.5;
  (3) : 0.05, 0.70, 0.20, 0.05;
  (4) : 0.01, 0.19, 0.70, 0.10;
  (5) : 0.0, 0.1, 0.4, 0.5;
  (6) : 0.1, 0.3, 0.3, 0.3;
}
probability ( R_APB_REPSTIM_POST_DECR | R_APB_NMT ) {
  (0) : 0.949, 0.020, 0.010, 0.001, 0.020;
  (1) : 0.02, 0.10, 0.80, 0.06, 0.02;
  (2) : 0.001, 0.010, 0.020, 0.949, 0.020;
  (3) : 0.25, 0.61, 0.10, 0.02, 0.02;
  (4) : 0.01, 0.10, 0.80, 0.07, 0.02;
  (5) : 0.001, 0.010, 0.020, 0.949, 0.020;
  (6) : 0.23, 0.23, 0.22, 0.22, 0.10;
}
probability ( R_APB_REPSTIM_FACILI | R_APB_NMT ) {
  (0) : 0.95, 0.02, 0.01, 0.02;
  (1) : 0.010, 0.889, 0.100, 0.001;
  (2) : 0.010, 0.080, 0.909, 0.001;
  (3) : 0.89, 0.08, 0.01, 0.02;
  (4) : 0.48, 0.50, 0.01, 0.01;
  (5) : 0.020, 0.949, 0.030, 0.001;
  (6) : 0.25, 0.25, 0.25, 0.25;
}
probability ( R_APB_REPSTIM_DECR | R_APB_NMT ) {
  (0) : 0.949, 0.020, 0.010, 0.001, 0.020;
  (1) : 0.04, 0.20, 0.70, 0.04, 0.02;
  (2) : 0.001, 0.010, 0.040, 0.929, 0.020;
  (3) : 0.35, 0.57, 0.05, 0.01, 0.02;
  (4) : 0.02, 0.10, 0.80, 0.06, 0.02;
  (5) : 0.001, 0.010, 0.040, 0.929, 0.020;
  (6) : 0.245, 0.245, 0.245, 0.245, 0.020;
}
probability ( R_APB_REPSTIM_CMAPAMP | R_APB_ALLAMP_WA ) {
  (0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1) : 0.0013001298, 0.1159119826, 0.1280129808, 0.1330129800, 0.1300129805, 0.1194119821, 0.1031099845, 0.0838083874, 0.0639063904, 0.0459045931, 0.0310030953, 0.0197019970, 0.0117011982, 0.0066006590, 0.0035003495, 0.0017001697, 0.0008000799, 0.0004000399, 0.0001000100, 0.0001000100, 0.0000000000;
  (2) : 0.0000000000, 0.0002999699, 0.0012998696, 0.0040995888, 0.0111988968, 0.0260973924, 0.0515947850, 0.0867912748, 0.1246879638, 0.1524849558, 0.1588839539, 0.1410859591, 0.1067889690, 0.0687930801, 0.0377961890, 0.0176981949, 0.0070992879, 0.0023997593, 0.0006999298, 0.0001999799, 0.0000000000;
  (3) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000999899, 0.0006999293, 0.0030996870, 0.0110988892, 0.0312968696, 0.0703929317, 0.1247878790, 0.1747828305, 0.1934808123, 0.1692828358, 0.1169878865, 0.0639935379, 0.0275971732, 0.0093990509, 0.0024997476, 0.0004999495;
  (4) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00030003, 0.00210021, 0.01070110, 0.03860390, 0.09900990, 0.17961800, 0.23162300, 0.21192100, 0.13751400, 0.06320630, 0.02070210, 0.00470047;
  (5) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0003000302, 0.0031003117, 0.0190019106, 0.0723072405, 0.1736170972, 0.2622261469, 0.2497251398, 0.1497150838, 0.0567057318, 0.0133013074;
  (6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0007, 0.0037, 0.0151, 0.0464, 0.1065, 0.1843, 0.2393, 0.2335, 0.1704;
  (7) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00020002, 0.00110011, 0.00610061, 0.02490250, 0.07680770, 0.17791801, 0.30893102, 0.40404002;
  (8) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0005, 0.0038, 0.0208, 0.0860, 0.2659, 0.6229;
}
probability ( R_APB_NMT | R_DE_REGEN_APB_NMT, R_MYAS_APB_NMT ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 0) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 0) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (5, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (6, 0) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 1) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (6, 1) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (2, 2) : 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (3, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (6, 2) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (5, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (6, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (6, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (1, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (4, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (5, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (6, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 6) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 6) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 6) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 6) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 6) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (5, 6) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (6, 6) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_APB_MUPINSTAB | R_APB_NMT ) {
  (0) : 0.95, 0.05;
  (1) : 0.1, 0.9;
  (2) : 0.03, 0.97;
  (3) : 0.2, 0.8;
  (4) : 0.1, 0.9;
  (5) : 0.03, 0.97;
  (6) : 0.1, 0.9;
}
probability ( R_APB_DE_REGEN | R_LNL_DIFFN_APB_DE_REGEN, R_MYOP_MYDY_APB_DE_REGEN ) {
  (0, 0) : 1.0, 0.0;
  (1, 0) : 0.0, 1.0;
  (0, 1) : 0.0, 1.0;
  (1, 1) : 0.0, 1.0;
}
probability ( R_APB_MUPSATEL | R_APB_DE_REGEN ) {
  (0) : 0.95, 0.05;
  (1) : 0.2, 0.8;
}
probability ( R_APB_QUAN_MUPPOLY | R_APB_DE_REGEN, R_APB_EFFMUS ) {
  (0, 0) : 0.109, 0.548, 0.343;
  (1, 0) : 0.004, 0.122, 0.874;
  (0, 1) : 0.340, 0.564, 0.096;
  (1, 1) : 0.015, 0.261, 0.724;
  (0, 2) : 0.925, 0.075, 0.000;
  (1, 2) : 0.091, 0.526, 0.383;
  (0, 3) : 0.796, 0.201, 0.003;
  (1, 3) : 0.061, 0.465, 0.474;
  (0, 4) : 0.637, 0.348, 0.015;
  (1, 4) : 0.039, 0.396, 0.565;
  (0, 5) : 0.340, 0.564, 0.096;
  (1, 5) : 0.015, 0.261, 0.724;
  (0, 6) : 0.340, 0.564, 0.096;
  (1, 6) : 0.015, 0.261, 0.724;
}
probability ( R_APB_QUAL_MUPPOLY | R_APB_QUAN_MUPPOLY ) {
  (0) : 0.95, 0.05;
  (1) : 0.3, 0.7;
  (2) : 0.05, 0.95;
}
probability ( R_APB_QUAL_MUPDUR | R_APB_MUPDUR ) {
  (0) : 0.8309, 0.1677, 0.0014;
  (1) : 0.49, 0.49, 0.02;
  (2) : 0.1065, 0.7870, 0.1065;
  (3) : 0.02, 0.49, 0.49;
  (4) : 0.0014, 0.1677, 0.8309;
  (5) : 0.0001, 0.0392, 0.9607;
  (6) : 0.2597, 0.4806, 0.2597;
}
probability ( R_APB_MUPDUR | R_APB_EFFMUS ) {
  (0) : 0.9388, 0.0412, 0.0000, 0.0000, 0.0000, 0.0000, 0.0200;
  (1) : 0.0396, 0.9008, 0.0396, 0.0000, 0.0000, 0.0000, 0.0200;
  (2) : 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.02;
  (3) : 0.0000, 0.0000, 0.0396, 0.9008, 0.0396, 0.0000, 0.0200;
  (4) : 0.0000, 0.0000, 0.0000, 0.0412, 0.9380, 0.0008, 0.0200;
  (5) : 0.0000, 0.0000, 0.0000, 0.0039, 0.2546, 0.7215, 0.0200;
  (6) : 0.0900, 0.2350, 0.3236, 0.2350, 0.0900, 0.0064, 0.0200;
}
probability ( R_APB_QUAN_MUPDUR | R_APB_MUPDUR ) {
  (0) : 0.099819962, 0.183336930, 0.240247909, 0.224544915, 0.149729943, 0.071214173, 0.024204791, 0.005801158, 0.001000200, 0.000100020, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (1) : 0.0102010105, 0.0369037380, 0.0951095980, 0.1747171800, 0.2289232358, 0.2140212204, 0.1426141469, 0.0678068698, 0.0230023237, 0.0056005658, 0.0010001010, 0.0001000101, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (2) : 0.0000, 0.0002, 0.0025, 0.0177, 0.0739, 0.1852, 0.2785, 0.2515, 0.1363, 0.0444, 0.0087, 0.0010, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3) : 0.000000e+00, 0.000000e+00, 2.999698e-04, 1.999798e-03, 1.019899e-02, 3.679627e-02, 9.489042e-02, 1.742829e-01, 2.283768e-01, 2.134788e-01, 1.422859e-01, 6.769315e-02, 2.299768e-02, 5.599436e-03, 9.998992e-04, 9.998992e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (4) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.999698e-04, 1.999798e-03, 1.019899e-02, 3.679627e-02, 9.489042e-02, 1.742829e-01, 2.283768e-01, 2.134788e-01, 1.422859e-01, 6.769315e-02, 2.299768e-02, 5.599436e-03, 9.998992e-04, 9.998992e-05, 0.000000e+00;
  (5) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 9.999003e-05, 3.999601e-04, 1.799821e-03, 6.999302e-03, 2.189781e-02, 5.409462e-02, 1.051890e-01, 1.612840e-01, 1.949811e-01, 1.859811e-01, 1.398860e-01, 8.289172e-02, 3.879611e-02, 5.699432e-03;
  (6) : 0.0201, 0.0341, 0.0529, 0.0748, 0.0966, 0.1138, 0.1224, 0.1202, 0.1078, 0.0882, 0.0658, 0.0449, 0.0279, 0.0159, 0.0082, 0.0039, 0.0017, 0.0007, 0.0001;
}
probability ( R_APB_QUAL_MUPAMP | R_APB_MUPAMP ) {
  (0) : 0.4289, 0.5209, 0.0499, 0.0003, 0.0000;
  (1) : 0.0647, 0.5494, 0.3679, 0.0180, 0.0000;
  (2) : 0.00000000, 0.04790478, 0.87538756, 0.07670766, 0.00000000;
  (3) : 0.000000000, 0.008699129, 0.283771963, 0.677931912, 0.029596996;
  (4) : 0.0000, 0.0002, 0.0376, 0.6283, 0.3339;
  (5) : 0.0000, 0.0000, 0.0010, 0.0788, 0.9202;
  (6) : 0.0960, 0.1884, 0.2830, 0.3014, 0.1312;
}
probability ( R_APB_MUPAMP | R_APB_EFFMUS ) {
  (0) : 0.782, 0.195, 0.003, 0.000, 0.000, 0.000, 0.020;
  (1) : 0.1043101012, 0.7710777479, 0.1043101012, 0.0003000303, 0.0000000000, 0.0000000000, 0.0200020194;
  (2) : 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.02;
  (3) : 0.00000000, 0.00029997, 0.10109000, 0.74712503, 0.13148700, 0.00000000, 0.01999800;
  (4) : 0.0000, 0.0000, 0.0024, 0.1528, 0.7968, 0.0280, 0.0200;
  (5) : 0.0000, 0.0000, 0.0000, 0.0028, 0.0968, 0.8804, 0.0200;
  (6) : 0.1328, 0.1932, 0.2189, 0.1932, 0.1726, 0.0693, 0.0200;
}
probability ( R_APB_QUAN_MUPAMP | R_APB_MUPAMP ) {
  (0) : 0.0008002392, 0.0037011061, 0.0135040858, 0.0381113600, 0.0835250123, 0.1425428503, 0.1895568010, 0.1963587938, 0.1583478337, 0.0994296956, 0.0486145490, 0.0185055806, 0.0055016442, 0.0013003886, 0.0002000598, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (1) : 0.0000000000, 0.0001000199, 0.0008001595, 0.0037007377, 0.0135026916, 0.0381075764, 0.0835166482, 0.1425289116, 0.1895378825, 0.1963388783, 0.1583319018, 0.0994198384, 0.0486096699, 0.0185036885, 0.0055010966, 0.0013002592, 0.0002000399, 0.0000000000, 0.0000000000, 0.0000000000;
  (2) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0005000502, 0.0037003714, 0.0187019069, 0.0639064236, 0.1475150546, 0.2302230852, 0.2431240900, 0.1737170643, 0.0840084311, 0.0275028102, 0.0061006123, 0.0009000903, 0.0001000100, 0.0000000000, 0.0000000000;
  (3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0008, 0.0037, 0.0135, 0.0381, 0.0835, 0.1426, 0.1896, 0.1963, 0.1583, 0.0995, 0.0487, 0.0185, 0.0055, 0.0013;
  (4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0008, 0.0038, 0.0136, 0.0383, 0.0841, 0.1435, 0.1909, 0.1977, 0.1594, 0.1001, 0.0490, 0.0187;
  (5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0006, 0.0024, 0.0082, 0.0232, 0.0542, 0.1041, 0.1645, 0.2137, 0.2283, 0.2007;
  (6) : 0.0045, 0.0078, 0.0127, 0.0197, 0.0289, 0.0403, 0.0531, 0.0664, 0.0787, 0.0883, 0.0939, 0.0946, 0.0903, 0.0817, 0.0700, 0.0569, 0.0438, 0.0319, 0.0221, 0.0144;
}
probability ( R_APB_TA_CONCL | R_APB_EFFMUS ) {
  (0) : 0.00, 0.00, 0.01, 0.10, 0.88, 0.01;
  (1) : 0.00, 0.00, 0.10, 0.80, 0.09, 0.01;
  (2) : 0.00, 0.03, 0.93, 0.03, 0.00, 0.01;
  (3) : 0.17, 0.50, 0.30, 0.02, 0.00, 0.01;
  (4) : 0.44, 0.50, 0.05, 0.00, 0.00, 0.01;
  (5) : 0.80, 0.18, 0.01, 0.00, 0.00, 0.01;
  (6) : 0.17, 0.17, 0.17, 0.17, 0.16, 0.16;
}
probability ( R_APB_EFFMUS | R_APB_NMT, R_APB_MUSIZE ) {
  (0, 0) : 0.935, 0.045, 0.000, 0.000, 0.000, 0.000, 0.020;
  (1, 0) : 0.97, 0.01, 0.00, 0.00, 0.00, 0.00, 0.02;
  (2, 0) : 0.978, 0.002, 0.000, 0.000, 0.000, 0.000, 0.020;
  (3, 0) : 0.9487, 0.0313, 0.0000, 0.0000, 0.0000, 0.0000, 0.0200;
  (4, 0) : 0.9715, 0.0085, 0.0000, 0.0000, 0.0000, 0.0000, 0.0200;
  (5, 0) : 0.9786, 0.0014, 0.0000, 0.0000, 0.0000, 0.0000, 0.0200;
  (6, 0) : 0.7800777270, 0.1866189347, 0.0108010962, 0.0023002292, 0.0002000199, 0.0000000000, 0.0200019930;
  (0, 1) : 0.0609, 0.8781, 0.0400, 0.0010, 0.0000, 0.0000, 0.0200;
  (1, 1) : 0.7259, 0.2514, 0.0026, 0.0001, 0.0000, 0.0000, 0.0200;
  (2, 1) : 0.9451, 0.0348, 0.0001, 0.0000, 0.0000, 0.0000, 0.0200;
  (3, 1) : 0.1015899756, 0.8506147958, 0.0271972935, 0.0005999399, 0.0000000000, 0.0000000000, 0.0199979952;
  (4, 1) : 4.312568e-01, 5.435457e-01, 5.099488e-03, 9.998995e-05, 0.000000e+00, 0.000000e+00, 1.999799e-02;
  (5, 1) : 0.89051093, 0.08929109, 0.00019998, 0.00000000, 0.00000000, 0.00000000, 0.01999800;
  (6, 1) : 0.5030501962, 0.3808381485, 0.0634063247, 0.0266027104, 0.0054005421, 0.0007000703, 0.0200020078;
  (0, 2) : 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.02;
  (1, 2) : 0.0039, 0.9275, 0.0482, 0.0004, 0.0000, 0.0000, 0.0200;
  (2, 2) : 0.7359, 0.2437, 0.0004, 0.0000, 0.0000, 0.0000, 0.0200;
  (3, 2) : 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.02;
  (4, 2) : 0.0000, 0.3917, 0.5883, 0.0000, 0.0000, 0.0000, 0.0200;
  (5, 2) : 0.0029, 0.9767, 0.0004, 0.0000, 0.0000, 0.0000, 0.0200;
  (6, 2) : 0.164516082, 0.449845225, 0.191019096, 0.129213065, 0.038403819, 0.007000704, 0.020002010;
  (0, 3) : 0.0000000, 0.0000000, 0.0155016, 0.9427942, 0.0217022, 0.0000000, 0.0200020;
  (1, 3) : 9.999003e-05, 2.160781e-01, 6.095392e-01, 1.512850e-01, 2.999701e-03, 0.000000e+00, 1.999801e-02;
  (2, 3) : 0.14028603, 0.79742014, 0.04049601, 0.00179982, 0.00000000, 0.00000000, 0.01999800;
  (3, 3) : 0.0000, 0.0000, 0.0582, 0.9100, 0.0118, 0.0000, 0.0200;
  (4, 3) : 0.0000, 0.0004, 0.6513, 0.3273, 0.0010, 0.0000, 0.0200;
  (5, 3) : 0.0001, 0.5640, 0.4045, 0.0114, 0.0000, 0.0000, 0.0200;
  (6, 3) : 0.04669529, 0.26567295, 0.23467695, 0.26097395, 0.13068697, 0.04129589, 0.01999800;
  (0, 4) : 0.0000, 0.0000, 0.0000, 0.0216, 0.9368, 0.0216, 0.0200;
  (1, 4) : 0.000000000, 0.002999702, 0.267673137, 0.600740306, 0.106489054, 0.002099791, 0.019998010;
  (2, 4) : 0.006799319, 0.564343944, 0.336465966, 0.070192993, 0.002199780, 0.000000000, 0.019997998;
  (3, 4) : 0.00000000, 0.00000000, 0.00000000, 0.07950798, 0.88898882, 0.01150120, 0.02000200;
  (4, 4) : 0.0000, 0.0000, 0.0119, 0.7118, 0.2555, 0.0008, 0.0200;
  (5, 4) : 0.000000000, 0.005099492, 0.567643181, 0.399160128, 0.008099193, 0.000000000, 0.019998006;
  (6, 4) : 0.007899215, 0.098490258, 0.179582106, 0.316668187, 0.251475148, 0.125887074, 0.019998012;
  (0, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0221, 0.9579, 0.0200;
  (1, 5) : 0.00000000, 0.00000000, 0.01640160, 0.33373297, 0.53505394, 0.09480949, 0.02000200;
  (2, 5) : 0.000000000, 0.060906057, 0.426642701, 0.426842701, 0.063606355, 0.002000199, 0.020001986;
  (3, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0805, 0.8995, 0.0200;
  (4, 5) : 0.0000, 0.0000, 0.0000, 0.0166, 0.7089, 0.2545, 0.0200;
  (5, 5) : 0.000000000, 0.000000000, 0.035703627, 0.624362482, 0.313631241, 0.006300635, 0.020002015;
  (6, 5) : 0.0008999091, 0.0250974747, 0.0942905048, 0.2637737336, 0.3321666645, 0.2637737336, 0.0199979798;
  (0, 6) : 0.1220, 0.2515, 0.1645, 0.2065, 0.1490, 0.0865, 0.0200;
  (1, 6) : 0.26392592, 0.32763290, 0.13961396, 0.13591396, 0.07720768, 0.03570359, 0.02000199;
  (2, 6) : 0.44245565, 0.33496673, 0.09249073, 0.06899304, 0.03019698, 0.01089889, 0.01999798;
  (3, 6) : 0.1350, 0.2632, 0.1638, 0.1995, 0.1397, 0.0788, 0.0200;
  (4, 6) : 0.1946, 0.3041, 0.1558, 0.1682, 0.1047, 0.0526, 0.0200;
  (5, 6) : 0.31873197, 0.34323397, 0.12641299, 0.11131099, 0.05680569, 0.02350240, 0.02000200;
  (6, 6) : 0.23812395, 0.28702894, 0.13641397, 0.15231497, 0.10450998, 0.06160619, 0.02000200;
}
probability ( R_APB_MVA_AMP | R_APB_EFFMUS ) {
  (0) : 0.00, 0.04, 0.96;
  (1) : 0.01, 0.15, 0.84;
  (2) : 0.05, 0.90, 0.05;
  (3) : 0.50, 0.49, 0.01;
  (4) : 0.85, 0.15, 0.00;
  (5) : 0.96, 0.04, 0.00;
  (6) : 0.33, 0.34, 0.33;
}
probability ( R_APB_MULOSS | R_MED_BLOCK_WA, R_APB_MALOSS ) {
  (0, 0) : 0.98, 0.00, 0.00, 0.00, 0.00, 0.02;
  (1, 0) : 0.9746, 0.0054, 0.0000, 0.0000, 0.0000, 0.0200;
  (2, 0) : 0.0664, 0.9136, 0.0000, 0.0000, 0.0000, 0.0200;
  (3, 0) : 0.0160, 0.1801, 0.7138, 0.0701, 0.0000, 0.0200;
  (4, 0) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 1) : 0.0167, 0.9613, 0.0020, 0.0000, 0.0000, 0.0200;
  (1, 1) : 0.003400341, 0.952995248, 0.023602406, 0.000000000, 0.000000000, 0.020002005;
  (2, 1) : 0.0002, 0.2725, 0.7073, 0.0000, 0.0000, 0.0200;
  (3, 1) : 0.0009, 0.0263, 0.4192, 0.5336, 0.0000, 0.0200;
  (4, 1) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 2) : 0.0001999799, 0.0534946775, 0.9237076121, 0.0025997389, 0.0000000000, 0.0199979916;
  (1, 2) : 0.00000000, 0.02340229, 0.94509453, 0.01150119, 0.00000000, 0.02000199;
  (2, 2) : 0.0000, 0.0048, 0.7523, 0.2229, 0.0000, 0.0200;
  (3, 2) : 0.000000000, 0.001300131, 0.063706430, 0.914991430, 0.000000000, 0.020002009;
  (4, 2) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 3) : 0.0000000000, 0.0003000301, 0.0481048082, 0.9315931583, 0.0000000000, 0.0200020034;
  (1, 3) : 0.00000000, 0.00010001, 0.02700271, 0.95289527, 0.00000000, 0.02000201;
  (2, 3) : 0.0000, 0.0000, 0.0091, 0.9709, 0.0000, 0.0200;
  (3, 3) : 0.0000, 0.0001, 0.0087, 0.9712, 0.0000, 0.0200;
  (4, 3) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (1, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (2, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (3, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (4, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 5) : 0.1427, 0.2958, 0.4254, 0.1161, 0.0000, 0.0200;
  (1, 5) : 0.1157, 0.2677, 0.4444, 0.1522, 0.0000, 0.0200;
  (2, 5) : 0.06939309, 0.20107998, 0.45265495, 0.25687397, 0.00000000, 0.01999800;
  (3, 5) : 0.0173, 0.0696, 0.2854, 0.6077, 0.0000, 0.0200;
  (4, 5) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
}
probability ( R_APB_MVA_RECRUIT | R_APB_MULOSS, R_APB_VOL_ACT ) {
  (0, 0) : 0.9295, 0.0705, 0.0000, 0.0000;
  (1, 0) : 0.4821, 0.5165, 0.0014, 0.0000;
  (2, 0) : 0.0661, 0.7993, 0.1346, 0.0000;
  (3, 0) : 0.00149985, 0.13658602, 0.86191413, 0.00000000;
  (4, 0) : 0.0, 0.0, 0.0, 1.0;
  (5, 0) : 0.2639737, 0.4343566, 0.3016697, 0.0000000;
  (0, 1) : 0.1707, 0.7000, 0.1293, 0.0000;
  (1, 1) : 0.0366, 0.5168, 0.4466, 0.0000;
  (2, 1) : 0.0043, 0.1788, 0.8169, 0.0000;
  (3, 1) : 0.0002999699, 0.0347964836, 0.9649035465, 0.0000000000;
  (4, 1) : 0.0, 0.0, 0.0, 1.0;
  (5, 1) : 0.1146, 0.3465, 0.5389, 0.0000;
  (0, 2) : 0.0038, 0.1740, 0.8222, 0.0000;
  (1, 2) : 0.0005, 0.0594, 0.9401, 0.0000;
  (2, 2) : 0.0001, 0.0205, 0.9794, 0.0000;
  (3, 2) : 0.0000, 0.0061, 0.9939, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 1.0;
  (5, 2) : 0.0360, 0.2144, 0.7496, 0.0000;
  (0, 3) : 0.0, 0.0, 0.0, 1.0;
  (1, 3) : 0.0, 0.0, 0.0, 1.0;
  (2, 3) : 0.0, 0.0, 0.0, 1.0;
  (3, 3) : 0.0, 0.0, 0.0, 1.0;
  (4, 3) : 0.0, 0.0, 0.0, 1.0;
  (5, 3) : 0.0, 0.0, 0.0, 1.0;
}
probability ( R_APB_MALOSS | R_LNLT1_LP_BE_APB_MALOSS, R_DIFFN_LNLW_APB_MALOSS ) {
  (0, 0) : 0.98, 0.00, 0.00, 0.00, 0.00, 0.02;
  (1, 0) : 2.199781e-03, 9.777022e-01, 9.999002e-05, 0.000000e+00, 0.000000e+00, 1.999800e-02;
  (2, 0) : 0.0002, 0.0471, 0.9297, 0.0030, 0.0000, 0.0200;
  (3, 0) : 0.0000, 0.0003, 0.0424, 0.9373, 0.0000, 0.0200;
  (4, 0) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 1) : 2.199781e-03, 9.777022e-01, 9.999002e-05, 0.000000e+00, 0.000000e+00, 1.999800e-02;
  (1, 1) : 0.0000, 0.0361, 0.9439, 0.0000, 0.0000, 0.0200;
  (2, 1) : 0.0000, 0.0014, 0.3987, 0.5799, 0.0000, 0.0200;
  (3, 1) : 0.000, 0.000, 0.005, 0.975, 0.000, 0.020;
  (4, 1) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 2) : 0.0002, 0.0471, 0.9297, 0.0030, 0.0000, 0.0200;
  (1, 2) : 0.0000, 0.0014, 0.3987, 0.5799, 0.0000, 0.0200;
  (2, 2) : 0.000, 0.000, 0.013, 0.967, 0.000, 0.020;
  (3, 2) : 0.0000, 0.0000, 0.0014, 0.9786, 0.0000, 0.0200;
  (4, 2) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 3) : 0.0000, 0.0003, 0.0424, 0.9373, 0.0000, 0.0200;
  (1, 3) : 0.000, 0.000, 0.005, 0.975, 0.000, 0.020;
  (2, 3) : 0.0000, 0.0000, 0.0014, 0.9786, 0.0000, 0.0200;
  (3, 3) : 0.0000, 0.0000, 0.0005, 0.9795, 0.0000, 0.0200;
  (4, 3) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (0, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (1, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (2, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (3, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
  (4, 4) : 0.00, 0.00, 0.00, 0.00, 0.98, 0.02;
}
probability ( R_APB_MUSIZE | R_MYOP_MYDY_APB_MUSIZE, R_LNL_DIFFN_APB_MUSIZE ) {
  (0, 0) : 0.9791, 0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0200;
  (1, 0) : 0.9637, 0.0163, 0.0000, 0.0000, 0.0000, 0.0000, 0.0200;
  (2, 0) : 0.92219219, 0.05780581, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000200;
  (3, 0) : 0.3979397533, 0.5663566488, 0.0155015904, 0.0002000199, 0.0000000000, 0.0000000000, 0.0200019876;
  (4, 0) : 0.0435, 0.7319, 0.1930, 0.0114, 0.0002, 0.0000, 0.0200;
  (5, 0) : 0.00120012, 0.23172294, 0.58825886, 0.14741496, 0.01120110, 0.00020002, 0.02000200;
  (0, 1) : 0.9637, 0.0163, 0.0000, 0.0000, 0.0000, 0.0000, 0.0200;
  (1, 1) : 0.7493, 0.2257, 0.0049, 0.0001, 0.0000, 0.0000, 0.0200;
  (2, 1) : 0.0537, 0.8568, 0.0684, 0.0011, 0.0000, 0.0000, 0.0200;
  (3, 1) : 3.899611e-03, 3.810621e-01, 5.065491e-01, 8.409162e-02, 4.299571e-03, 9.999002e-05, 1.999800e-02;
  (4, 1) : 0.0000, 0.0395, 0.5059, 0.3550, 0.0758, 0.0038, 0.0200;
  (5, 1) : 0.00000000, 0.00110011, 0.13571401, 0.40254004, 0.36313603, 0.07750781, 0.02000200;
  (0, 2) : 0.92219219, 0.05780581, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000200;
  (1, 2) : 0.0537, 0.8568, 0.0684, 0.0011, 0.0000, 0.0000, 0.0200;
  (2, 2) : 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.02;
  (3, 2) : 0.00000000, 0.00000000, 0.09080908, 0.81858183, 0.07060709, 0.00000000, 0.02000200;
  (4, 2) : 0.0000, 0.0000, 0.0001, 0.0721, 0.8357, 0.0721, 0.0200;
  (5, 2) : 0.0000, 0.0000, 0.0000, 0.0001, 0.0778, 0.9021, 0.0200;
  (0, 3) : 0.3979397533, 0.5663566488, 0.0155015904, 0.0002000199, 0.0000000000, 0.0000000000, 0.0200019876;
  (1, 3) : 3.899611e-03, 3.810621e-01, 5.065491e-01, 8.409162e-02, 4.299571e-03, 9.999002e-05, 1.999800e-02;
  (2, 3) : 0.00000000, 0.00000000, 0.09080908, 0.81858183, 0.07060709, 0.00000000, 0.02000200;
  (3, 3) : 0.000000000, 0.000000000, 0.003599645, 0.165483225, 0.645435878, 0.165483225, 0.019998027;
  (4, 3) : 0.0000, 0.0000, 0.0000, 0.0034, 0.1993, 0.7773, 0.0200;
  (5, 3) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01620161, 0.96379638, 0.02000201;
  (0, 4) : 0.0435, 0.7319, 0.1930, 0.0114, 0.0002, 0.0000, 0.0200;
  (1, 4) : 0.0000, 0.0395, 0.5059, 0.3550, 0.0758, 0.0038, 0.0200;
  (2, 4) : 0.0000, 0.0000, 0.0001, 0.0721, 0.8357, 0.0721, 0.0200;
  (3, 4) : 0.0000, 0.0000, 0.0000, 0.0034, 0.1993, 0.7773, 0.0200;
  (4, 4) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01620161, 0.96379638, 0.02000201;
  (5, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0011, 0.9789, 0.0200;
  (0, 5) : 0.00120012, 0.23172294, 0.58825886, 0.14741496, 0.01120110, 0.00020002, 0.02000200;
  (1, 5) : 0.00000000, 0.00110011, 0.13571401, 0.40254004, 0.36313603, 0.07750781, 0.02000200;
  (2, 5) : 0.0000, 0.0000, 0.0000, 0.0001, 0.0778, 0.9021, 0.0200;
  (3, 5) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01620161, 0.96379638, 0.02000201;
  (4, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0011, 0.9789, 0.0200;
  (5, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9799, 0.0200;
}
probability ( R_APB_MUSCLE_VOL | R_APB_MUSIZE, R_APB_MALOSS ) {
  (0, 0) : 0.9896, 0.0104;
  (1, 0) : 0.8137, 0.1863;
  (2, 0) : 0.0209, 0.9791;
  (3, 0) : 0.009, 0.991;
  (4, 0) : 0.003, 0.997;
  (5, 0) : 0.0004, 0.9996;
  (6, 0) : 0.4212, 0.5788;
  (0, 1) : 0.9976, 0.0024;
  (1, 1) : 0.9603, 0.0397;
  (2, 1) : 0.5185, 0.4815;
  (3, 1) : 0.1087, 0.8913;
  (4, 1) : 0.0278, 0.9722;
  (5, 1) : 0.0046, 0.9954;
  (6, 1) : 0.5185, 0.4815;
  (0, 2) : 0.999, 0.001;
  (1, 2) : 0.9893, 0.0107;
  (2, 2) : 0.9588, 0.0412;
  (3, 2) : 0.6377, 0.3623;
  (4, 2) : 0.2716, 0.7284;
  (5, 2) : 0.0779, 0.9221;
  (6, 2) : 0.6336, 0.3664;
  (0, 3) : 0.9995, 0.0005;
  (1, 3) : 0.9969, 0.0031;
  (2, 3) : 0.9953, 0.0047;
  (3, 3) : 0.9518, 0.0482;
  (4, 3) : 0.8234, 0.1766;
  (5, 3) : 0.5986, 0.4014;
  (6, 3) : 0.7685, 0.2315;
  (0, 4) : 0.9989, 0.0011;
  (1, 4) : 0.9984, 0.0016;
  (2, 4) : 0.9984, 0.0016;
  (3, 4) : 0.9975, 0.0025;
  (4, 4) : 0.9965, 0.0035;
  (5, 4) : 0.9956, 0.0044;
  (6, 4) : 0.9857, 0.0143;
  (0, 5) : 0.9363, 0.0637;
  (1, 5) : 0.8403, 0.1597;
  (2, 5) : 0.6534, 0.3466;
  (3, 5) : 0.4689, 0.5311;
  (4, 5) : 0.3174, 0.6826;
  (5, 5) : 0.1948, 0.8052;
  (6, 5) : 0.5681, 0.4319;
}
probability ( R_APB_VOL_ACT ) {
   1.0, 0.0, 0.0, 0.0;
}
probability ( R_APB_FORCE | R_APB_VOL_ACT, R_APB_ALLAMP_WA ) {
  (0, 0) : 0.000000000, 0.000000000, 0.000000000, 0.004099592, 0.190781078, 0.805119330;
  (1, 0) : 0.0000, 0.0000, 0.0000, 0.0010, 0.0578, 0.9412;
  (2, 0) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 9.999003e-05, 1.259870e-02, 9.873013e-01;
  (3, 0) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.005900592, 0.994099408;
  (0, 1) : 0.001599841, 0.018598109, 0.093590643, 0.267873123, 0.365863168, 0.252475116;
  (1, 1) : 0.0002, 0.0034, 0.0260, 0.1312, 0.3333, 0.5059;
  (2, 1) : 0.0000, 0.0003, 0.0044, 0.0446, 0.2226, 0.7281;
  (3, 1) : 0.0000, 0.0000, 0.0005, 0.0098, 0.1058, 0.8839;
  (0, 2) : 0.01490150, 0.23542400, 0.53315299, 0.20152000, 0.01490150, 0.00010001;
  (1, 2) : 0.0009, 0.0256, 0.1800, 0.4308, 0.3036, 0.0591;
  (2, 2) : 0.0000, 0.0005, 0.0128, 0.1328, 0.4061, 0.4478;
  (3, 2) : 0.0000, 0.0000, 0.0003, 0.0091, 0.1181, 0.8725;
  (0, 3) : 0.1538, 0.6493, 0.1936, 0.0033, 0.0000, 0.0000;
  (1, 3) : 0.0098, 0.1589, 0.4714, 0.3101, 0.0485, 0.0013;
  (2, 3) : 0.0001, 0.0049, 0.0751, 0.3632, 0.4290, 0.1277;
  (3, 3) : 0.0000, 0.0000, 0.0008, 0.0215, 0.1873, 0.7904;
  (0, 4) : 0.6667, 0.3291, 0.0042, 0.0000, 0.0000, 0.0000;
  (1, 4) : 0.053705427, 0.420442210, 0.454845227, 0.069006935, 0.002000201, 0.000000000;
  (2, 4) : 0.0005000503, 0.0273027177, 0.2433241582, 0.5013503259, 0.2118211377, 0.0157016102;
  (3, 4) : 0.00000000, 0.00009999, 0.00249975, 0.04719530, 0.27557199, 0.67463297;
  (0, 5) : 0.94689472, 0.05310528, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (1, 5) : 0.1173119284, 0.5658566549, 0.3010298164, 0.0157015904, 0.0001000099, 0.0000000000;
  (2, 5) : 0.00120012, 0.06020601, 0.38623904, 0.45424505, 0.09540951, 0.00270027;
  (3, 5) : 0.0000, 0.0001, 0.0044, 0.0713, 0.3326, 0.5916;
  (0, 6) : 0.9782, 0.0218, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 6) : 0.410159144, 0.479352168, 0.106989037, 0.003499651, 0.000000000, 0.000000000;
  (2, 6) : 0.0256025936, 0.2470249382, 0.4838478790, 0.2174219456, 0.0256025936, 0.0005000499;
  (3, 6) : 0.0000000, 0.0020002, 0.0279028, 0.1812180, 0.4112410, 0.3776380;
  (0, 7) : 0.9971, 0.0029, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 7) : 0.7017, 0.2755, 0.0226, 0.0002, 0.0000, 0.0000;
  (2, 7) : 0.1171, 0.4443, 0.3699, 0.0654, 0.0033, 0.0000;
  (3, 7) : 0.0004, 0.0107, 0.0847, 0.3035, 0.3988, 0.2019;
  (0, 8) : 0.9996, 0.0004, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 8) : 0.8804, 0.1161, 0.0035, 0.0000, 0.0000, 0.0000;
  (2, 8) : 0.3270670752, 0.4879511122, 0.1726830397, 0.0119988028, 0.0002999701, 0.0000000000;
  (3, 8) : 0.003000304, 0.042604351, 0.194819234, 0.384938462, 0.292929352, 0.081708298;
}
probability ( R_MED_ALLDEL_WA | R_MED_RDLDDEL, R_MED_DCV_WA ) {
  (0, 0) : 0.9996, 0.0004, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 0) : 0.05119491, 0.22907702, 0.56624305, 0.15348502, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (2, 0) : 0.0001, 0.0035, 0.0632, 0.9326, 0.0006, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 0) : 0.001299870, 0.001999799, 0.004399559, 0.018698094, 0.121287960, 0.747924752, 0.104389966, 0.000000000, 0.000000000;
  (4, 0) : 0.0001000100, 0.0001000100, 0.0003000300, 0.0009000901, 0.0045004505, 0.0434043048, 0.5767580635, 0.3739370411, 0.0000000000;
  (0, 1) : 0.5607, 0.4393, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0192981, 0.1286870, 0.4928510, 0.3591640, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000;
  (2, 1) : 0.0000, 0.0011, 0.0283, 0.9651, 0.0055, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 1) : 0.001000100, 0.001600160, 0.003700369, 0.016101596, 0.109310975, 0.742173829, 0.126112971, 0.000000000, 0.000000000;
  (4, 1) : 0.0001, 0.0001, 0.0002, 0.0008, 0.0041, 0.0399, 0.5568, 0.3980, 0.0000;
  (0, 2) : 0.0069, 0.7963, 0.1968, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 2) : 0.0027, 0.0328, 0.2398, 0.7246, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0002, 0.0075, 0.8889, 0.1034, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 2) : 0.0007999199, 0.0011998799, 0.0027997197, 0.0124987985, 0.0915907890, 0.7235279131, 0.1675829799, 0.0000000000, 0.0000000000;
  (4, 2) : 0.0001, 0.0001, 0.0002, 0.0007, 0.0034, 0.0348, 0.5239, 0.4368, 0.0000;
  (0, 3) : 0.0000, 0.0184, 0.9806, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 3) : 0.000100010, 0.003300329, 0.054705489, 0.938193803, 0.003700369, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (2, 3) : 0.00000000, 0.00000000, 0.00030003, 0.18501899, 0.81468098, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (3, 3) : 0.0004999501, 0.0007999201, 0.0017998203, 0.0086991316, 0.0702930127, 0.6802321224, 0.2376760428, 0.0000000000, 0.0000000000;
  (4, 3) : 0.0001, 0.0001, 0.0001, 0.0005, 0.0027, 0.0287, 0.4783, 0.4895, 0.0000;
  (0, 4) : 0.0000, 0.0001, 0.0179, 0.9820, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 4) : 0.0000000000, 0.0001999799, 0.0047995176, 0.5039497480, 0.4910507545, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (2, 4) : 0.0000, 0.0000, 0.0000, 0.0032, 0.9968, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 4) : 0.0002, 0.0004, 0.0009, 0.0047, 0.0440, 0.5737, 0.3761, 0.0000, 0.0000;
  (4, 4) : 0.0000000000, 0.0000000000, 0.0001000100, 0.0003000299, 0.0018001794, 0.0210020933, 0.4072408697, 0.5695568177, 0.0000000000;
  (0, 5) : 0.0000, 0.0002, 0.0030, 0.1393, 0.8575, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 5) : 0.0000, 0.0000, 0.0003, 0.0188, 0.9804, 0.0005, 0.0000, 0.0000, 0.0000;
  (2, 5) : 0.00000000, 0.00000000, 0.00000000, 0.00150015, 0.93139314, 0.06710671, 0.00000000, 0.00000000, 0.00000000;
  (3, 5) : 0.0001, 0.0001, 0.0002, 0.0013, 0.0150, 0.3152, 0.6681, 0.0000, 0.0000;
  (4, 5) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0001000200, 0.0008001597, 0.0111021958, 0.2804558934, 0.7075417311, 0.0000000000;
  (0, 6) : 0.0000, 0.0000, 0.0000, 0.0006, 0.8145, 0.1849, 0.0000, 0.0000, 0.0000;
  (1, 6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0494, 0.9506, 0.0000, 0.0000, 0.0000;
  (2, 6) : 0.000, 0.000, 0.000, 0.000, 0.001, 0.999, 0.000, 0.000, 0.000;
  (3, 6) : 0.0000, 0.0000, 0.0000, 0.0001, 0.0017, 0.0786, 0.9196, 0.0000, 0.0000;
  (4, 6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0035, 0.1380, 0.8583, 0.0000;
  (0, 7) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0004000399, 0.0113010984, 0.5973599164, 0.3909389453, 0.0000000000, 0.0000000000;
  (1, 7) : 0.000000000, 0.000000000, 0.000000000, 0.000100010, 0.003400339, 0.299029895, 0.697469756, 0.000000000, 0.000000000;
  (2, 7) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0007, 0.1112, 0.8881, 0.0000, 0.0000;
  (3, 7) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00010001, 0.00940094, 0.95939595, 0.03110310, 0.00000000;
  (4, 7) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0003000301, 0.0253025119, 0.9743974580, 0.0000000000;
  (0, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MED_LAT_WA | R_MED_ALLDEL_WA ) {
  (0) : 0.0059005950, 0.1326131127, 0.5032504278, 0.3226322742, 0.0350035298, 0.0006000605, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (1) : 0.0006999302, 0.0195980059, 0.1689830507, 0.4254571276, 0.3127690938, 0.0671933202, 0.0052994716, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (2) : 0.0000, 0.0007, 0.0194, 0.1669, 0.4202, 0.3089, 0.0829, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3) : 0.000000000, 0.000000000, 0.001000101, 0.010901110, 0.063506460, 0.194719185, 0.393439374, 0.292129278, 0.042804341, 0.001500151, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (4) : 0.0001, 0.0003, 0.0011, 0.0034, 0.0090, 0.0210, 0.0528, 0.1370, 0.2128, 0.2375, 0.1826, 0.1229, 0.0179, 0.0016, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (5) : 9.998997e-05, 1.999799e-04, 3.999598e-04, 7.999198e-04, 1.399859e-03, 2.499749e-03, 5.099488e-03, 1.199880e-02, 2.149789e-02, 3.539649e-02, 8.919108e-02, 1.397859e-01, 1.828819e-01, 2.811719e-01, 1.889809e-01, 3.589639e-02, 2.599739e-03, 9.998997e-05, 0.000000e+00;
  (6) : 0.0002, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.0012, 0.0022, 0.0032, 0.0047, 0.0109, 0.0176, 0.0280, 0.0629, 0.1563, 0.2269, 0.2569, 0.2269, 0.0000;
  (7) : 0.0007999196, 0.0008999095, 0.0010998894, 0.0011998793, 0.0013998592, 0.0016998291, 0.0024997486, 0.0036996280, 0.0045995375, 0.0055994369, 0.0105988942, 0.0155983914, 0.0213978882, 0.0432956762, 0.1006899446, 0.1648839093, 0.2535748605, 0.3664627984, 0.0000000000;
  (8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_APB_ALLAMP_WA | R_APB_EFFMUS, R_APB_MULOSS ) {
  (0, 0) : 0.0026002590, 0.3687368562, 0.6075607630, 0.0208020919, 0.0003000299, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (1, 0) : 0.0000, 0.0002, 0.4149, 0.4809, 0.0802, 0.0218, 0.0020, 0.0000, 0.0000;
  (2, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0215, 0.9785, 0.0000, 0.0000, 0.0000;
  (3, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0018, 0.0536, 0.8696, 0.0750, 0.0000;
  (4, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0736, 0.8528, 0.0736;
  (5, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0794, 0.9205;
  (6, 0) : 0.0003, 0.0060, 0.1050, 0.2050, 0.1633, 0.1410, 0.1771, 0.1279, 0.0744;
  (0, 1) : 0.0409, 0.8924, 0.0661, 0.0006, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0100, 0.7700, 0.2049, 0.0128, 0.0022, 0.0001, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0000, 0.0000, 0.2489, 0.7398, 0.0113, 0.0000, 0.0000, 0.0000;
  (3, 1) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0042004223, 0.3468351873, 0.5348532888, 0.1137110614, 0.0004000402, 0.0000000000;
  (4, 1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0064, 0.0855, 0.7880, 0.1197, 0.0004;
  (5, 1) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0002999703, 0.1164881200, 0.7667237897, 0.1164881200;
  (6, 1) : 0.001900191, 0.023002314, 0.190019116, 0.262326160, 0.162916099, 0.124412076, 0.126413077, 0.074007445, 0.035003521;
  (0, 2) : 0.2926, 0.7043, 0.0031, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 2) : 0.0091, 0.4203, 0.5312, 0.0380, 0.0012, 0.0002, 0.0000, 0.0000, 0.0000;
  (2, 2) : 0.00000000, 0.00000000, 0.30946899, 0.68073199, 0.00949905, 0.00029997, 0.00000000, 0.00000000, 0.00000000;
  (3, 2) : 0.0000, 0.0000, 0.0180, 0.6298, 0.2744, 0.0746, 0.0032, 0.0000, 0.0000;
  (4, 2) : 0.000000000, 0.000000000, 0.000100010, 0.104609965, 0.428142859, 0.356835882, 0.107110965, 0.003200319, 0.000000000;
  (5, 2) : 0.000000000, 0.000000000, 0.000000000, 0.002500252, 0.097809860, 0.249825152, 0.532353324, 0.114111070, 0.003400342;
  (6, 2) : 0.01440141, 0.09930999, 0.30283027, 0.27022724, 0.12431211, 0.08210827, 0.06520656, 0.03020303, 0.01140111;
  (0, 3) : 0.7810, 0.2189, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 3) : 0.2669268051, 0.7161714772, 0.0166016879, 0.0003000298, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (2, 3) : 9.998998e-05, 1.027900e-01, 8.792118e-01, 1.779820e-02, 9.998998e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (3, 3) : 0.0000000000, 0.0044004389, 0.8111807890, 0.1762179542, 0.0073007281, 0.0009000898, 0.0000000000, 0.0000000000, 0.0000000000;
  (4, 3) : 0.00000000, 0.00009999, 0.41295901, 0.49655001, 0.07189280, 0.01729830, 0.00119988, 0.00000000, 0.00000000;
  (5, 3) : 0.00000000, 0.00000000, 0.07810778, 0.51965188, 0.26102595, 0.11671198, 0.02340230, 0.00110011, 0.00000000;
  (6, 3) : 0.1169, 0.3697, 0.2867, 0.1411, 0.0431, 0.0234, 0.0133, 0.0045, 0.0013;
  (0, 4) : 0.9907, 0.0093, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 4) : 0.9858, 0.0142, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 4) : 0.9865, 0.0135, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 4) : 0.982, 0.018, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000;
  (4, 4) : 0.9779, 0.0221, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (5, 4) : 0.973, 0.027, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000;
  (6, 4) : 9.370063e-01, 6.289372e-02, 9.999003e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (0, 5) : 0.3595638418, 0.5147487736, 0.0940905586, 0.0239975894, 0.0045995380, 0.0019997991, 0.0007999196, 0.0001999799, 0.0000000000;
  (1, 5) : 0.13358701, 0.38546103, 0.26977302, 0.13078701, 0.04009600, 0.02189780, 0.01269870, 0.00439956, 0.00129987;
  (2, 5) : 0.0096, 0.0788, 0.2992, 0.2816, 0.1319, 0.0873, 0.0689, 0.0313, 0.0114;
  (3, 5) : 0.002600519, 0.028905788, 0.204040914, 0.265752888, 0.159431933, 0.119923950, 0.119023950, 0.068413671, 0.031906387;
  (4, 5) : 0.0004999499, 0.0083991582, 0.1181879752, 0.2138789551, 0.1629839658, 0.1381859710, 0.1690829645, 0.1198879748, 0.0688930855;
  (5, 5) : 0.0001, 0.0021, 0.0586, 0.1473, 0.1427, 0.1363, 0.2057, 0.1798, 0.1274;
  (6, 5) : 0.05210518, 0.16081594, 0.22722291, 0.20131992, 0.10661096, 0.07920787, 0.08310827, 0.05590558, 0.03370339;
}
probability ( R_MED_AMP_WA | R_APB_ALLAMP_WA ) {
  (0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1) : 0.0003000301, 0.2850290713, 0.2346230587, 0.1781180445, 0.1245120311, 0.0803080201, 0.0476048119, 0.0261026065, 0.0132013033, 0.0061006115, 0.0026002607, 0.0010001003, 0.0004000401, 0.0001000100, 0.0000000000, 0.0000000000, 0.0000000000;
  (2) : 0.000000000, 0.013501401, 0.036903703, 0.079407907, 0.135314012, 0.181818016, 0.193219017, 0.162216015, 0.107711010, 0.056405605, 0.023402302, 0.007600761, 0.002000200, 0.000400040, 0.000100010, 0.000000000, 0.000000000;
  (3) : 0.000000e+00, 0.000000e+00, 9.998988e-05, 5.999393e-04, 3.599635e-03, 1.649838e-02, 5.349464e-02, 1.232878e-01, 2.012797e-01, 2.334767e-01, 1.921807e-01, 1.120888e-01, 4.649535e-02, 1.369858e-02, 2.799716e-03, 3.999596e-04, 0.000000e+00;
  (4) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0001000099, 0.0010000985, 0.0069006799, 0.0309030546, 0.0926091639, 0.1855187273, 0.2496246331, 0.2250226692, 0.1358138004, 0.0549054193, 0.0149014781, 0.0027002660;
  (5) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0003000302, 0.0031003117, 0.0190019106, 0.0723072405, 0.1736170972, 0.2622261469, 0.2497251398, 0.1497150838, 0.0567057318, 0.0133013074;
  (6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0017, 0.0093, 0.0355, 0.0964, 0.1860, 0.2545, 0.2471, 0.1693;
  (7) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0004, 0.0029, 0.0155, 0.0599, 0.1641, 0.3182, 0.4390;
  (8) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000100010, 0.001500151, 0.011501204, 0.063106321, 0.244524083, 0.679268231;
}
probability ( R_MED_ALLCV_EW | R_MED_DCV_EW, R_MED_RDLDCV_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0699, 0.8102, 0.1199, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.00000000, 0.04790481, 0.95209519, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (3, 0) : 0.0008999102, 0.0089991017, 0.0887911169, 0.8186181555, 0.0826917157, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (4, 0) : 0.0000000000, 0.0000000000, 0.0005000499, 0.1011099848, 0.8477848729, 0.0506050924, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (5, 0) : 0.00000000, 0.00000000, 0.00000000, 0.00050005, 0.07910790, 0.90019005, 0.02020200, 0.00000000, 0.00000000, 0.00000000;
  (6, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0006, 0.0734, 0.8393, 0.0867, 0.0000, 0.0000;
  (7, 0) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0003000301, 0.0947095445, 0.8945894205, 0.0104010049, 0.0000000000;
  (8, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0020, 0.0932, 0.9048, 0.0000;
  (9, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999;
  (0, 1) : 0.006600656, 0.145514907, 0.838083463, 0.009800974, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (1, 1) : 0.0005, 0.0239, 0.4369, 0.5387, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0003, 0.0239, 0.9748, 0.0010, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 1) : 0.0000, 0.0002, 0.0036, 0.2467, 0.7339, 0.0156, 0.0000, 0.0000, 0.0000, 0.0000;
  (4, 1) : 0.0000, 0.0000, 0.0000, 0.0050, 0.3134, 0.6811, 0.0005, 0.0000, 0.0000, 0.0000;
  (5, 1) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0071007119, 0.5493551483, 0.4433441197, 0.0002000201, 0.0000000000, 0.0000000000;
  (6, 1) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.009100911, 0.528353047, 0.462546042, 0.000000000, 0.000000000;
  (7, 1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0216, 0.8359, 0.1425, 0.0000;
  (8, 1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0004, 0.0355, 0.9641, 0.0000;
  (9, 1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999;
  (0, 2) : 0.0000, 0.0000, 0.0047, 0.9951, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 2) : 0.0000, 0.0000, 0.0004, 0.9005, 0.0991, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0000, 0.0000, 0.1288, 0.8712, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 2) : 0.0000000, 0.0000000, 0.0000000, 0.0113011, 0.5711569, 0.4175420, 0.0000000, 0.0000000, 0.0000000, 0.0000000;
  (4, 2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0214, 0.9354, 0.0432, 0.0000, 0.0000, 0.0000;
  (5, 2) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 9.998997e-05, 5.649438e-02, 9.323067e-01, 1.109890e-02, 0.000000e+00, 0.000000e+00;
  (6, 2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0004, 0.1430, 0.8551, 0.0015, 0.0000;
  (7, 2) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.001999798, 0.354864716, 0.643135486, 0.000000000;
  (8, 2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0122, 0.9877, 0.0000;
  (9, 2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999;
  (0, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 3) : 0.000, 0.000, 0.000, 0.000, 0.000, 0.997, 0.003, 0.000, 0.000, 0.000;
  (2, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2336, 0.7664, 0.0000, 0.0000, 0.0000;
  (3, 3) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.002599739, 0.993400662, 0.003999599, 0.000000000, 0.000000000;
  (4, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2469, 0.7531, 0.0000, 0.0000;
  (5, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0020, 0.9939, 0.0041, 0.0000;
  (6, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0314, 0.9686, 0.0000;
  (7, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.9998, 0.0000;
  (8, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.9997, 0.0000;
  (9, 3) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0012, 0.1305, 0.8445, 0.0238, 0.0000;
  (1, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0004, 0.0740, 0.8589, 0.0667, 0.0000;
  (2, 4) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00010001, 0.03730371, 0.80288023, 0.15971605, 0.00000000;
  (3, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0070, 0.3517, 0.6413, 0.0000;
  (4, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.0581, 0.9416, 0.0000;
  (5, 4) : 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.006, 0.994, 0.000;
  (6, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0008, 0.9992, 0.0000;
  (7, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999, 0.0000;
  (8, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.9998, 0.0000;
  (9, 4) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0321, 0.9677, 0.0000;
  (1, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0184, 0.9815, 0.0000;
  (2, 5) : 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.010101, 0.989899, 0.000000;
  (3, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0042, 0.9958, 0.0000;
  (4, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0011, 0.9989, 0.0000;
  (5, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.9997, 0.0000;
  (6, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999, 0.0000;
  (7, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  (8, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.9999, 0.0000;
  (9, 5) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MED_CV_EW | R_MED_ALLCV_EW ) {
  (0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0006, 0.0168, 0.1184, 0.2960, 0.3227, 0.1783, 0.0560, 0.0112;
  (1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0005, 0.0155, 0.1165, 0.2969, 0.3229, 0.1782, 0.0564, 0.0114, 0.0016;
  (2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0006, 0.0039, 0.0589, 0.2434, 0.3586, 0.2350, 0.0808, 0.0164, 0.0022, 0.0002;
  (3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0007, 0.0176, 0.1966, 0.2515, 0.2699, 0.1688, 0.0690, 0.0203, 0.0046, 0.0009, 0.0001, 0.0000;
  (4) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00560056, 0.09000899, 0.30563098, 0.30933098, 0.20391999, 0.06730670, 0.01520150, 0.00260026, 0.00040004, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0006, 0.0496, 0.2972, 0.4059, 0.2097, 0.0258, 0.0098, 0.0013, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (6) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.789820e-02, 2.945709e-01, 4.469549e-01, 1.901810e-01, 4.339569e-02, 6.599339e-03, 2.999699e-04, 9.998998e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (7) : 0.0000, 0.0000, 0.0265, 0.5431, 0.3624, 0.0622, 0.0054, 0.0004, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (8) : 0.0000000000, 0.1265129064, 0.7654764336, 0.1006099255, 0.0070006948, 0.0004000397, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (9) : 0.9999, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
}
probability ( R_MED_BLOCK_EW | R_DIFFN_MED_BLOCK, R_LNLBE_MED_BLOCK ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (3, 0) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0001, 0.4980, 0.5019, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (3, 1) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (1, 2) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0007, 0.4328, 0.5665, 0.0000;
  (3, 2) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (1, 3) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (2, 3) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (3, 3) : 0.0003000299, 0.0011001095, 0.0125012945, 0.9860985661, 0.0000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MED_AMPR_EW | R_MED_BLOCK_EW ) {
  (0) : 0.0879, 0.4192, 0.4232, 0.0693, 0.0004, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1) : 0.00189981, 0.03439660, 0.25667401, 0.52914701, 0.17348301, 0.00439956, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (2) : 0.0001, 0.0010, 0.0076, 0.0403, 0.1720, 0.3730, 0.3420, 0.0633, 0.0007, 0.0000, 0.0000, 0.0000;
  (3) : 0.0009000901, 0.0015001501, 0.0026002602, 0.0049004903, 0.0095009506, 0.0176018011, 0.0347035021, 0.0768077046, 0.1668170100, 0.3414340204, 0.3432340205, 0.0000000000;
  (4) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_ALLCV_WD | R_MEDD2_LSLOW_WD, R_MEDD2_DSLOW_WD ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 2.359760e-02, 2.578741e-01, 6.403361e-01, 7.809222e-02, 9.999002e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (2, 0) : 0.0000, 0.0021, 0.1149, 0.7277, 0.1553, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 0) : 0.0003, 0.0011, 0.0050, 0.0205, 0.0870, 0.2822, 0.4514, 0.1525, 0.0000;
  (4, 0) : 0.000000e+00, 0.000000e+00, 9.999003e-05, 2.999700e-04, 2.199780e-03, 1.919810e-02, 1.251870e-01, 8.530152e-01, 0.000000e+00;
  (0, 1) : 0.030497007, 0.967203222, 0.002299771, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (1, 1) : 0.0017, 0.0421, 0.4597, 0.4852, 0.0113, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0001, 0.0146, 0.3403, 0.6424, 0.0026, 0.0000, 0.0000, 0.0000;
  (3, 1) : 0.0001000199, 0.0004000797, 0.0021004183, 0.0101019917, 0.0516102577, 0.2184438209, 0.4646926190, 0.2525507929, 0.0000000000;
  (4, 1) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0001999799, 0.0013998594, 0.0136985940, 0.1028899547, 0.8818116120, 0.0000000000;
  (0, 2) : 0.0003999601, 0.0618938149, 0.8881112131, 0.0495950119, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (1, 2) : 0.0000000000, 0.0019001905, 0.0749075195, 0.5694571481, 0.3532350918, 0.0005000501, 0.0000000000, 0.0000000000, 0.0000000000;
  (2, 2) : 0.0000, 0.0000, 0.0007, 0.0498, 0.7640, 0.1854, 0.0001, 0.0000, 0.0000;
  (3, 2) : 0.0000000000, 0.0001000100, 0.0006000602, 0.0034003413, 0.0223022087, 0.1336130521, 0.4145411617, 0.4254431659, 0.0000000000;
  (4, 2) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0001000100, 0.0007000702, 0.0086008622, 0.0782078203, 0.9123912373, 0.0000000000;
  (0, 3) : 0.0000, 0.0001, 0.0655, 0.9082, 0.0262, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 3) : 0.000000000, 0.000000000, 0.002200218, 0.105110893, 0.825182158, 0.067506731, 0.000000000, 0.000000000, 0.000000000;
  (2, 3) : 0.0000, 0.0000, 0.0000, 0.0012, 0.1370, 0.8421, 0.0197, 0.0000, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0001, 0.0008, 0.0073, 0.0649, 0.3063, 0.6206, 0.0000;
  (4, 3) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000300030, 0.005000499, 0.056405593, 0.938293878, 0.000000000;
  (0, 4) : 0.0000, 0.0000, 0.0001, 0.0555, 0.9370, 0.0074, 0.0000, 0.0000, 0.0000;
  (1, 4) : 0.000000000, 0.000000000, 0.000000000, 0.002200219, 0.170016929, 0.806280661, 0.021502191, 0.000000000, 0.000000000;
  (2, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0034, 0.4375, 0.5591, 0.0000, 0.0000;
  (3, 4) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0001000101, 0.0016001608, 0.0226023120, 0.1747170926, 0.8009804245, 0.0000000000;
  (4, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0026, 0.0379, 0.9594, 0.0000;
  (0, 5) : 0.0000, 0.0000, 0.0000, 0.0002, 0.0491, 0.8863, 0.0644, 0.0000, 0.0000;
  (1, 5) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.002199779, 0.233776864, 0.762023558, 0.001999799, 0.000000000;
  (2, 5) : 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0208021, 0.8094809, 0.1697170, 0.0000000;
  (3, 5) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000200020, 0.005200521, 0.072607312, 0.921992147, 0.000000000;
  (4, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0012, 0.0230, 0.9758, 0.0000;
  (0, 6) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0002999701, 0.0955904363, 0.8966103408, 0.0074992528, 0.0000000000;
  (1, 6) : 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.010099, 0.489451, 0.500450, 0.000000;
  (2, 6) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00010001, 0.03920390, 0.96069609, 0.00000000;
  (3, 6) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00120012, 0.02960300, 0.96919688, 0.00000000;
  (4, 6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0005, 0.0140, 0.9855, 0.0000;
  (0, 7) : 0.0002000200, 0.0006000599, 0.0019001898, 0.0064006393, 0.0247024973, 0.0974096893, 0.2855289686, 0.5832579358, 0.0000000000;
  (1, 7) : 0.0001, 0.0002, 0.0007, 0.0026, 0.0117, 0.0585, 0.2222, 0.7040, 0.0000;
  (2, 7) : 0.0000, 0.0001, 0.0002, 0.0009, 0.0051, 0.0329, 0.1636, 0.7972, 0.0000;
  (3, 7) : 0.0000, 0.0000, 0.0001, 0.0004, 0.0022, 0.0159, 0.0994, 0.8820, 0.0000;
  (4, 7) : 0.0000, 0.0000, 0.0000, 0.0001, 0.0006, 0.0058, 0.0523, 0.9412, 0.0000;
  (0, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_CV_WD | R_MEDD2_ALLCV_WD ) {
  (0) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0001999799, 0.0108988943, 0.0983901488, 0.3165678354, 0.3531648164, 0.1748829091, 0.0402959790, 0.0055994371;
  (1) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0001999799, 0.0084991466, 0.0767922693, 0.2810718876, 0.3322668671, 0.2011799195, 0.0815917674, 0.0160983936, 0.0020997892, 0.0001999799;
  (2) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0002000202, 0.0139014121, 0.1120110974, 0.2918292539, 0.3134312727, 0.1915191666, 0.0610061531, 0.0130013113, 0.0028002824, 0.0003000303, 0.0000000000, 0.0000000000;
  (3) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0029002897, 0.0706070922, 0.3151319653, 0.3836379578, 0.1709169812, 0.0476047948, 0.0082008191, 0.0009000899, 0.0001000100, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (4) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0001999801, 0.0393961244, 0.3184681975, 0.4194582601, 0.1755821089, 0.0418958260, 0.0044995528, 0.0004999503, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (5) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0156983970, 0.3144689403, 0.4660529114, 0.1715829674, 0.0290970945, 0.0027997195, 0.0002999699, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (6) : 0.0000000000, 0.0000000000, 0.0308030901, 0.5769578154, 0.3398338913, 0.0482047846, 0.0040003987, 0.0002000199, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (7) : 0.000000000, 0.046504689, 0.848284805, 0.099909977, 0.005100509, 0.000200020, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (8) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_MEDD2_ALLAMP_WD | R_MEDD2_DISP_WD, R_MEDD2_EFFAXLOSS ) {
  (0, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0215, 0.9785;
  (1, 0) : 0.0000, 0.0000, 0.0000, 0.3192, 0.6448, 0.0360;
  (2, 0) : 0.0000, 0.0000, 0.0051, 0.9940, 0.0009, 0.0000;
  (3, 0) : 0.0000, 0.0000, 0.9945, 0.0055, 0.0000, 0.0000;
  (0, 1) : 0.0000, 0.0000, 0.0000, 0.3443, 0.6228, 0.0329;
  (1, 1) : 0.00000000, 0.00000000, 0.04740470, 0.93949398, 0.01290130, 0.00020002;
  (2, 1) : 0.0000, 0.0000, 0.9599, 0.0401, 0.0000, 0.0000;
  (3, 1) : 0.0000, 0.0000, 0.9999, 0.0001, 0.0000, 0.0000;
  (0, 2) : 0.0000, 0.0000, 0.0248, 0.9704, 0.0048, 0.0000;
  (1, 2) : 0.00000000, 0.00000000, 0.93309328, 0.06690672, 0.00000000, 0.00000000;
  (2, 2) : 0.0000, 0.0001, 0.9994, 0.0005, 0.0000, 0.0000;
  (3, 2) : 0.0000, 0.7508, 0.2492, 0.0000, 0.0000, 0.0000;
  (0, 3) : 0.0000, 0.1028, 0.8793, 0.0178, 0.0001, 0.0000;
  (1, 3) : 0.0000, 0.8756, 0.1237, 0.0007, 0.0000, 0.0000;
  (2, 3) : 0.0000, 0.9969, 0.0031, 0.0000, 0.0000, 0.0000;
  (3, 3) : 0.0000, 0.9999, 0.0001, 0.0000, 0.0000, 0.0000;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_MEDD2_AMP_WD | R_MEDD2_ALLAMP_WD ) {
  (0) : 0.7500748274, 0.1878189568, 0.0476047891, 0.0112010974, 0.0026002594, 0.0006000599, 0.0001000100, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (1) : 0.3184, 0.2478, 0.1780, 0.1158, 0.0688, 0.0380, 0.0189, 0.0086, 0.0036, 0.0014, 0.0005, 0.0002, 0.0000, 0.0000, 0.0000;
  (2) : 0.0117011899, 0.0392038663, 0.0935093196, 0.1676168558, 0.2189218117, 0.2095208198, 0.1465148740, 0.0747074358, 0.0287028753, 0.0078007733, 0.0016001586, 0.0002000198, 0.0000000000, 0.0000000000, 0.0000000000;
  (3) : 0.000000e+00, 0.000000e+00, 9.998996e-05, 1.299869e-03, 1.089890e-02, 5.269468e-02, 1.562839e-01, 2.701729e-01, 2.742729e-01, 1.635839e-01, 5.689428e-02, 1.219880e-02, 1.499849e-03, 9.998996e-05, 0.000000e+00;
  (4) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 9.999003e-05, 2.199781e-03, 2.179781e-02, 1.032900e-01, 2.591741e-01, 3.254671e-01, 2.091791e-01, 6.709332e-02, 1.079890e-02, 8.999103e-04;
  (5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0004, 0.0072, 0.0656, 0.2449, 0.3735, 0.2388, 0.0624, 0.0072;
}
probability ( R_MEDD2_ALLCV_EW | R_MEDD2_LSLOW_EW, R_MEDD2_DSLOW_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0264, 0.9149, 0.0587, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0000, 0.0218, 0.9469, 0.0313, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 0) : 0.0003000301, 0.0019001909, 0.0140014063, 0.0788079355, 0.3238321457, 0.4596462068, 0.1212120545, 0.0003000301, 0.0000000000;
  (4, 0) : 0.0000000000, 0.0000000000, 0.0001000101, 0.0005000504, 0.0041004130, 0.0384038280, 0.2145211566, 0.7423745419, 0.0000000000;
  (0, 1) : 0.030497007, 0.967203222, 0.002299771, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (1, 1) : 0.0006, 0.0944, 0.8841, 0.0209, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0007, 0.2183, 0.7790, 0.0020, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 1) : 9.998998e-05, 3.999600e-04, 4.299570e-03, 3.209680e-02, 1.955800e-01, 5.048500e-01, 2.603740e-01, 2.299770e-03, 0.000000e+00;
  (4, 1) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0002000201, 0.0021002108, 0.0239024088, 0.1648160610, 0.8089812993, 0.0000000000;
  (0, 2) : 0.0003999601, 0.0618938149, 0.8881112131, 0.0495950119, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (1, 2) : 0.0000, 0.0018, 0.1956, 0.7860, 0.0166, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0000, 0.0077, 0.4577, 0.5345, 0.0001, 0.0000, 0.0000, 0.0000;
  (3, 2) : 0.0000, 0.0001, 0.0007, 0.0074, 0.0735, 0.4044, 0.4938, 0.0201, 0.0000;
  (4, 2) : 0.0000, 0.0000, 0.0000, 0.0001, 0.0008, 0.0123, 0.1119, 0.8749, 0.0000;
  (0, 3) : 0.0000, 0.0001, 0.0655, 0.9082, 0.0262, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 3) : 0.0000, 0.0000, 0.0026, 0.2655, 0.7316, 0.0003, 0.0000, 0.0000, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0000, 0.0166, 0.9182, 0.0652, 0.0000, 0.0000, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0001, 0.0010, 0.0172, 0.2179, 0.6479, 0.1159, 0.0000;
  (4, 3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.0055, 0.0699, 0.9243, 0.0000;
  (0, 4) : 0.0000, 0.0000, 0.0001, 0.0555, 0.9370, 0.0074, 0.0000, 0.0000, 0.0000;
  (1, 4) : 0.0000, 0.0000, 0.0000, 0.0044, 0.5355, 0.4600, 0.0001, 0.0000, 0.0000;
  (2, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0398, 0.9460, 0.0142, 0.0000, 0.0000;
  (3, 4) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 9.998998e-05, 1.799820e-03, 5.589439e-02, 4.600539e-01, 4.821519e-01, 0.000000e+00;
  (4, 4) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0021, 0.0390, 0.9588, 0.0000;
  (0, 5) : 0.0000, 0.0000, 0.0000, 0.0002, 0.0491, 0.8863, 0.0644, 0.0000, 0.0000;
  (1, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0047, 0.5053, 0.4900, 0.0000, 0.0000;
  (2, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.1104, 0.8881, 0.0013, 0.0000;
  (3, 5) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0041, 0.1033, 0.8925, 0.0000;
  (4, 5) : 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00069993, 0.01889810, 0.98040197, 0.00000000;
  (0, 6) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0002999701, 0.0955904363, 0.8966103408, 0.0074992528, 0.0000000000;
  (1, 6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0218, 0.8352, 0.1430, 0.0000;
  (2, 6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0020, 0.3203, 0.6777, 0.0000;
  (3, 6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.0194, 0.9803, 0.0000;
  (4, 6) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0093, 0.9905, 0.0000;
  (0, 7) : 0.0002000200, 0.0006000599, 0.0019001898, 0.0064006393, 0.0247024973, 0.0974096893, 0.2855289686, 0.5832579358, 0.0000000000;
  (1, 7) : 0.0001, 0.0003, 0.0010, 0.0036, 0.0154, 0.0712, 0.2467, 0.6617, 0.0000;
  (2, 7) : 0.0000000000, 0.0001000100, 0.0005000499, 0.0019001897, 0.0093009283, 0.0504049909, 0.2072209627, 0.7305728685, 0.0000000000;
  (3, 7) : 0.0000, 0.0000, 0.0001, 0.0004, 0.0026, 0.0190, 0.1153, 0.8626, 0.0000;
  (4, 7) : 0.0000, 0.0000, 0.0000, 0.0001, 0.0006, 0.0062, 0.0562, 0.9369, 0.0000;
  (0, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 8) : 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_CV_EW | R_MEDD2_ALLCV_EW ) {
  (0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0012, 0.0172, 0.1126, 0.2484, 0.3210, 0.1935, 0.0803, 0.0258;
  (1) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0009999003, 0.0148985043, 0.0995900289, 0.2261770656, 0.2933710851, 0.2261770656, 0.1019900296, 0.0288971084, 0.0064993519, 0.0013998604;
  (2) : 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0025002518, 0.0317032225, 0.1445141026, 0.2639261874, 0.2925292077, 0.1670171186, 0.0669067475, 0.0245025174, 0.0054005438, 0.0009000906, 0.0001000101, 0.0000000000;
  (3) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0005, 0.0189, 0.1567, 0.3371, 0.2862, 0.1429, 0.0465, 0.0095, 0.0014, 0.0003, 0.0000, 0.0000, 0.0000, 0.0000;
  (4) : 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.049901e-02, 1.564841e-01, 3.778623e-01, 3.006703e-01, 1.251871e-01, 2.449762e-02, 4.199584e-03, 4.999504e-04, 9.999009e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (5) : 0.0000, 0.0000, 0.0000, 0.0045, 0.1674, 0.4536, 0.2834, 0.0774, 0.0118, 0.0017, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (6) : 0.00000000, 0.00000000, 0.01090110, 0.42284216, 0.44464416, 0.10661104, 0.01370141, 0.00120012, 0.00010001, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000;
  (7) : 0.0000, 0.0199, 0.8041, 0.1629, 0.0125, 0.0006, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (8) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
probability ( R_MEDD2_BLOCK_EW | R_DIFFN_MEDD2_BLOCK, R_LNLBE_MEDD2_BLOCK_EW ) {
  (0, 0) : 1.0, 0.0, 0.0, 0.0, 0.0;
  (1, 0) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (3, 0) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (4, 0) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 1) : 0.0077, 0.9923, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0001, 0.4980, 0.5019, 0.0000, 0.0000;
  (2, 1) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (3, 1) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (4, 1) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 2) : 0.0007, 0.0234, 0.9759, 0.0000, 0.0000;
  (1, 2) : 0.0000, 0.0032, 0.9968, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0007, 0.4328, 0.5665, 0.0000;
  (3, 2) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (4, 2) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 3) : 0.0019, 0.0060, 0.0588, 0.9333, 0.0000;
  (1, 3) : 0.0010, 0.0033, 0.0376, 0.9581, 0.0000;
  (2, 3) : 0.0003, 0.0011, 0.0159, 0.9827, 0.0000;
  (3, 3) : 0.0003000299, 0.0011001095, 0.0125012945, 0.9860985661, 0.0000000000;
  (4, 3) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (0, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (1, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (2, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (3, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
  (4, 4) : 0.0, 0.0, 0.0, 0.0, 1.0;
}
probability ( R_MEDD2_DISP_EWD | R_MEDD2_DISP_WD, R_MEDD2_DISP_EW ) {
  (0, 0) : 0.0000, 0.0000, 0.0742, 0.9045, 0.0213, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0742, 0.9045, 0.0213, 0.0000, 0.0000;
  (2, 0) : 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.133, 0.867;
  (3, 0) : 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.133, 0.867;
  (0, 1) : 0.0001, 0.2215, 0.7732, 0.0052, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0000000, 0.0000000, 0.0000000, 0.1315130, 0.8576859, 0.0108011, 0.0000000, 0.0000000, 0.0000000;
  (2, 1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0108, 0.8577, 0.1315;
  (3, 1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0108, 0.8577, 0.1315;
  (0, 2) : 0.1315, 0.8577, 0.0108, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 2) : 0.0000, 0.0000, 0.0742, 0.9045, 0.0213, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0052, 0.7732, 0.2215, 0.0001;
  (3, 2) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0052, 0.7732, 0.2215, 0.0001;
  (0, 3) : 0.9933, 0.0067, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 3) : 0.0404, 0.9192, 0.0404, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 3) : 0.0000, 0.0000, 0.0213, 0.9045, 0.0742, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 3) : 0.0000, 0.0000, 0.0213, 0.9045, 0.0742, 0.0000, 0.0000, 0.0000, 0.0000;
}
probability ( R_MEDD2_AMPR_EW | R_MEDD2_DISP_EWD, R_MEDD2_BLOCK_EW ) {
  (0, 0) : 0.0000000000, 0.2826720254, 0.7164280645, 0.0008999101, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000;
  (1, 0) : 0.0000, 0.0000, 0.5547, 0.4268, 0.0183, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 0) : 0.000000000, 0.000000000, 0.009000898, 0.512750892, 0.415241913, 0.058905888, 0.003900389, 0.000200020, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (3, 0) : 0.0000, 0.0000, 0.0000, 0.0635, 0.4459, 0.3732, 0.0995, 0.0162, 0.0015, 0.0002, 0.0000, 0.0000;
  (4, 0) : 0.00000000, 0.00000000, 0.00000000, 0.00290029, 0.11411100, 0.39514002, 0.31983201, 0.13151301, 0.02980300, 0.00580058, 0.00090009, 0.00000000;
  (5, 0) : 0.0000, 0.0000, 0.0000, 0.0001, 0.0136, 0.1578, 0.3285, 0.2966, 0.1404, 0.0483, 0.0135, 0.0012;
  (6, 0) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.001200241, 0.038507718, 0.174635080, 0.301560139, 0.261352120, 0.144729067, 0.065113030, 0.012902606;
  (7, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0062, 0.0580, 0.1828, 0.2779, 0.2392, 0.1675, 0.0683;
  (8, 0) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0009, 0.0157, 0.0823, 0.2012, 0.2516, 0.2559, 0.1924;
  (0, 1) : 0.0000, 0.9103, 0.0897, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 1) : 0.0000, 0.0004, 0.8945, 0.1036, 0.0015, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 1) : 0.000000000, 0.000000000, 0.114011009, 0.716972057, 0.159816013, 0.008900891, 0.000300030, 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.000000000;
  (3, 1) : 0.0000, 0.0000, 0.0026, 0.3111, 0.5155, 0.1499, 0.0190, 0.0018, 0.0001, 0.0000, 0.0000, 0.0000;
  (4, 1) : 0.0000, 0.0000, 0.0000, 0.0451, 0.3717, 0.4039, 0.1433, 0.0313, 0.0041, 0.0005, 0.0001, 0.0000;
  (5, 1) : 0.0000, 0.0000, 0.0000, 0.0035, 0.1122, 0.3738, 0.3186, 0.1444, 0.0376, 0.0083, 0.0015, 0.0001;
  (6, 1) : 0.00000000, 0.00000000, 0.00000000, 0.00020002, 0.02260230, 0.18901901, 0.33173302, 0.27442701, 0.12521301, 0.04310430, 0.01240120, 0.00130013;
  (7, 1) : 0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.003100311, 0.061506212, 0.210921040, 0.305131058, 0.234423045, 0.121712023, 0.052805310, 0.010401002;
  (8, 1) : 0.0000, 0.0000, 0.0000, 0.0000, 0.0004, 0.0166, 0.1002, 0.2325, 0.2777, 0.2039, 0.1251, 0.0436;
  (0, 2) : 0.0000, 0.9974, 0.0026, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (1, 2) : 0.0000, 0.3856, 0.6048, 0.0095, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (2, 2) : 0.0000, 0.0073, 0.8191, 0.1638, 0.0095, 0.0003, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;
  (3, 2) : 0.0000, 0.0001, 0.3860, 0.4993, 0.1036, 0.0101, 0.0008, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000;
  (4, 2) : 0.0000, 0.0000, 0.0913, 0.5259, 0.3027, 0.0681, 0.0104, 0.0014, 0.0002, 0.0000, 0.0000, 0.0000;
  (5, 2) : 0.000000e+00, 0.000000e+00, 1.499850e-02, 3.068690e-01, 4.203580e-01, 1.928810e-01, 5.119490e-02, 1.139890e-02, 1.899810e-03, 2.999700e-04, 9.998998e-05, 0.000000e+00;
  (6, 2) : 0.0000, 0.0000, 0.0023, 0.1333, 0.3728, 0.3075, 0.1292, 0.0421, 0.0099, 0.0024, 0.0005, 0.0000;
  (7, 2) : 0.0000000000, 0.0000000000, 0.0003000302, 0.0444044369, 0.2413242003, 0.3431342848, 0.2208221833, 0.1025100851, 0.0337034280, 0.0104010086, 0.0030003025, 0.0004000403;
  (8, 2) : 0.0000, 0.0000, 0.0000, 0.0138, 0.1311, 0.2956, 0.2729, 0.1711, 0.0745, 0.0286, 0.0104, 0.0020;
  (0, 3) : 0.000000e+00, 9.467053e-01, 4.919512e-02, 3.499651e-03, 4.999502e-04, 9.999003e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (1, 3) : 0.000000e+00, 8.721126e-01, 1.109890e-01, 1.349870e-02, 2.499749e-03, 5.999398e-04, 1.999799e-04, 9.998995e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00;
  (2, 3) : 0.0000000000, 0.7782557821, 0.1801359496, 0.0312061913, 0.0075014979, 0.0020003994, 0.0006001198, 0.0002000399, 0.0001000200, 0.0000000000, 0.0000000000, 0.0000000000;
  (3, 3) : 0.0000, 0.6780, 0.2436, 0.0548, 0.0156, 0.0050, 0.0018, 0.0007, 0.0003, 0.0001, 0.0001, 0.0000;
  (4, 3) : 0.0000, 0.5789, 0.2957, 0.0820, 0.0270, 0.0096, 0.0037, 0.0017, 0.0007, 0.0004, 0.0002, 0.0001;
  (5, 3) : 0.0000, 0.4850, 0.3340, 0.1106, 0.0411, 0.0162, 0.0068, 0.0033, 0.0015, 0.0008, 0.0004, 0.0003;
  (6, 3) : 0.0000000000, 0.4048399271, 0.3566359358, 0.1367139754, 0.0562055899, 0.0240023957, 0.0108010981, 0.0054005390, 0.0027002695, 0.0014001397, 0.0008000799, 0.0005000499;
  (7, 3) : 0.000000000, 0.331633656, 0.367126619, 0.161267832, 0.072685424, 0.033493265, 0.015996783, 0.008498291, 0.004399115, 0.002399518, 0.001499698, 0.000999799;
  (8, 3) : 0.0000, 0.2731, 0.3669, 0.1808, 0.0881, 0.0434, 0.0217, 0.0120, 0.0064, 0.0036, 0.0023, 0.0017;
  (0, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (1, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (2, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (3, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (4, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (5, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (6, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (7, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  (8, 4) : 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
}
