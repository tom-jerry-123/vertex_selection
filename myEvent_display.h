//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Mar 10 17:52:49 2023 by ROOT version 6.06/02
// from TTree EventTree/EventTree
// found on file: /eos/user/l/losanti/ITkNtuples/OutDir_1/hist-Rel21sample.root
//////////////////////////////////////////////////////////

#ifndef myEvent_display_h
#define myEvent_display_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include <vector>
#include <string.h>
#include <iostream>
#include <fstream>
using namespace std;
// Fixed size dimensions of array or collections stored in the TTree if any.

class myEvent_display {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           mcChannelNumber;
   Int_t           EventNumber;
   Int_t           RunNumber;
   Int_t           BCID;
   Float_t         mu;
   Float_t         muActual;
   Float_t         beamspot_x;
   Float_t         beamspot_y;
   Float_t         beamspot_z;
   Float_t         beamspot_sigX;
   Float_t         beamspot_sigY;
   Float_t         beamspot_sigZ;
   vector<float>   *met_Truth;
   vector<float>   *mpx_Truth;
   vector<float>   *mpy_Truth;
   vector<float>   *sumet_Truth;
   vector<float>   *track_prob;
   vector<float>   *track_d0;
   vector<float>   *track_z0;
   vector<float>   *track_theta;
   vector<float>   *track_phi;
   vector<float>   *track_qOverP;
   //vector<float>   *track_t;
   //vector<float>   *track_z;
   vector<float>* track_z =0;
   vector<float>* track_t =0;

   vector<float>   *track_var_d0;
   vector<float>   *track_var_z0;
   vector<float>   *track_var_phi;
   vector<float>   *track_var_theta;
   vector<float>   *track_var_qOverP;
   vector<float>   *track_cov_d0z0;
   vector<float>   *track_cov_d0phi;
   vector<float>   *track_cov_d0theta;
   vector<float>   *track_cov_d0qOverP;
   vector<float>   *track_cov_z0phi;
   vector<float>   *track_cov_z0theta;
   vector<float>   *track_cov_z0qOverP;
   vector<float>   *track_cov_phitheta;
   vector<float>   *track_cov_phiqOverP;
   vector<float>   *track_cov_tehtaqOverP;
   vector<float>   *track_t30;
   vector<float>   *track_t60;
   vector<float>   *track_t90;
   vector<int>     *tracks_numPix;
   vector<int>     *tracks_numSCT;
   vector<int>     *tracks_numPix1L;
   vector<int>     *tracks_numPix2L;
   vector<int>     *track_status;
   vector<float>   *jet_pt;
   vector<float>   *jet_eta;
   vector<float>   *jet_phi;
   vector<float>   *jet_m;
   vector<float>   *jet_q;
   vector<float>   *jet_ptmatched_pt;
   vector<float>   *jet_ptmatched_eta;
   vector<float>   *jet_ptmatched_phi;
   vector<float>   *jet_ptmatched_m;
   vector<float>   *jet_drmatched_pt;
   vector<float>   *jet_drmatched_eta;
   vector<float>   *jet_drmatched_phi;
   vector<float>   *jet_drmatched_m;
   vector<int>     *jet_isPU;
   vector<int>     *jet_isHS;
   vector<int>     *jet_label;
   vector<float>   *recovertex_x;
   vector<float>   *recovertex_y;
   vector<float>   *recovertex_z;
   vector<float>   *recovertex_sumPt2;
   vector<int>     *recovertex_isPU;
   vector<int>     *recovertex_isHS;
   vector<float>   *truthvertex_x;
   vector<float>   *truthvertex_y;
   vector<float>   *truthvertex_z;
   vector<float>   *truthvertex_t;
   vector<int>     *truthvertex_isPU;
   vector<int>     *truthvertex_isHS;
   vector<vector<int> > *jet_tracks_idx;
   vector<vector<int> > *recovertex_tracks_idx;
   vector<vector<int> > *truthvertex_tracks_idx;
   vector<vector<float> > *recovertex_tracks_weight;

   // List of branches
   TBranch        *b_mcChannelNumber;   //!
   TBranch        *b_EventNumber;   //!
   TBranch        *b_RunNumber;   //!
   TBranch        *b_BCID;   //!
   TBranch        *b_mu;   //!
   TBranch        *b_muActual;   //!
   TBranch        *b_beamspot_x;   //!
   TBranch        *b_beamspot_y;   //!
   TBranch        *b_beamspot_z;   //!
   TBranch        *b_beamspot_sigX;   //!
   TBranch        *b_beamspot_sigY;   //!
   TBranch        *b_beamspot_sigZ;   //!
   TBranch        *b_met_Truth;   //!
   TBranch        *b_mpx_Truth;   //!
   TBranch        *b_mpy_Truth;   //!
   TBranch        *b_sumet_Truth;   //!
   TBranch        *b_track_prob;   //!
   TBranch        *b_track_d0;   //!
   TBranch        *b_track_z0;   //!
   TBranch        *b_track_theta;   //!
   TBranch        *b_track_phi;   //!
   TBranch        *b_track_qOverP;   //!
   TBranch        *b_track_t;   //!
   TBranch        *b_track_z;   //!
   TBranch        *b_track_var_d0;   //!
   TBranch        *b_track_var_z0;   //!
   TBranch        *b_track_var_phi;   //!
   TBranch        *b_track_var_theta;   //!
   TBranch        *b_track_var_qOverP;   //!
   TBranch        *b_track_cov_d0z0;   //!
   TBranch        *b_track_cov_d0phi;   //!
   TBranch        *b_track_cov_d0theta;   //!
   TBranch        *b_track_cov_d0qOverP;   //!
   TBranch        *b_track_cov_z0phi;   //!
   TBranch        *b_track_cov_z0theta;   //!
   TBranch        *b_track_cov_z0qOverP;   //!
   TBranch        *b_track_cov_phitheta;   //!
   TBranch        *b_track_cov_phiqOverP;   //!
   TBranch        *b_track_cov_tehtaqOverP;   //!
   TBranch        *b_track_t30;   //!
   TBranch        *b_track_t60;   //!
   TBranch        *b_track_t90;   //!
   TBranch        *b_tracks_numPix;   //!
   TBranch        *b_tracks_numSCT;   //!
   TBranch        *b_tracks_numPix1L;   //!
   TBranch        *b_tracks_numPix2L;   //!
   TBranch        *b_track_status;   //!
   TBranch        *b_jet_pt;   //!
   TBranch        *b_jet_eta;   //!
   TBranch        *b_jet_phi;   //!
   TBranch        *b_jet_m;   //!
   TBranch        *b_jet_q;   //!
   TBranch        *b_jet_ptmatched_pt;   //!
   TBranch        *b_jet_ptmatched_eta;   //!
   TBranch        *b_jet_ptmatched_phi;   //!
   TBranch        *b_jet_ptmatched_m;   //!
   TBranch        *b_jet_drmatched_pt;   //!
   TBranch        *b_jet_drmatched_eta;   //!
   TBranch        *b_jet_drmatched_phi;   //!
   TBranch        *b_jet_drmatched_m;   //!
   TBranch        *b_jet_isPU;   //!
   TBranch        *b_jet_isHS;   //!
   TBranch        *b_jet_label;   //!
   TBranch        *b_recovertex_x;   //!
   TBranch        *b_recovertex_y;   //!
   TBranch        *b_recovertex_z;   //!
   TBranch        *b_recovertex_sumPt2;   //!
   TBranch        *b_recovertex_isPU;   //!
   TBranch        *b_recovertex_isHS;   //!
   TBranch        *b_truthvertex_x;   //!
   TBranch        *b_truthvertex_y;   //!
   TBranch        *b_truthvertex_z;   //!
   TBranch        *b_truthvertex_t;   //!
   TBranch        *b_truthvertex_isPU;   //!
   TBranch        *b_truthvertex_isHS;   //!
   TBranch        *b_jet_tracks_idx;   //!
   TBranch        *b_recovertex_tracks_idx;   //!
   TBranch        *b_truthvertex_tracks_idx;   //!
   TBranch        *b_recovertex_tracks_weight;   //!

   myEvent_display(TTree *tree=0);
   virtual ~myEvent_display();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
    


   //Z-rho event display (vertex tracks)
   void DisplayZRho(int event=100, int vtxID = 0, bool drawHGTD = false,
                    int TrkSelection = 0, bool NEWSEL = false);
    
    //INTERNAL METHODS
     //----------------
     
     int VS2();
    
    float ComputeVS0ForVertex(int INDEX);
    float ComputeVS1ForVertex(int INDEX);
    float ComputeVSA1ForVertex(int INDEX);
    float ComputeVSA2ForVertex(int INDEX);

    int GetMatchedVertex();
    
    
};

#endif

#ifdef myEvent_display_cxx
myEvent_display::myEvent_display(TTree *tree) : fChain(0)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
       TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("tt_hist-Rel21_19files.root"); //hist-Rel21sample_199files.root");
       //TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/Users/wasikul/Desktop/4D_tracking/VBF_hist-Rel21sample.root"); //hist-Rel21sample_199files.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("tt_hist-Rel21_19files.root");
         //f = new TFile("/Users/wasikul/Desktop/4D_tracking/VBF_hist-Rel21sample.root");
      /*
       TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("VBF_hist-Rel21sample_199files.root"); //hist-Rel21sample_199files.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("VBF_hist-Rel21sample_199files.root");
       */
      }
      f->GetObject("EventTree",tree);

   }
   Init(tree);
}

myEvent_display::~myEvent_display()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t myEvent_display::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t myEvent_display::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}




void myEvent_display::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   met_Truth = 0;
   mpx_Truth = 0;
   mpy_Truth = 0;
   sumet_Truth = 0;
   track_prob = 0;
   track_d0 = 0;
   track_z0 = 0;
   track_theta = 0;
   track_phi = 0;
   track_qOverP = 0;
   track_t = 0;
   track_z = 0;
   track_var_d0 = 0;
   track_var_z0 = 0;
   track_var_phi = 0;
   track_var_theta = 0;
   track_var_qOverP = 0;
   track_cov_d0z0 = 0;
   track_cov_d0phi = 0;
   track_cov_d0theta = 0;
   track_cov_d0qOverP = 0;
   track_cov_z0phi = 0;
   track_cov_z0theta = 0;
   track_cov_z0qOverP = 0;
   track_cov_phitheta = 0;
   track_cov_phiqOverP = 0;
   track_cov_tehtaqOverP = 0;
   track_t30 = 0;
   track_t60 = 0;
   track_t90 = 0;
   tracks_numPix = 0;
   tracks_numSCT = 0;
   tracks_numPix1L = 0;
   tracks_numPix2L = 0;
   track_status = 0;
   jet_pt = 0;
   jet_eta = 0;
   jet_phi = 0;
   jet_m = 0;
   jet_q = 0;
   jet_ptmatched_pt = 0;
   jet_ptmatched_eta = 0;
   jet_ptmatched_phi = 0;
   jet_ptmatched_m = 0;
   jet_drmatched_pt = 0;
   jet_drmatched_eta = 0;
   jet_drmatched_phi = 0;
   jet_drmatched_m = 0;
   jet_isPU = 0;
   jet_isHS = 0;
   jet_label = 0;
   recovertex_x = 0;
   recovertex_y = 0;
   recovertex_z = 0;
   recovertex_sumPt2 = 0;
   recovertex_isPU = 0;
   recovertex_isHS = 0;
   truthvertex_x = 0;
   truthvertex_y = 0;
   truthvertex_z = 0;
   truthvertex_t = 0;
   truthvertex_isPU = 0;
   truthvertex_isHS = 0;
   jet_tracks_idx = 0;
   recovertex_tracks_idx = 0;
   truthvertex_tracks_idx = 0;
   recovertex_tracks_weight = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("mcChannelNumber", &mcChannelNumber, &b_mcChannelNumber);
   fChain->SetBranchAddress("EventNumber", &EventNumber, &b_EventNumber);
   fChain->SetBranchAddress("RunNumber", &RunNumber, &b_RunNumber);
   fChain->SetBranchAddress("BCID", &BCID, &b_BCID);
   fChain->SetBranchAddress("mu", &mu, &b_mu);
   fChain->SetBranchAddress("muActual", &muActual, &b_muActual);
   fChain->SetBranchAddress("beamspot_x", &beamspot_x, &b_beamspot_x);
   fChain->SetBranchAddress("beamspot_y", &beamspot_y, &b_beamspot_y);
   fChain->SetBranchAddress("beamspot_z", &beamspot_z, &b_beamspot_z);
   fChain->SetBranchAddress("beamspot_sigX", &beamspot_sigX, &b_beamspot_sigX);
   fChain->SetBranchAddress("beamspot_sigY", &beamspot_sigY, &b_beamspot_sigY);
   fChain->SetBranchAddress("beamspot_sigZ", &beamspot_sigZ, &b_beamspot_sigZ);
   fChain->SetBranchAddress("met_Truth", &met_Truth, &b_met_Truth);
   fChain->SetBranchAddress("mpx_Truth", &mpx_Truth, &b_mpx_Truth);
   fChain->SetBranchAddress("mpy_Truth", &mpy_Truth, &b_mpy_Truth);
   fChain->SetBranchAddress("sumet_Truth", &sumet_Truth, &b_sumet_Truth);
   fChain->SetBranchAddress("track_prob", &track_prob, &b_track_prob);
   fChain->SetBranchAddress("track_d0", &track_d0, &b_track_d0);
   fChain->SetBranchAddress("track_z0", &track_z0, &b_track_z0);
   fChain->SetBranchAddress("track_theta", &track_theta, &b_track_theta);
   fChain->SetBranchAddress("track_phi", &track_phi, &b_track_phi);
   fChain->SetBranchAddress("track_qOverP", &track_qOverP, &b_track_qOverP);
   //fChain->SetBranchAddress("track_t", &track_t, &b_track_t);
   //fChain->SetBranchAddress("track_z", &track_z, &b_track_z);
   fChain->SetBranchAddress("track_var_d0", &track_var_d0, &b_track_var_d0);
   fChain->SetBranchAddress("track_var_z0", &track_var_z0, &b_track_var_z0);
   fChain->SetBranchAddress("track_var_phi", &track_var_phi, &b_track_var_phi);
   fChain->SetBranchAddress("track_var_theta", &track_var_theta, &b_track_var_theta);
   fChain->SetBranchAddress("track_var_qOverP", &track_var_qOverP, &b_track_var_qOverP);
   fChain->SetBranchAddress("track_cov_d0z0", &track_cov_d0z0, &b_track_cov_d0z0);
   fChain->SetBranchAddress("track_cov_d0phi", &track_cov_d0phi, &b_track_cov_d0phi);
   fChain->SetBranchAddress("track_cov_d0theta", &track_cov_d0theta, &b_track_cov_d0theta);
   fChain->SetBranchAddress("track_cov_d0qOverP", &track_cov_d0qOverP, &b_track_cov_d0qOverP);
   fChain->SetBranchAddress("track_cov_z0phi", &track_cov_z0phi, &b_track_cov_z0phi);
   fChain->SetBranchAddress("track_cov_z0theta", &track_cov_z0theta, &b_track_cov_z0theta);
   fChain->SetBranchAddress("track_cov_z0qOverP", &track_cov_z0qOverP, &b_track_cov_z0qOverP);
   fChain->SetBranchAddress("track_cov_phitheta", &track_cov_phitheta, &b_track_cov_phitheta);
   fChain->SetBranchAddress("track_cov_phiqOverP", &track_cov_phiqOverP, &b_track_cov_phiqOverP);
   fChain->SetBranchAddress("track_cov_tehtaqOverP", &track_cov_tehtaqOverP, &b_track_cov_tehtaqOverP);
   fChain->SetBranchAddress("track_t30", &track_t30, &b_track_t30);
   fChain->SetBranchAddress("track_t60", &track_t60, &b_track_t60);
   fChain->SetBranchAddress("track_t90", &track_t90, &b_track_t90);
   fChain->SetBranchAddress("tracks_numPix", &tracks_numPix, &b_tracks_numPix);
   fChain->SetBranchAddress("tracks_numSCT", &tracks_numSCT, &b_tracks_numSCT);
   fChain->SetBranchAddress("tracks_numPix1L", &tracks_numPix1L, &b_tracks_numPix1L);
   fChain->SetBranchAddress("tracks_numPix2L", &tracks_numPix2L, &b_tracks_numPix2L);
   fChain->SetBranchAddress("track_status", &track_status, &b_track_status);
   fChain->SetBranchAddress("jet_pt", &jet_pt, &b_jet_pt);
   fChain->SetBranchAddress("jet_eta", &jet_eta, &b_jet_eta);
   fChain->SetBranchAddress("jet_phi", &jet_phi, &b_jet_phi);
   fChain->SetBranchAddress("jet_m", &jet_m, &b_jet_m);
   fChain->SetBranchAddress("jet_q", &jet_q, &b_jet_q);
   fChain->SetBranchAddress("jet_ptmatched_pt", &jet_ptmatched_pt, &b_jet_ptmatched_pt);
   fChain->SetBranchAddress("jet_ptmatched_eta", &jet_ptmatched_eta, &b_jet_ptmatched_eta);
   fChain->SetBranchAddress("jet_ptmatched_phi", &jet_ptmatched_phi, &b_jet_ptmatched_phi);
   fChain->SetBranchAddress("jet_ptmatched_m", &jet_ptmatched_m, &b_jet_ptmatched_m);
   fChain->SetBranchAddress("jet_drmatched_pt", &jet_drmatched_pt, &b_jet_drmatched_pt);
   fChain->SetBranchAddress("jet_drmatched_eta", &jet_drmatched_eta, &b_jet_drmatched_eta);
   fChain->SetBranchAddress("jet_drmatched_phi", &jet_drmatched_phi, &b_jet_drmatched_phi);
   fChain->SetBranchAddress("jet_drmatched_m", &jet_drmatched_m, &b_jet_drmatched_m);
   fChain->SetBranchAddress("jet_isPU", &jet_isPU, &b_jet_isPU);
   fChain->SetBranchAddress("jet_isHS", &jet_isHS, &b_jet_isHS);
   fChain->SetBranchAddress("jet_label", &jet_label, &b_jet_label);
   fChain->SetBranchAddress("recovertex_x", &recovertex_x, &b_recovertex_x);
   fChain->SetBranchAddress("recovertex_y", &recovertex_y, &b_recovertex_y);
   fChain->SetBranchAddress("recovertex_z", &recovertex_z, &b_recovertex_z);
   fChain->SetBranchAddress("recovertex_sumPt2", &recovertex_sumPt2, &b_recovertex_sumPt2);
   fChain->SetBranchAddress("recovertex_isPU", &recovertex_isPU, &b_recovertex_isPU);
   fChain->SetBranchAddress("recovertex_isHS", &recovertex_isHS, &b_recovertex_isHS);
   fChain->SetBranchAddress("truthvertex_x", &truthvertex_x, &b_truthvertex_x);
   fChain->SetBranchAddress("truthvertex_y", &truthvertex_y, &b_truthvertex_y);
   fChain->SetBranchAddress("truthvertex_z", &truthvertex_z, &b_truthvertex_z);
   fChain->SetBranchAddress("truthvertex_t", &truthvertex_t, &b_truthvertex_t);
   fChain->SetBranchAddress("truthvertex_isPU", &truthvertex_isPU, &b_truthvertex_isPU);
   fChain->SetBranchAddress("truthvertex_isHS", &truthvertex_isHS, &b_truthvertex_isHS);
   fChain->SetBranchAddress("jet_tracks_idx", &jet_tracks_idx, &b_jet_tracks_idx);
   fChain->SetBranchAddress("recovertex_tracks_idx", &recovertex_tracks_idx, &b_recovertex_tracks_idx);
   fChain->SetBranchAddress("truthvertex_tracks_idx", &truthvertex_tracks_idx, &b_truthvertex_tracks_idx);
   fChain->SetBranchAddress("recovertex_tracks_weight", &recovertex_tracks_weight, &b_recovertex_tracks_weight);
   Notify();
}

Bool_t myEvent_display::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void myEvent_display::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t myEvent_display::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef myEvent_display_cxx
