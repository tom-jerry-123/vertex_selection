#define myEvent_display_cxx
#include "myEvent_display.h"
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TFile.h"
#include "TTree.h"
#include "TLine.h"
#include <cmath>
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "TLorentzVector.h"
#include "TFile.h"
#include "TTree.h"
#include "TProfile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TGraph.h"
#include "TChain.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TVector3.h"
#include "TRandom.h"
#include "TROOT.h"
#include "TKey.h"
#include <map>
#include "TColor.h"

using namespace ROOT;
typedef std::pair<float, int> pairs;
//using namespace std;
TString save_file_as = "testing";
//TString save_file_as = "forward_region/time_truth_cut_4_t30_hpt3";
TString save_Evt_info = "Event_display_plots/Evt_no_";
//TString save_file_as = "Evt_no_";

//#include "fastjet/ClusterSequence.hh"
//#include "fastjet/PseudoJet.hh"

//using namespace fastjet;

#include <map>
#include <string>
#include "TColor.h"

struct Color {
    int r, g, b;
};

double getNewDzpara( float ETA, float PT)
{
  ETA = fabs(ETA);
  double *p_v = new double[7];
  if( (PT <= 1.5) ){
    double p_tmp[] = { 0.0314036, 0.790955, -2.65987, 3.62073, -2.18228, 0.614866, -0.0634521 };
    std::copy(p_tmp,p_tmp+7,p_v);
  }
  if( (PT > 1.5) && (PT <=2.5) ){
    double p_tmp[] = { 0.0229273, 0.540101, -1.80727, 2.45187, -1.47382, 0.414345, -0.0426769 };
    std::copy(p_tmp,p_tmp+7,p_v);
  }
  if( (PT > 2.5) && (PT <=5.0) ){
    double p_tmp[] = { 0.0163773, 0.345112, -1.14474, 1.54382, -0.923523, 0.258617, -0.0265446 };
    std::copy(p_tmp,p_tmp+7,p_v);
  }
  if( (PT > 5) && (PT <= 10) ){
    double p_tmp[] = { 0.010919, 0.179329, -0.581971, 0.773186, -0.45679, 0.126608, -0.012875 };
    std::copy(p_tmp,p_tmp+7,p_v);
  }
  if( (PT > 10) ){
    double p_tmp[] = { 0.00835945, 0.0957783, -0.299255, 0.38722, -0.22351, 0.0607521, -0.00606524 };
    std::copy(p_tmp,p_tmp+7,p_v);
  }
  
  double Dzpara =  p_v[0] + p_v[1]*ETA + p_v[2]*pow(ETA,2) +p_v[3]*pow(ETA,3) + p_v[4]*pow(ETA,4);
  Dzpara += p_v[5]*pow(ETA,5)+p_v[6]*pow(ETA,6);
  Dzpara *= 2.0;
  return fabs(Dzpara);
}

double getDzpara( float ETA, float PT, bool ptbin)
{
  double *p_v=new double[6];
  ETA = fabs(ETA);
  if(ETA<2.5){
    double p_tmp[]={0.126153, -0.457555, 0.891792, -0.767631, 0.31876, -0.0471466};
    std::copy(p_tmp,p_tmp+6,p_v);
  } else{
    double p_tmp[]={-694.194, 1117.99, -714.316, 226.29, -35.5291, 2.2137};
    std::copy(p_tmp,p_tmp+6,p_v);
  }
  
  if(ETA<2.5) ETA=2.5;
  
  double Dzpara = p_v[0] + p_v[1]*ETA + p_v[2]*pow(ETA,2) +p_v[3]*pow(ETA,3) +
    p_v[4]*pow(ETA,4) + p_v[5]*pow(ETA,5);
  Dzpara*=2.0;
  return fabs(Dzpara);
}


void myText(Double_t x,Double_t y,Color_t color,char *text) 
{
  TLatex l; 
  l.SetNDC();
  l.SetTextColor(color);
  l.SetTextSize(0.04);
  l.DrawLatex(x,y,text);
}

void ATLAS_LABEL(Double_t x,Double_t y,Color_t color) 
{
  TLatex *   tex = new TLatex(x,y,"ATLAS Simulation Internal");
  tex->SetNDC();
  tex->SetTextFont(72);
  tex->SetLineWidth(2);
  tex->Draw();
}

void ATLAS_LABEL_PRELIMINARY(Double_t x,Double_t y,Color_t color)
{
  TLatex *   tex = new TLatex(x,y,"ATLAS Simulation Preliminary");
  tex->SetNDC();
  tex->SetTextFont(72);
  tex->SetLineWidth(2);
  tex->Draw();
}

void SetCanvasAttr(TCanvas *tc)
{
  tc->Range(-28,-70.10724,149.1636,56.43432);
  tc->SetFillColor(0);
  tc->SetBorderMode(0);
  tc->SetBorderSize(2);
  //  tc->SetGridy();
  tc->SetTickx(1);
  tc->SetTicky(1);
  tc->SetLeftMargin(0.158046);
  tc->SetRightMargin(0.05172414);
  tc->SetTopMargin(0.05084746);
  tc->SetBottomMargin(0.1588983);
  tc->SetFrameBorderMode(0);
  tc->SetFrameBorderMode(0); 
}


void myEvent_display::Loop()
{
    
    if (fChain == 0) return;

    Long64_t nentries = fChain->GetEntriesFast();

    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
       Long64_t ientry = LoadTree(jentry);
       if (ientry < 0) break;
       nb = fChain->GetEntry(jentry);   nbytes += nb;
       // if (Cut(ientry) < 0) continue;
    }
 }

  

float myEvent_display::ComputeVS0ForVertex(int INDEX)
{
  float result = 0.0;
  TRandom* HGTD = new TRandom;

  //loop over tracks in vertex
  //for(int i=0; i<vtxtrkpt->at(INDEX).size(); i++) {
  cout<< "number of tracks: " << recovertex_tracks_idx->at(INDEX).size()<<endl;
  for(int i = 0; i< recovertex_tracks_idx->at(INDEX).size(); i++) {
            //selection cuts
                  
            int V_idex = recovertex_tracks_idx->at(INDEX)[i];
            float p = abs(1/(track_qOverP->at(V_idex)));
            float eta = - log(tan((track_theta->at(V_idex))/2));
            float track_pT = p/(cosh(eta));
            float pt = track_pT/1000;
            float pt2 = abs(1e-3/(track_qOverP->at(V_idex)))* sin(track_theta->at(V_idex));
            float phi = track_phi->at(V_idex);
            float z = track_z0->at(V_idex) - recovertex_z->at(INDEX); //vtxtrkz0 changed to track_z0
            //float s = z/track_var_z0->at(V_idex);
            //int vertexID = truthvertex_tracks_idx->at(INDEX)[i];
            //float w  =  recovertex_tracks_weight->at(INDEX)[i];
            float Dz0para = getDzpara(fabs(eta), pt, false);
            //float time = truthvertex_t->at(vertexID);
         
    //selection cuts
    
    if(pt<0.9 || pt>45.0) continue;
    if(abs(eta)>3.8) continue;
    if(abs(z)>Dz0para) continue;
    
    result  += pt*pt;
  }
  return result;
}

//-----------------------------------------------
void myEvent_display::DisplayZRho(int event, int vtxID, bool drawHGTD, int TrkSelection, bool NEWSEL)
//-----------------------------------------------
{
  //ADD tracks from nearby vertices in grey
  //Add colors for timing (as option to turn on/off)
  //Look at events where PV selection fails

  
  float TIME = 30.0;
  
  cout << "-------------------------------" << endl;
  cout << " Event Display for event# " << event << endl;
  cout << " vtxID = " << vtxID << endl;
  cout << "-------------------------------" << endl << endl;
  
  TRandom* HGTD = new TRandom;

  if(event>=fChain->GetEntriesFast()) return;
  Long64_t ientry = LoadTree(event);
  if (ientry < 0) return;
  fChain->GetEntry(event);

  int vertexIndex = GetMatchedVertex();

  cout << "Truth vertex = " << truthvertex_z->at(0) << endl;
  cout << "Default  reco vertex (0) = " << recovertex_z->at(0) << endl;

  cout << "Sum pt for vertex (" << vtxID << ") = " << endl;
  float sumpt = ComputeVS0ForVertex(vtxID);
  cout << "Sum pt for vertex (" << vtxID << ") = " << sumpt << endl;

  //vertex z
  float z = recovertex_z->at(vtxID);
  cout << endl << "Plotting Vertex reco z = " << z << ", sumpt = " << sumpt << endl;
  
  float Distance = abs(truthvertex_z->at(0)-recovertex_z->at(vtxID));
  cout << endl;
  if(Distance<0.5) cout << " GOOD PV";
  else cout << "Wrong PV";
  cout << endl;
    
  
  

  //print reco vertices close to the HS truth vertex
  for(int i=0; i< recovertex_z->size(); i++) {
    //cout << "Reco vertex#" << i << ", z = " << recovertex_z->at(i) << endl;
    if(fabs(recovertex_z->at(i)-truthvertex_z->at(0))<5) {
      //cout << "Reco vertex " << i << ", z = " << recovertex_z->at(i) << ", @ " << fabs(recovertex_z->at(i)-truthvertex_z->at(0)) << "mm" << endl;
    }
  }
  
  //set scale
  float X1 = z - 5.0;
  float X2 = z + 5.0;

  
  //--------------------------
  gStyle->SetPalette(1);
  TCanvas *c5 = (TCanvas*)gROOT->FindObject("c5");
  if (c5) {c5->Clear();}
  else {
    c5 = new TCanvas("c5","",1000,500);
    c5->SetFillColor(0);
  }
  gStyle->SetOptStat(0);
  c5->SetLeftMargin(0.15);
  c5->SetRightMargin(0.15);
  c5->SetLogy(0);
  //--------------------------

  TH2F* detector = new TH2F("detector","", 100,X1,X2, 100, -1,1);
  detector->SetXTitle("z [mm]");
  detector->SetYTitle("R [mm]");
  detector->Draw();

  if(drawHGTD) {

    for(int i=0; i<100; i++) {
      float eta = 2.4+i*(1.4/100);
      
      float Theta = 2*atan(exp(-eta));
      float  y = 5*tan(Theta);
      float sign = 1.0;
      if(eta<0) sign = -1;
      TLine* l = new TLine(z,0,(z+5)*sign,y);
      l->SetLineStyle(1);
      l->SetLineColor(5);
      l->Draw();
      TLine* ll = new TLine(z,0,(z+5)*sign,-y);
      ll->SetLineStyle(1);
      ll->SetLineColor(5);
      ll->Draw();
      
    }
    for(int i=0; i<100; i++) {
      float eta = -3.8+i*(1.4/100);
      float Theta = 2*atan(exp(-eta));
      float  y = 5*tan(Theta);
      float sign = 1.0;
      if(eta<0) sign = -1;
      TLine* l = new TLine(z,0,(5-z)*sign,y);
      l->SetLineStyle(1);
      l->SetLineColor(5);
      l->Draw();
      TLine* ll = new TLine(z,0,(5-z)*sign,-y);
      ll->SetLineStyle(1);
    ll->SetLineColor(5);
    ll->Draw();
    }
  }

  float Rpt;
  float Rpt_time;

  TRandom* rand = new TRandom();
  float trackPT = 0;
  float trackPT_3 =0;
  
  //Draw jets
  //-------------------------------
  cout << endl << "Jets: " << endl;
  int n = jet_pt->size();
  for(int i=0; i<n; i++) {

    if(jet_pt->at(i)<40) continue;
    //if(jettype->at(i)!=1) continue;
    
    float eta = jet_eta->at(i);
    float phi = jet_phi->at(i);
    float pt = jet_pt->at(i);
      
    //float px = pt*cos(phi);
    float pz = pt*sinh(eta);
    //float rho = pt*sin(phi)/abs(sin(phi));

    float signX = eta/abs(eta);
    float signY = sin(phi)/abs(sin(phi));
    
    float theta = atan(pt/abs(pz));
    float x = (pt/40)*cos(theta)*signX;
    float y = (pt/40)*sin(theta)*signY;

    cout << "  Pt, Eta, Phi  = " << jet_pt->at(i) << ",   "
     << jet_eta->at(i) <<  ", " << jet_phi->at(i) << ", HS = " << jet_isHS->at(i) << endl;
      
      TLine* ll = new TLine(z,0,z+x,y);
      if(jet_isHS->at(i)==1) {
      //    TLine* ll = new TLine(z,0,z+x,y);
          ll->SetLineStyle(1);
          ll->SetLineWidth(25);
          ll->SetLineColorAlpha(2,0.1);
      }
 //   if(jet_isHS->at(i)==1) ll->SetLineColorAlpha(2,0.1);
    else ll->SetLineColorAlpha(4,0.1);
    
    ll->Draw();
  }
  

  float leading_track_time = truthvertex_t->at(0);

 
  
  
  //plot HS vertex
  //-------------------------------
  float MET = 0;
  float met_x = 0, met_y = 0;

  //for(int i=0; i<vtxtrkpt->at(vtxID).size(); i++) {
  for(int i = 0; i< recovertex_tracks_idx->at(vtxID).size(); i++) {
    //selection cuts
    int V_idex = recovertex_tracks_idx->at(vtxID)[i];
    float p = abs(1/(track_qOverP->at(V_idex)));
    float eta = - log(tan((track_theta->at(V_idex))/2));
    float track_pT = p/(cosh(eta));
    float pt = track_pT/1000;
    float pt2 = abs(1e-3/(track_qOverP->at(V_idex)))* sin(track_theta->at(V_idex));
    float phi = track_phi->at(V_idex);
    float z0 = track_z0->at(V_idex) - recovertex_z->at(vtxID); //vtxtrkz0 changed to track_z0
    //float s = z/track_var_z0->at(V_idex);
    //int vertexID = truthvertex_tracks_idx->at(vtxID)[i];
    //int vertexID = recovertex_tracks_idx->at(vtxID)[i];
    int vertexID =  recovertex_isPU->at(vtxID);
    //float w  =  recovertex_tracks_weight->at(vtxID)[i];
    float Dz0para = getDzpara(fabs(eta), pt, false);


    //float deta = jet_eta->at(JETID) - eta;
    //float dphi = jet_phi->at(JETID) - track_phi->at(V_idex);
    //if(dphi>TMath::Pi()) dphi -= 2*TMath::Pi();
    //float Dr = sqrt(deta*deta + dphi*dphi);
      
    float vertex_time_truth = truthvertex_t->at(0);
    float vertex_time_reco = rand->Gaus(vertex_time_truth, 10.0);
    float time_cut_reco = (track_t30->at(V_idex)-vertex_time_reco)/30.0;
      

    if(pt<1.0 || pt>45.) continue;
    if(abs(eta)>4.0) continue;

    //if (((track_z0->at(V_idex)-recovertex_z->at(vtxID))/sqrt(track_var_z0->at(V_idex)))>2.0) continue;
    //if(Dr>0.2) continue;

    //trackPT += pt;
    if(abs(time_cut_reco)<2.5) trackPT_3 += pt;
    //if(abs(time_cut_reco)>2.5) continue; // Removed time cut
    trackPT += pt;
      

    if(TrkSelection == 0) if(abs(z0)>Dz0para) continue;

    float px = pt*cos(phi);
    float py = pt*sin(phi);
    met_x += px;
    met_y += py;

    //ARIEL

    //jet selection
    //---------------------
    bool pass = false;

    if(abs(eta)>2.4) {
      float Rmin = 0.5;
      float PT;
      for(int u=0; u<jet_pt->size(); u++) {
    if(jet_pt->at(u)>30) {
      //float jet_eta = jet_eta->at(u);
      //float jet_phi = jet_phi->at(u);
      
      float deta = eta - jet_eta->at(u);
      float dphi = phi - jet_phi->at(u);
      if(dphi>TMath::Pi()) dphi -= 2*TMath::Pi();
      float DR = sqrt(deta*deta+dphi*dphi);
      if(DR<Rmin) {
        pass = true;
      }
    }
      } //loop over jets
    } else {
      pass = true;
    }
    if(NEWSEL && !pass) continue;
    //ARIEL


    float pz = pt*sinh(eta);
    float signX = eta/abs(eta);
    float signY = sin(phi)/abs(sin(phi));
    float theta = atan(pt/abs(pz));
    float x = (pt/2)*cos(theta)*signX;
    float y = (pt/2)*sin(theta)*signY;

    
    
    TLine* l;
    //l = new TLine(z0+z,0, (z0+z+pz/3.),rho/3.);
    l = new TLine(z0+z,0, z0+z+x,y);
    if(vertexID==0 && track_status->at(V_idex)==0) l->SetLineColor(4); //chnged color
    //if(track_status->at(i)==1) l->SetLineColor(3);
   // if(vertexID==vtxID && track_status->at(i)==0) l->SetLineColor(2);
    else l->SetLineColor(2);
    //if(pt>15.0) l->SetLineWidth(3);
    l->Draw("same");

    //if(vertexID==0 && recovertex_tracks_idx->at(vtxID)[i]!=vtxID) {
 //   if(vertexID!=0) {
//      l->SetLineColor(3);
  //  }
      
      
     //added this 28th match
   
  }

  MET = sqrt(met_x*met_x+met_y*met_y);
  cout << "Vertex MET = " << MET << endl;
  

 
  //draw other RECO vertices
  //-------------------------------
  for(int i=0; i< recovertex_z->size(); i++) {
    if(fabs(recovertex_z->at(i)-z)<5) {
      TEllipse* r = new TEllipse(recovertex_z->at(i),0,0.1,0.1/5);
      //cout << "Vertex z = " << recovertex_z->at(i) << endl;
      r->SetFillColor(1);
      r->SetLineColor(1);
      if(i==vtxID) r->SetFillColor(2);
      r->Draw();
    }
  }


  //draw TRUTH vertices
  //-------------------------------
  for(int i=truthvertex_z->size()-1; i>=0; i--) {
    if(fabs(truthvertex_z->at(i)-z)<5) {
      TBox* r = new TBox(truthvertex_z->at(i)-0.03,-0.75, truthvertex_z->at(i)+0.03,-0.85);
      //cout << "Truth Vertex z = " << truthvertex_z->at(i) << endl;
      r->SetFillColor(1);
      if(i==0) r->SetFillColor(2);
      r->Draw();
    }
  }

   std::vector<float> uniqueZValues;
        


      //draw tracks in nearest HS vertex
      //-------------------------------
      for(int j=0; j< recovertex_z->size(); j++) {
        //if(j == vtxID) continue; // Skip iteration when j is equal to vtxID
          
        //cout << " j : " << j << endl;
        if (recovertex_isHS->at(j) == 1) {
            
          float Z = recovertex_z->at(j);
          for(int i = 0; i< recovertex_tracks_idx->at(j).size(); i++) {

            int V_idex = recovertex_tracks_idx->at(j)[i];

            float p = abs(1/(track_qOverP->at(V_idex)));
            float eta = - log(tan((track_theta->at(V_idex))/2));
            float track_pT = p/(cosh(eta));
            float pt = track_pT/1000;
            float pt2 = abs(1e-3/(track_qOverP->at(V_idex)))* sin(track_theta->at(V_idex));
            float phi = track_phi->at(V_idex);
            float z0 = track_z0->at(V_idex) - recovertex_z->at(0); //vtxtrkz0 changed to track_z0
            //float s = z/track_var_z0->at(V_idex);
            //int vertexID = truthvertex_tracks_idx->at(j)[i];
            int vertexID = recovertex_tracks_idx->at(j)[i];
            //float w  =  recovertex_tracks_weight->at(j)[i];
            float Dz0para = getDzpara(fabs(eta), pt, false);
            //float time = truthvertex_t->at(vertexID);
            //float time_trk = HGTD->Gaus(time,TIME);
                
            //if(pt<0.9 || pt>45.0) continue;
            if(pt<1.0) continue;
            //if(abs(eta)>3.8) continue;

            if(TrkSelection == 0) if(abs(z0)>Dz0para) continue;
                
            float pz = pt*sinh(eta);
            float signX = eta/abs(eta);
            float signY = sin(phi)/abs(sin(phi));
            float theta = atan(pt/abs(pz));
            float x = (pt/2)*cos(theta)*signX;
            float y = (pt/2)*sin(theta)*signY;
        
            TLine* l;
            l = new TLine(Z+z0,0, Z+z0+x,y);
            
            //l->Draw("same");
            

          }
            
           //Draw reco jets
            for(int i=0; i<jet_pt->size(); i++) {
               
              if(jet_pt->at(i)>40.0) {
                
                TEllipse* r = new TEllipse(jet_eta->at(i),jet_phi->at(i),0.4,0.4);
                r->SetLineStyle(1);
                    
                if(jet_isHS->at(i)==1)
              r->SetFillColorAlpha(2,0.1);
                else
              r->SetFillColorAlpha(4,0.1);
                r->Draw();
              }
            }

         
          
        }
      }
     

        
    
    
    
    
  //axis
  //-------------------------------
  TLine* u = new TLine(z-5,0,z+5,0);
  u->SetLineWidth(2);
  u->Draw("same");

  TLine* u2 = new TLine(z-5,-0.8,z+5,-0.8);
  u2->SetLineWidth(2);
  u2->Draw("same");


  char g1[40];
  sprintf(g1,"Reco z = %2.2f mm", recovertex_z->at(vtxID));
  myText(0.18,0.85,1,g1);

  char g2[40];
  sprintf(g2,"Truth z = %2.2f mm", truthvertex_z->at(0));
  myText(0.18,0.81,1,g2);

 
  char g3[40];
  sprintf(g3,"Sum p^{2}_{T} = %3.1f GeV^{2}", sumpt);
  myText(0.18,0.77,1,g3);
 

  myText(0.78,0.21,1,"Truth");
  myText(0.78,0.52,1,"Reco");

 
  {
    TH1F* ll1 = new TH1F();
    ll1->SetLineColor(4); ll1->SetMarkerStyle(0);
    TH1F* ll2 = new TH1F();
    ll2->SetLineColor(2); ll2->SetMarkerStyle(0);
    //TH1F* ll3 = new TH1F();
    //ll3->SetLineColor(3); ll2->SetMarkerStyle(3);
    TLegend* leg2 = new TLegend(0.55,0.65,0.80,0.84,NULL,"brNDC");
    leg2->SetFillColor(kWhite);
    leg2->SetBorderSize(0);
    leg2->SetTextFont(42);
    leg2->SetTextSize(0.04);
    leg2->AddEntry(ll1,"HS Tracks");
    leg2->AddEntry(ll2,"PU Tracks");

    //leg2->AddEntry(ll2,"PU tracks");
    //leg2->AddEntry(ll3,"HS tracks from PU vertx");
    leg2->Draw();
  }


  if(TrkSelection==0)
    myText(0.18, 0.30, 1, "#eta-p_{T} z_{0} cut");
  else
    myText(0.18, 0.30, 1, "Vertex fit");
  
  
  ATLAS_LABEL_PRELIMINARY(0.45,0.85,1);

   {
    char g1[40];
    sprintf(g1,"zrho_event%i_vtxid%i.pdf", event, vtxID);
    c5->Print(g1);

    char g2[40];
    sprintf(g2,"zrho_event%i_vtxid%i.eps", event, vtxID);
    c5->Print(g2);

    char g3[40];
    sprintf(g3,"zrho_event%i_vtxid%i.png", event, vtxID);
    c5->Print(g3);
  }
    

  
}


 
