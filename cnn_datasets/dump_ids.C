#include <TFile.h>
#include <TTree.h>
#include <fstream>
#include <iostream>

void dump_ids(const char* inroot="MPID_test_set_full.root",
              const char* treename="image2d_image2d_binary_tree",
              const char* outcsv="MPID_test_set.csv")
{
  TFile f(inroot,"READ");
  if (f.IsZombie()) { std::cerr << "Cannot open " << inroot << "\n"; return; }
  auto t = (TTree*)f.Get(treename);
  if (!t) { std::cerr << "No tree " << treename << "\n"; return; }

  // Branch types shown by your t->Print(): ULong_t
  ULong_t run=0, subrun=0, event=0;
  t->SetBranchAddress("_run",    &run);
  t->SetBranchAddress("_subrun", &subrun);
  t->SetBranchAddress("_event",  &event);

  std::ofstream csv(outcsv);
  csv << "run_number,subrun_number,event_number\n";
  Long64_t n = t->GetEntries();
  for (Long64_t i=0; i<n; ++i) {
    t->GetEntry(i);
    csv << run << "," << subrun << "," << event << "\n";
  }
  csv.close();
  std::cout << "Wrote " << outcsv << " with " << n << " rows\n";
}
