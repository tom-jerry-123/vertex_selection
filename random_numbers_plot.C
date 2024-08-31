//random_numbers_plot.C
#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom3.h>

void random_numbers_plot() {
    // Create a canvas
    TCanvas *c1 = new TCanvas("c1", "Random Numbers", 800, 600);

    // Create a histogram with 50 bins from 0 to 100
    TH1F *hist = new TH1F("hist", "Histogram of 100 Random Numbers;Value;Frequency", 50, 0, 100);

    // Create a random number generator
    TRandom3 *rand = new TRandom3(0); // Seed with 0 for random seed

    // Generate 100 random numbers and fill the histogram
    for (int i = 0; i < 100; i++) {
        double num = rand->Uniform(0, 100); // Generate a number between 0 and 100
        hist->Fill(num);
    }

    // Draw the histogram
    hist->Draw();

    // Save the canvas to a file
    c1->SaveAs("random_numbers_hist.png");

    // Clean up
    delete hist;
    delete rand;
    delete c1;
}

