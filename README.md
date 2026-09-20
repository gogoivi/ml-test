Why is the GAN architecture so tricky - how do I make it so the discriminator model dpoesnt outpreform the generator by so much

For nn-unet just like how it uses skip connections to pass in the spacial data could I do the same thing with geometric data (through another ml model)?


=== Class Frequencies (All Samples) ===
  Background: 98.9981%
  Nerve: 0.4782%
  Vessel: 0.5237%
Loaded 296 total samples
Train samples: 236, Test samples: 60
 33%|█████████████████████████████████████████████████████████████████████████▎                                                                                                                                                    | 33/100 [06:48<13:39, 12.23s/it]Stopping at Epoch 33
 33%|█████████████████████████████████████████████████████████████████████████▎                                                                                                                                                    | 33/100 [07:02<14:16, 12.79s/it]

=== Sample 0 ===
  Background: Precision=0.998, Recall=0.995, Predicted=108953, Actual=109293
  Nerve: Precision=0.778, Recall=0.876, Predicted=1064, Actual=945
  Vessel: Precision=0.397, Recall=0.644, Predicted=575, Actual=354

=== Sample 1 ===
  Background: Precision=0.999, Recall=0.988, Predicted=108158, Actual=109406
  Nerve: Precision=0.625, Recall=0.969, Predicted=937, Actual=605
  Vessel: Precision=0.324, Recall=0.835, Predicted=1497, Actual=581

=== Sample 2 ===
  Background: Precision=1.000, Recall=0.983, Predicted=107852, Actual=109736
  Nerve: Precision=0.428, Recall=1.000, Predicted=926, Actual=396
  Vessel: Precision=0.244, Recall=0.961, Predicted=1814, Actual=460

=== Sample 3 ===
  Background: Precision=1.000, Recall=0.992, Predicted=109073, Actual=109893
  Nerve: Precision=0.620, Recall=0.939, Predicted=695, Actual=459
  Vessel: Precision=0.210, Recall=0.721, Predicted=824, Actual=240

=== Sample 4 ===
  Background: Precision=0.999, Recall=0.994, Predicted=109287, Actual=109847
  Nerve: Precision=0.749, Recall=0.893, Predicted=692, Actual=580
  Vessel: Precision=0.223, Recall=0.830, Predicted=613, Actual=165

=== Sample 5 ===
  Background: Precision=1.000, Recall=0.993, Predicted=109275, Actual=110016
  Nerve: Precision=0.578, Recall=0.944, Predicted=465, Actual=285
  Vessel: Precision=0.315, Recall=0.921, Predicted=852, Actual=291

Val: Wall time,Step,Value
1789861300.582418,0,0.16756394505500793
1789861337.9205418,3,0.08653700351715088
1789861375.2377844,6,0.09410107135772705
1789861412.2963927,9,0.057856835424900055
1789861449.7803009,12,0.05789848789572716
1789861486.8181293,15,0.05550352483987808
1789861524.0518224,18,0.05669132247567177
1789861561.2410214,21,0.06052986532449722
1789861597.8899715,24,0.06275112181901932
1789861634.2953582,27,0.05611690506339073
1789861671.8188243,30,0.06206132844090462
1789861708.831423,33,0.07140371948480606 

Train: Wall time,Step,Value
1789861300.5823817,0,0.3695516884326935
1789861337.920518,3,0.08874688297510147
1789861375.2377572,6,0.07222568988800049
1789861412.2963712,9,0.06260848790407181
1789861449.7802775,12,0.055624671280384064
1789861486.8181057,15,0.04871402308344841
1789861524.0518005,18,0.044550132006406784
1789861561.240999,21,0.04078260436654091
1789861597.8899498,24,0.035503923892974854
1789861634.2953346,27,0.03402050957083702
1789861671.8187997,30,0.0302627794444561
1789861708.8313985,33,0.027821173891425133