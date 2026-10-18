try 5 fold on 50 epochs per fold - check the results on that and then compare that to ours: train two or more models simultaneously
after epoch ∼50 or when loss stagnates (patience = counter) then add all pixels above prob vessel 95% and all prob nerve >99% (both models need to pred above this cutoff) - could maybe use similar model but with different transformations applied to them?
reset counter and keep going with this new model + vessel attion tequinique until patience reaches counter again - then stop.
-The only issue with this is that it could mess with the training data
-Another potential solution is to train a gan that attempts to distinguish between real and fake vessels and if it can then it knows that the pred is wrong - meaning that there should be a penalty for putting it there -it is removed and loss is increased? but since all these vessels will be on the outside the input to the gan will need to only be the few vessels that are pred and the corresponding background, and that is mixed in with real data from patches of random size with a semi-random amt of pixels? -
can also have a note of the labels that are pseudolabels - and if either model ever falls below like 80% certaintly that a pseuolabel is the correct class than that label is removed as a gt, and maybe lessen the loss weighting on pseudo labels both for correct and incorrect

-Entropy regularization and entropy loss?

Other option is to make boundary box around where the vessels extend to and decrease loss if the model is predicting stuff outside of there

-Also test with 5 fold 50 epoch nn-unet and increase the epochs for the relevant custom models

(Effects - addressing undersegmenting, incomplete data, not enough data etc)

-Also find a metric that is appropriate for incomplete segs