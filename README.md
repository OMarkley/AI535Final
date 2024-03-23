Testing evaluation needed to be done by submitting data to Sintel through their website.
Receiving an account to do this could take up to 12 hours (we're guessing more since it's the weekend).
As a results, we're separating data from the training set to use for our testing in this project.

Training scenes moved into testing set:
- alley_2
- ambush_4
- bamboo_2
- bandage_2

They are both labeled as belonging to 'training' folders, this is becuase of how the pytorch
dataset module for the Sintel set is defined.