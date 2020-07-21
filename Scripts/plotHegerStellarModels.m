function plotHegerStellarModels

HE16C=importdata('../stellarProfiles/HE16C');
HE16C.textdata{1}

grid = HE16C.data(:,find(strcmp(HE16C.textdata,'grid')))



end