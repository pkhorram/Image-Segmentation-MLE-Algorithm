clear all
clc
close all

%% Importing the Data

load 'TrainingSamplesDCT_8_new.mat'
load 'Zig-Zag Pattern.txt'

%% Measuring the Prior Probabilities and Displaying on the Histogram

X = categorical({'Background','Foreground'});
X = reordercats(X,{'Background','Foreground'});
Y = [1053/(1053+250),250/(1053+250)];
bar(X,Y)
title("Histogram Estimate for Priors of Background and Foreground")
grid on 
box on

py_G = 1053/(1053+250);
py_F = 250/(1053+250);
%% Measurement of MEAN, STANDARD DEVIATION, and Probability distribution for every one of the 64 features.

% Background 
Bmean =  sum(TrainsampleDCT_BG)/length(TrainsampleDCT_BG);
Bstd = sum((TrainsampleDCT_BG - Bmean).^2)/length(TrainsampleDCT_BG);
% Foreground
Fmean =  sum(TrainsampleDCT_FG)/length(TrainsampleDCT_FG);
Fstd = sum((TrainsampleDCT_FG - Fmean).^2)/length(TrainsampleDCT_FG);

% Plot of distribution of each feature for 1053 training data of the
% background.
% 8 most important features

a = [1,2,3,4,5,56,57,64];
figure(2)
for i = 1:length(a)
   
    x = TrainsampleDCT_BG(:,a(i));
    y = (1/(Bstd(a(i))*sqrt(2*pi)))*exp(-0.5*((x-Bmean(a(i))).^2)/Bstd(a(i)));
    x1 = TrainsampleDCT_FG(:,a(i));
    y1 = (1/(Fstd(a(i))*sqrt(2*pi)))*exp(-0.5*((x1-Fmean(a(i))).^2)/Fstd(a(i)));
    
    subplot(4,2,i), scatter(x,y), hold on, scatter(x1,y1,'.r'),grid on, legend('Background','Foreground'),...
        title(['Conditional Probability Distribution of Background and Foreground for Feature ' , num2str(a(i))]);



end

% 8 Least important features     

a = [63,62,60,59,8,54,37,7];
figure(3)
for i = 1:length(a)
   
    x = TrainsampleDCT_BG(:,a(i));
    y = (1/(Bstd(a(i))*sqrt(2*pi)))*exp(-0.5*((x-Bmean(a(i))).^2)/Bstd(a(i)));
    x1 = TrainsampleDCT_FG(:,a(i));
    y1 = (1/(Fstd(a(i))*sqrt(2*pi)))*exp(-0.5*((x1-Fmean(a(i))).^2)/Fstd(a(i)));
    
    subplot(4,2,i), scatter(x,y), hold on, scatter(x1,y1,'.r'),grid on, legend('Background','Foreground'),...
        title(['Conditional Probability Distribution of Background and Foreground for Feature ' , num2str(a(i))]);



end


%% Classifying the image based on the 8 important feature selected for BACKGROUND and FOREGROUND 


% First find the mean and covariance of the training set for the 8 features
% for background

% From maximum likelihood estimation use the formulas to find the
% covariance matrix and mean vector for the multivariate gaussian
% distribution (for 8 feature)

% For Background
data_background_8 = TrainsampleDCT_BG(:,[1,2,3,4,5,56,57,18]);
meanG = sum(data_background_8)/length(data_background_8);
covG =((data_background_8 - meanG)'*(data_background_8 - meanG))/length(data_background_8);

% For Foreground
data_foreground_8 = TrainsampleDCT_FG(:,[1,2,3,4,5,56,57,18]);
meanF = sum(data_foreground_8)/length(data_foreground_8);
covF =((data_foreground_8 - meanF)'*(data_foreground_8 - meanF))/length(data_foreground_8);

%% Testing the image segments 
% Load the data
image = imread('cheetah.bmp');
image = im2double(image);
image2 = padarray(image,[4 4],0,'both');

%% Preparing the data for testing.

rows = size(image,1);
columns = size(image,2);
pad = 4;
batch_size = 8;

A = zeros(rows,columns);
for r = 5 : 5+rows -1
    col = 5;
    while col <= columns
        block = image2([(r-pad):((r-pad) + batch_size - 1)],[(col-pad):((col-pad) + batch_size -1)]);
        vec = dct2(block);
        new_vec(Zig_Zag_Pattern(:)+1) = vec(:);

        back = new_vec(:,[1,2,3,4,5,56,57,18]);
        conditional_bg = (1/sqrt((2*pi)^8*det(covG)))*exp(-0.5*(back-meanG)*inv(covG)*(back-meanG)');
        Gprob = py_G*conditional_bg;
        fore = new_vec(:,[1,2,3,4,5,56,57,18]);
        conditional_fg = (1/sqrt((2*pi)^8*det(covF)))*exp(-0.5*(fore-meanF)*inv(covF)*(fore-meanF)');
        Fprob = py_F*conditional_fg;

        if(Fprob>Gprob)
           A(r-pad, col-pad) = 255;
        end

        col = col + 1;
    end



end

figure(4)
imagesc(A)
colormap(gray(255))

%% CALCULATING THE PROBABILITIES OF ERROR

figure(3)
compare = imread('cheetah_mask.bmp');
imagesc(compare)
colormap(gray(255))
total_param = size(compare,1)*size(compare,2);
P_error = ((total_param - sum(sum( A == compare)))/total_param)* 100;

%% Classifying the image based on all features for BACKGROUND and FOREGROUND


% for back ground
b = TrainsampleDCT_BG;
meanG = sum(b)/length(b);
covG =((b - meanG)'*(b - meanG))/length(b);

% For foreground
f = TrainsampleDCT_FG;
meanF = sum(f)/length(f);
covF =((f - meanF)'*(f - meanF))/length(f);

% Load the data
image = imread('cheetah.bmp');
image = im2double(image);
image2 = padarray(image,[4 4],0,'both');

rows = size(image,1);
columns = size(image,2);
pad = 4;
batch_size = 8;

A = zeros(rows,columns);
for r = 5 : 5+rows -1
    col = 5;
    while col <= columns 
        block = image2([(r-pad):((r-pad) + batch_size - 1)],[(col-pad):((col-pad) + batch_size -1)]);
        vec = dct2(block);
        new_vec(Zig_Zag_Pattern(:)+1) = vec(:); 
        
        conditional_bg = (1/sqrt((2*pi)^8*det(covG)))*exp(-0.5*(new_vec-meanG)*inv(covG)*(new_vec-meanG)');
        Gprob = py_G*conditional_bg; 
        conditional_fg = (1/sqrt((2*pi)^8*det(covF)))*exp(-0.5*(new_vec-meanF)*inv(covF)*(new_vec-meanF)');
        Fprob = py_F*conditional_fg;
        
        if(Fprob>Gprob)
           A(r-pad, col-pad) = 255;
        end
        
        col = col + 1;    
    end

    

end

figure(2)
imagesc(A)
colormap(gray(255)) 
% 
%% CALCULATING THE PROBABILITIES OF ERROR
figure(3)
compare = imread('cheetah_mask.bmp'); 
imagesc(compare)
colormap(gray(255))
total_param = size(compare,1)*size(compare,2);
P_error = ((total_param - sum(sum( A == compare)))/total_param)* 100;




