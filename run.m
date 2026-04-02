% Statistical Segmentation
% Scribble-guided version (scribbles generated externally)

clear
clc
fclose('all');
rng(0)   % Fix random seed for reproducibility

%% Paths

imgDir   = "T:\Labs\QMI\CK Data\panther\variability\uncropped\imagesTs_128";
scribDir = "T:\Labs\QMI\CK Data\panther\variability\uncropped\scribblesTs";

outDir   = "T:\Labs\QMI\CK Data\panther\variability\uncropped\test";
[status, message, messageid] = rmdir(outDir, 's');

if ~exist(outDir,'dir')
    mkdir(outDir);
end

%% Files

files = dir(fullfile(imgDir,'*.tif'));
fprintf("Found %d images\n",length(files));

%% Segmentation parameters

clear param
param.alpha = 1000;
param.randIterNum = 1500;
param.maxIter = 50;
param.verbose = false;

%% Loop

for i = 1:length(files)

    fprintf("Processing %d / %d : %s\n",i,length(files),files(i).name);

    %% Load image

    imgPath = fullfile(imgDir,files(i).name);
    I = imread(imgPath);
    %I = (I - min(I(:))) / (max(I(:)) - min(I(:))); %remove this for PDAC
    %and MSD
    I = double(I); % this comes after norm for PDAC internal
    p1  = prctile(I(:),1);
    p99 = prctile(I(:),99);
    
    I = (I - p1) / (p99 - p1);
    I(I < 0) = 0;
    I(I > 1) = 1;
    
    % convert back to double for processing
    %% Add the double sa
    %% Load scribble

    scribPath = fullfile(scribDir,files(i).name);

    if ~exist(scribPath,'file')
        warning("Missing scribble: %s",files(i).name);
        continue
    end

    scribble = imread(scribPath);
    scribble = double(scribble);
    fg = scribble == 1;
    fg = imdilate(fg, strel('disk',2));
    scribble(fg) = 1;
    

    %% Single instance
  
    L = MASS(I,param,scribble);
    BW = L; %> 0;
  
    %% Multi Instance
    %{
    CC = bwconncomp(fg);

    finalBW = false(size(fg));
    
    for k = 1:CC.NumObjects
    
        fg_instance = false(size(fg));
        fg_instance(CC.PixelIdxList{k}) = true;
    
        scribble_instance = zeros(size(scribble));
        scribble_instance(fg_instance) = 1;
        scribble_instance(scribble == 2) = 2;
    
        L = MASS(I,param,scribble_instance);
    
        BW_k = L > 0;
    
        finalBW = finalBW | BW_k;
    
    end
    
    BW = finalBW;
    %}
    outPath = fullfile(outDir,files(i).name);
    imwrite(uint8(BW),outPath);

end

fprintf("Done.\n");