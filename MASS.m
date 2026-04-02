function [L, distME] = MASS(I, param, scribble)
D=2;
if ischar(I)
    load(I, 'I')
end
if ~exist('param', 'var')
    param = [];
end
existL = exist('L', 'var');
if ~isfield(param, 'dim')
    if ndims(I)==4
        param.dim = 3;
    else
        param.dim = ndims(I);
    end
end

if ~isfield(param, 'power')
    param.power = 2;
end
if ~isfield(param, 'alpha')
    param.alpha = 2000 / (max(I(:))-min(I(:)))^param.power;
end
if ~isfield(param, 'maxIter')
    param.maxIter = 1000;
    
end
if ~isfield(param, 'randIterNum')
    param.randIterNum = 100;
end
if ~isfield(param, 'angleStep')
    param.angleStep = 1;
end
if ~isfield(param, 'indexGPU')
    param.indexGPU = false;
end


if ~isfield(param, 'crop')
    param.crop = true; % Keep 'true' for faster performance.
end

s0 = size(I);
s0 = s0(1:D);
nChannels = size(I, D+1);

mSize = max(s0);

% --- FG / BG scribble indices ---
if exist('scribble','var') && ~isempty(scribble)

    fgIdx = find(scribble == 1);   % tumor scribble
    bgIdx = find(scribble == 2);   % background scribble
    % --- build hard background exclusion mask ---
    bgMask = false(size(I));
    
    if ~isempty(bgIdx)
    
        bgMask(bgIdx) = true;
    
    end

else

    fgIdx = [];
    bgIdx = [];

end

% --- learn intensity model from scribbles ---
if ~isempty(fgIdx) && ~isempty(bgIdx)

    fgVals = I(fgIdx);
    bgVals = I(bgIdx);

    muFG = mean(fgVals);
    muBG = mean(bgVals);

    sigmaFG = std(fgVals) + 1e-6;
    sigmaBG = std(bgVals) + 1e-6;
    contrast = abs(muFG - muBG) / (sigmaFG + sigmaBG);

    pFG = exp(-(I-muFG).^2 ./ (2*sigmaFG^2));
    pBG = exp(-(I-muBG).^2 ./ (2*sigmaBG^2));

    tumorPrior = pFG ./ (pFG + pBG + eps);

else

    tumorPrior = ones(size(I));

end
outsideValue = I;
for d = 1:D
    outsideValue = max(outsideValue, [], d);
end
outsideValue = 2 * outsideValue * param.crop;
cI = bsxfun(@times, ones(mSize*ones(1,D)), outsideValue);
cropX = arrayfun(@(siz) round((mSize-siz)/2) + (1:siz), s0', 'UniformOutput', false);
cI(cropX{:},:) = I;
angles = 0:param.angleStep:179;
sBox = single(size(cI));
sBox = sBox(1:D);
if param.crop
    s = s0;
else
    s = sBox;
end
sA = size(angles,2);
padZeros = zeros([1 double(sBox(2:end))]);
X = cell(D,1);
cellOne2S = arrayfun(@(siz) 1:siz, sBox, 'UniformOutput', false);
[X{:}] = ndgrid(cellOne2S{:});
X = cellfun(@uint16, X, 'UniformOutput', false);
if param.indexGPU
    gpu = gpuDevice(param.indexGPU);
    if param.verbose
        disp(['Using the "' gpu.Name '" GPU...'])
    end
    X = cellfun(@gpuArray, X, 'UniformOutput', false);
else
    gpu = [];
end
aX = zeros([sA s D], 'uint16');
aD = zeros([sA s]  , 'uint16');

pool = [];
if isempty(pool)
    nCores = 0;
else
    nCores = pool.NumWorkers;
end
cI = bsxfun(@minus, cI, outsideValue);
% apply tumor prior weighting
tumorPrior = imresize(tumorPrior, size(cI(:,:,1)));

if isfield(param,'lambda')
    lambda = param.lambda;
else
    lambda = 0.68;
end
cI = cI .* (1 + lambda*(1 - tumorPrior));
tic
for a = 1:sA  % To use the Parallel Computing Toolbox, change to:   parfor (a = 1:sA, nCores)
    if param.verbose
        disp(['Direction # ' num2str(a) ' / ' num2str(sA)])
    end
    Irot_D = bsxfun(@plus, imrotate(cI, angles(a), 'bilinear', 'crop'), outsideValue);
    if param.indexGPU
        Irot_D = gpuArray(Irot_D);
    end
    if nChannels==1
        Irot_D = [abs(diff(Irot_D,1,1)).^param.power; padZeros];
    else
        Irot_D = [mean(abs(diff(Irot_D,1,1)).^param.power, D+1); padZeros];
    end
    [C, Irot_D] = computeME(Irot_D, param.alpha); % Irot_D is now the distance from the CM.
    C = round(single(C));
    C(C<1 | C>mSize) = nan;
    Irot_D = gather(uint16(round(imrotate(single(Irot_D), -angles(a), 'bilinear', 'crop'))));
    if param.crop
        aD(a,:,:) = Irot_D(cropX{:});
    else
        aD(a,:,:) = Irot_D;
    end
    nanC = isnan(C(:)) | isnan(X{2}(:));
    C(nanC) = 1;
    C = sub2ind(sBox, C(:), X{2}(:)); % C is now the CM indices.
    C(nanC) = 1;
    for d = 1:D
        Xrot = imrotate(X{d}, angles(a), 'nearest', 'crop');
        Xrot = Xrot(C);
        Xrot(nanC | isnan(Xrot)) = 0;
        Xrot = gather(imrotate(reshape(Xrot, sBox), -angles(a), 'nearest', 'crop'));
        if param.crop
            aX(a,:,:,d) = Xrot(cropX{:}) - cropX{d}(1) + 1;
        else
            aX(a,:,:,d) = Xrot;
        end
    end
end
t = toc;
mask = ~any(isnan(aX), D+2) & all(aX>=1, D+2) & all(bsxfun(@le, aX, permute(s, [3:(2+D), 1 2])), D+2);

% --- HARD BG BARRIER ---
if ~isempty(bgIdx)

    bgMask = false(s);
    bgMask(bgIdx) = true;

    for a = 1:size(mask,1)

        tmp = squeeze(mask(a,:,:));

        xPos = squeeze(aX(a,:,:,1));
        yPos = squeeze(aX(a,:,:,2));
        
        xPos = round(xPos);
        yPos = round(yPos);
        
        xPos(xPos < 1) = 1;
        yPos(yPos < 1) = 1;
        
        xPos(xPos > s(1)) = s(1);
        yPos(yPos > s(2)) = s(2);

        ind = sub2ind(s, xPos(:), yPos(:));
        ind = reshape(ind,s);

        tmp(bgMask) = false;
        tmp(ind(bgMask)) = false;

        mask(a,:,:) = tmp;

    end
end
aD = single(aD);
distBG = bwdist(bgMask);
mask = mask & (aD > 0);

if nargout > 1
    %aD(~mask) = nan;
    if ~existL
        L = [];
    end
    distME = squeeze(nanmean(aD,1));
    return
end
delete(pool)
clear aD
clear cI

aX = num2cell(aX, 1:(D+1));
aX = cellfun(@(c) c(mask(:)), aX, 'UniformOutput', false);
indL = uint32(sub2ind(s, aX{:}));
clear aX
nMask = single(squeeze(sum(mask,1)));
indM = nMask(:)>0;
if param.crop
    X = cellfun(@(c, cr) c(cropX{:}) - cr(1) + 1, X, cropX, 'UniformOutput', false);
    cropX = [];
end
X = cellfun(@(c) c(indM), X, 'UniformOutput', false);
nMask = nMask(indM); sNMask = length(nMask);
mask = false([sA s]);
mask(sub2ind([sA s], nMask, X{:})) = true;
mask = cumsum(mask, 1, 'reverse')>0;

nIter = 1;
if existL
    L = single(L);
else
    % --- initialize random labels ---
    L = randi([3 50], s, 'single');
    
    % enforce scribble labels
    if ~isempty(fgIdx)
        L(fgIdx) = 1;
    end
    
    if ~isempty(bgIdx)
        L(bgIdx) = 2;
       
    end
end
r = setdiff(1:prod(s), unique(L(:)));
L(isnan(L(:))) = r(randperm(nnz(isnan(L(:)))));
clear r
if param.indexGPU
    s = gpuArray(s);
    sA = gpuArray(sA);
    mask = gpuArray(mask);
    nMask = gpuArray(nMask);
    indL = gpuArray(indL);
    indM = gpuArray(indM);
    L = gpuArray(L);
end
maskSize = find(~any(any(any(mask,2),3),4), 1, 'first');
if isempty(maskSize)
    sT = [sA s];
else
    sT = [maskSize-1, s];
    mask = mask(1:sT(1),:,:,:);
end
keepWhile = true;
while keepWhile && nIter<=param.maxIter
    verbPhrase = ['Iteration # ' num2str(nIter) ' out of ' num2str(param.maxIter) '.'];
    if param.verbose
        disp(verbPhrase)
    end
    L0 = L;
    aL = nan(sT, 'like', L);
    aL(mask(:)) = L(indL);

    if nIter > param.randIterNum

        % standard voting
        L = squeeze(mode(aL,1));

    else
        L(indM) = aL(sub2ind(sT, round(rand(sNMask, 1, 'like', s) .* (nMask-1)) + 1, X{:}));
        if nIter == param.randIterNum
        end
    end
    nanL = isnan(L);
    
    L(nanL) = L0(nanL);

    % --- enforce scribble constraints ---
    if ~isempty(fgIdx)
        L(fgIdx) = 1;
    end
    
    if ~isempty(bgIdx)
        L(bgIdx) = 2;
    end

    if isequaln(L,L0)
        keepWhile = false;
    end
    nIter = nIter + 1;
end

%% Extract tumor cluster using FG/BG constraints

if ~isempty(fgIdx)

    tumorLabel = mode(L(fgIdx));

    BW = (L == tumorLabel);

    % keep only component touching FG scribble
    CC = bwconncomp(BW);
    
    keep = false(CC.NumObjects,1);
    disp(['Num components: ' num2str(CC.NumObjects)])
    disp(['FG pixels in BW: ' num2str(sum(BW(fgIdx)))])
    
    for k = 1:CC.NumObjects
        if any(ismember(CC.PixelIdxList{k}, fgIdx))
            keep(k) = true;
        end
    end
    
    BW = false(size(BW));
    for k = find(keep)'
        BW(CC.PixelIdxList{k}) = true;
    end
    disp(['Pixels after component filtering: ' num2str(nnz(BW))])
    
    
    BW = imfill(BW,'holes');
    
    disp(['Pixels after imfill: ' num2str(nnz(BW))])
    % --- estimate tumor diameter from scribble ---
    % scribble coordinates
    [y,x] = ind2sub(size(BW), fgIdx);
    
    n = numel(x);
    scribble_length = 0;
    
    for i = 1:n
        dx = x(i) - x;
        dy = y(i) - y;
        d = sqrt(dx.^2 + dy.^2);
        scribble_length = max(scribble_length, max(d));
    end

    % estimated tumor radius
    R = scribble_length / 2;
    
    % center of scribble
    cx = mean(x);
    cy = mean(y);
    
    [X,Y] = meshgrid(1:size(BW,2),1:size(BW,1));
    
    dist = sqrt((X-cx).^2 + (Y-cy).^2);
    
    % restrict CM mask
    BW = BW & (dist <= R);
    % --- ICC-oriented boundary stabilization (remove weak boundary pixels) ---

    % boundary pixels
    B = bwperim(BW);
    
    % count tumor neighbors (8-neighborhood)
    N = conv2(double(BW), ones(3), 'same');
    
    % keep boundary pixels that have strong support from neighbors
    keepBoundary = B & (N >= 5);
    
    % reconstruct mask: keep interior + supported boundary
    BW = (BW & ~B) | keepBoundary;


    %BW = bwareaopen(BW, 50);
    
    %BW = imopen(BW, strel('disk',2));

    L = BW;

end

