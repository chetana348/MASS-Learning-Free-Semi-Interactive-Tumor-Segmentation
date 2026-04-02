function [meanEstimate, distance] = computeME(gradSignal, scaleFactor, axisDim)

% Default dimension
if nargin < 3
    axisDim = 1;
end

% --- Coordinate grid along chosen dimension ---
gridShape = ones(1, ndims(gradSignal));
gridShape(axisDim) = size(gradSignal, axisDim);

coords = reshape(1:size(gradSignal, axisDim), gridShape);

% Support GPU arrays
if isa(gradSignal, 'gpuArray')
    coords = gpuArray(coords);
end

% --- Center coordinates ---
centerVal = mean(coords);
centeredCoords = coords - centerVal;

% --- Integrated signal ---
accumulated = scaleFactor * cumsum(gradSignal, axisDim);
clear gradSignal

% --- Zero-mean normalization ---
accumulated = accumulated - mean(accumulated, axisDim);

% --- Exponential transforms ---
expPos = exp(accumulated);
expNeg = exp(-accumulated);
clear accumulated

% --- Forward and reverse weighted sums ---
forwardNum  = cumsum(centeredCoords .* expPos, axisDim);
reverseNum  = cumsum(centeredCoords .* expNeg, axisDim, 'reverse');

forwardDen  = cumsum(expPos, axisDim);
reverseDen  = cumsum(expNeg, axisDim, 'reverse');

% --- Final mean estimate ---
numerator   = expNeg .* forwardNum + expPos .* reverseNum;
denominator = expNeg .* forwardDen + expPos .* reverseDen;

meanEstimate = numerator ./ denominator + centerVal;

% Handle numerical instability
meanEstimate(isinf(meanEstimate)) = nan;

% --- Optional distance output ---
if nargout > 1
    distance = abs(meanEstimate - coords);
end

end