function xhat = image_by_SBL(y, A, L)
    addpath('.\BSBL-FM');
    
    if(nargin == 2) L = 1; end

    blkStartLoc = 1 : L : size(A, 2);

    Result = BSBL_FM(A, y, blkStartLoc, 1, 'learnType', 0, 'verbose', 0);
    xhat = Result.x;
end