function phiR = phiIFFT(phiK,ibz,FFT)

global PW

fftGrid = FFT.fftGrid;
nPoints = FFT.nPoints;

GIndex = FFT.FFTIndex(:,ibz);
filter = find(GIndex>0);

nQOrbs = size(phiK,2);
phiR = zeros(nPoints,nQOrbs);

for iorb = 1 : nQOrbs
    phiCube = zeros(nPoints,1);
    phiCube(GIndex(filter)) = phiK(filter, iorb);
    phiCube = reshape(phiCube,fftGrid);
    
    phiCube  = ifftn(phiCube) * nPoints/PW.Volume;
    phiR(:,iorb)  = reshape(phiCube,[nPoints,1]);
end



%function phiR = phiIFFT(phiK, ibz)
%    global PW FFT1 FFT2 FFT3
%
%    FFTs = {FFT1, FFT2, FFT3};
%    nQOrbs = size(phiK, 2);
%
%    split1 = size(FFT1.FFTIndex,1);
%    split2 = size(FFT2.FFTIndex,1) + size(FFT3.FFTIndex,1);
%
%    phiK1 = phiK(:,:,1:split1);
%    phiK2 = phiK(:,:,split1+1:split2);
%    phiK3 = phiK(:,:,split2+1:end);
%    phiKs = {phiK1, phiK2, phiK3};
%
%    phiR = [];
%
%    for i = 1:3
%        FFT = FFTs{i};
%        phiK_part = phiKs{i};
%
%        fftGrid = FFT.fftGrid;
%        nPoints = FFT.nPoints;
%        GIndex = FFT.FFTIndex(:, ibz);
%        filter = find(GIndex > 0);
%
%        for iorb = 1:nQOrbs
%            phiCube = zeros(prod(nPoints), 1);
%            phiCube(GIndex(filter)) = phiK_part(filter, iorb);
%            phiCube = reshape(phiCube, fftGrid);
%            phiCube = ifftn(phiCube) * prod(nPoints) / PW.Volume;
%            phiR_tmp = reshape(phiCube, [prod(nPoints), 1]);
%
%            if size(phiR, 1) < length(phiR_tmp)
%                phiR(end+1:length(phiR_tmp), iorb) = 0;
%            end
%            phiR(1:length(phiR_tmp), iorb) = phiR(1:length(phiR_tmp), iorb) + phiR_tmp;
%        end
%    end
%end