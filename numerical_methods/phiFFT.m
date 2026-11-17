function phiK = phiFFT(phiR,ibz,FFT)

global PW

nWaves  = PW.nWaves;
fftGrid = FFT.fftGrid;
nPoints = FFT.nPoints;

GIndex = FFT.FFTIndex(:,ibz);
filter = find(GIndex>0);

nQOrbs = size(phiR,2);
phiK = zeros(nWaves,nQOrbs);

for iorb = 1 : nQOrbs
    phiCube = reshape(phiR(:,iorb),fftGrid);
    phiCube = fftn(phiCube) * PW.Volume/nPoints;
    phiCube = reshape(phiCube, [nPoints, 1]);
    phiK(filter, iorb) = phiCube(GIndex(filter));
end


