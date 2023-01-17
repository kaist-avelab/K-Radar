%% Processing radar tessurect
% arrDREA (4D Tesseract)
restoredefaultpath
addpath(genpath(pwd))

setUserDefinedParams;
setRadarParams;
srsAmanda(hwCfg, radarCfg, aoaCfg, 'init');

filePath = strcat(pathBaseDir, nameFolderRadarBin);
fileName = strcat(filePath, cellNameFiles{idxFile}, '_');

%% numFrames: Frame 개수
minFileSize = Inf;
for chipIdx = 1:hwCfg.numChips
    currentFileName = strcat(fileName, num2str(chipIdx), nameTailRadarBin);
    currentFileInfo = dir(currentFileName);
    currentFileSize = currentFileInfo.bytes;
    if (currentFileSize < minFileSize)
        minFileSize = currentFileSize;
    end
end
numFrames = floor(minFileSize/frameSizeInByte);
radarCube = single(zeros(radarCfg.numAdcSamples, hwCfg.numRxAntsPerChip, radarCfg.numChirpsPerLoop, radarCfg.numChirpLoops, hwCfg.numChips));

%% Iterate per frames
fprintf('Total frames = %d ...\n', numFrames)
cellPathRadarFiles = {};
for frameIdx = 1:numFrames
    fprintf('frameIdx = %d is being processed ...\n', frameIdx)
    for chipIdx = 1:hwCfg.numChips
        adcData = readAdcData(frameSizeInByte, fileName, chipIdx, frameIdx);             
        radarCube(:, :, :, :, chipIdx) = reshape(adcData(1:2:end), radarCfg.numAdcSamples, ...
                                                                   hwCfg.numRxAntsPerChip, ...
                                                                   radarCfg.numChirpsPerLoop, ...
                                                                   radarCfg.numChirpLoops) ...
                                       + 1j* ...
                                         reshape(adcData(2:2:end), radarCfg.numAdcSamples, ...
                                                                   hwCfg.numRxAntsPerChip, ...
                                                                   radarCfg.numChirpsPerLoop, ...
                                                                   radarCfg.numChirpLoops); 
    end
    radarCube = srsAmanda(hwCfg, radarCfg, aoaCfg, 'calib', radarCube); % calibration % Please use own function to generate radar data (We do not provide this function)
    rangeFFTout = fft(radarCube, dspCfg.numRangeBins, 1); % range FFT (w/o windowing)
    dopplerFFTout = fft(rangeFFTout, dspCfg.numDopplerBins, 4); % doppler FFT (w/o windowing)
    sizeRadarCube = size(dopplerFFTout); % Doppler-Range-Elevation-Azimuth Tensor (Tessurect)
    binRange = sizeRadarCube(1);
    binDoppler = sizeRadarCube(4);
    
    arrDREA = zeros(binDoppler,binRange,37,107,'single');
    for idxDoppler = 1:binDoppler
        for idxRange = 1:binRange
            aoaInput = squeeze(dopplerFFTout(idxRange, :, :, idxDoppler, :));
            arrDREA(idxDoppler,idxRange,:,:) = srsAmanda(hwCfg, radarCfg, aoaCfg, 'dbf', aoaInput, idxDoppler-1, dspCfg);
        end
    end
    % arrDREA = abs(arrDREA);
    arrDREA = double(arrDREA);
    
    % flip along elevation
    arrDREA = flip(arrDREA,3);

    nameMatFile = strcat('tesseract_', num2str(frameIdx, '%05.f'), '.mat');
    pathMatFile = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\radar_tesseract\', nameMatFile);
    cellPathRadarFiles{end+1} = pathMatFile;
    save(pathMatFile, 'arrDREA')
end

%%
nameMatFile = strcat(cellNameFiles{idxFile}, '_', 'cell_DREA', '.mat');
pathMatFile = strcat(pathBaseDir, 'generated_files\', cellNameFiles{idxFile}, '\cell_path\', nameMatFile);
save(pathMatFile, 'cellPathRadarFiles')
