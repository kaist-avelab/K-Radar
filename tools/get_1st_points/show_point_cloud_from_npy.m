clc, clear, close all

% Functions readNPYheader & readNPY are from
% 'https://github.com/kwikteam/npy-matlab/blob/master/npy-matlab/readNPY.m'
% Thanks to Kwik Team

load('info_arr.mat')

inds = readNPY('ind_00033.npy');

r = arrRange(inds(:,1));
az = arrAzimuth(inds(:,2));
el = arrElevation(inds(:,3));

% Radar polar to General polar coordinate
az = -az*pi/180.;
el = -el*pi/180.;

% For flipped azimuth & elevation angle
x = (r .* cos(el) .* cos(az))';
y = (r .* cos(el) .* sin(az))';
z = (r .* sin(el))';

ptCloud = pointCloud([x, y, z]);
pcshow(ptCloud);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Point Cloud Visualization');

function [arrayShape, dataType, fortranOrder, littleEndian, totalHeaderLength, npyVersion] = readNPYheader(filename)
    % function [arrayShape, dataType, fortranOrder, littleEndian, ...
    %       totalHeaderLength, npyVersion] = readNPYheader(filename)
    %
    % parse the header of a .npy file and return all the info contained
    % therein.
    %
    % Based on spec at http://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
    
    fid = fopen(filename);
    
    % verify that the file exists
    if (fid == -1)
        if ~isempty(dir(filename))
            error('Permission denied: %s', filename);
        else
            error('File not found: %s', filename);
        end
    end
    
    try
        
        dtypesMatlab = {'uint8','uint16','uint32','uint64','int8','int16','int32','int64','single','double', 'logical'};
        dtypesNPY = {'u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8', 'f4', 'f8', 'b1'};
        
        
        magicString = fread(fid, [1 6], 'uint8=>uint8');
        
        if ~all(magicString == [147,78,85,77,80,89])
            error('readNPY:NotNUMPYFile', 'Error: This file does not appear to be NUMPY format based on the header.');
        end
        
        majorVersion = fread(fid, [1 1], 'uint8=>uint8');
        minorVersion = fread(fid, [1 1], 'uint8=>uint8');
        
        npyVersion = [majorVersion minorVersion];
        
        headerLength = fread(fid, [1 1], 'uint16=>uint16');
        
        totalHeaderLength = 10+headerLength;
        
        arrayFormat = fread(fid, [1 headerLength], 'char=>char');
        
        % to interpret the array format info, we make some fairly strict
        % assumptions about its format...
        
        r = regexp(arrayFormat, '''descr''\s*:\s*''(.*?)''', 'tokens');
        if isempty(r)
            error('Couldn''t parse array format: "%s"', arrayFormat);
        end
        dtNPY = r{1}{1};    
        
        littleEndian = ~strcmp(dtNPY(1), '>');
        
        dataType = dtypesMatlab{strcmp(dtNPY(2:3), dtypesNPY)};
            
        r = regexp(arrayFormat, '''fortran_order''\s*:\s*(\w+)', 'tokens');
        fortranOrder = strcmp(r{1}{1}, 'True');
        
        r = regexp(arrayFormat, '''shape''\s*:\s*\((.*?)\)', 'tokens');
        shapeStr = r{1}{1}; 
        arrayShape = str2num(shapeStr(shapeStr~='L'));
    
        
        fclose(fid);
        
    catch me
        fclose(fid);
        rethrow(me);
    end
end

function data = readNPY(filename)
    % Function to read NPY files into matlab.
    % *** Only reads a subset of all possible NPY files, specifically N-D arrays of certain data types.
    % See https://github.com/kwikteam/npy-matlab/blob/master/tests/npy.ipynb for
    % more.
    %
    
    [shape, dataType, fortranOrder, littleEndian, totalHeaderLength, ~] = readNPYheader(filename);
    
    if littleEndian
        fid = fopen(filename, 'r', 'l');
    else
        fid = fopen(filename, 'r', 'b');
    end
    
    try
    
        [~] = fread(fid, totalHeaderLength, 'uint8');
    
        % read the data
        data = fread(fid, prod(shape), [dataType '=>' dataType]);
    
        if length(shape)>1 && ~fortranOrder
            data = reshape(data, shape(end:-1:1));
            data = permute(data, [length(shape):-1:1]);
        elseif length(shape)>1
            data = reshape(data, shape);
        end
    
        fclose(fid);
    
    catch me
        fclose(fid);
        rethrow(me);
    end
end

