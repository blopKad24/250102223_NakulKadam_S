% NAKUL KADAM, SIGNALS, ELECTROTHON'26.

% -------------------------------------------------------------------------
%  TASK 1 — Spatial Cleansing: Median Filter : 
%  We remove the salt&pepper noise from the cursed image.


 clc; clear; close all;

% -------------------------------------------------------------------------
% 1. LOAD IMAGE : 

filename = 'cursed_schematic_02.png';                                      %change filename (image) here.

if ~isfile(filename)
    error('Image not found: "%s"\nMake sure it is in the same folder as this script.', filename);
end

img_original = imread(filename);

img_gray = double(img_original);                                           % our math is for 2D images, so we convert to grayscale even if a color pic uploaded. double() here simply converts values from integer to decimal.


img_double = img_gray / 255.0;                                             % Normalise to [0, 1], cuz PSNR, SSIM, colormap require 0to1; and convenience of rounding errors during FFT & IFFT.

fprintf('Image loaded: %d x %d pixels\n', size(img_double, 1), size(img_double, 2));



% -------------------------------------------------------------------------
% 2. MANUAL MEDIAN FILTER : 
%    We slide a square kernel over every pixel and pick the median of the
%    values. This kills the salt&pepper noise; as they stand out from their
%    neighbourings pixels.

[rows, cols] = size(img_double);
KERNEL_SIZE = 3;                                                           % 3x3 Kernel.
half         = floor(KERNEL_SIZE / 2);                                     % Deciding tiles about the central tile (1 for a 3x3, 2 for 5x5...).
img_filtered = zeros(rows, cols);                                          % Only used to initialize some values instead of leaving it floating.

% create the dummy image on which the kernel will hover (258x258).
img_padded = [img_double(half:-1:1, half:-1:1),...                         % Top left corner reflected.
              img_double(half:-1:1, :), ...                                % Top edge reflected above itself.
              img_double(half:-1:1, end:-1:end-half+1);...                 % Top right corner reflected.

              img_double(:, half:-1:1), ...                                % Left edge reflected to it's left.
              img_double,...                                               % Centre.
              img_double(:, end:-1:end-half+1);...                         % Right edge refleced to it's right.

              img_double(end:-1:end-half+1, half:-1:1), ...                % Bottom Left corner reflected.
              img_double(end:-1:end-half+1, :),...                         % Bottom edge reflected below itself.
              img_double(end:-1:end-half+1, end:-1:end-half+1)];           % Bottom Right corner reflected.

for r = 1 : rows                                                           % Sliding the kernel.
    for c = 1 : cols
        
        patch = img_padded(r : r + KERNEL_SIZE - 1, ...                    % Each 3x3 patch from img_padded is copied as 3x3 array. 
                           c : c + KERNEL_SIZE - 1);

        neighbourhood      = patch(:);                                     % Each 3x3 patch is converted to a column of 9 values.
        sorted_vals        = sort(neighbourhood);                          % The 9 values are arranged in asc order.
        median_index       = ceil(numel(sorted_vals) / 2);                 % Index of the median value is picked.
        img_filtered(r, c) = sorted_vals(median_index);                    % The central tile location is assigned with the median value.
    end
end

fprintf('Median filter done.\n');




% ---------------------------------------------------------
% 3. QUALITY METRICS : 
%    PSNR = 10 * log10(MAX^2 / MSE)   where MAX=1 for [0,1] images.
%    PSNR measures the error in decibels (higher is better).
%    SSIM measures the structural similarity. (closer to 1 means accurate).

                                                                           % PNSR calculation.
mse_val  = mean((img_double(:) - img_filtered(:)).^2);                     % MSE calculated for PSNR. 
if mse_val == 0                                                            % MSE=0 indicates identical images. Thus, PSNR->infinity (by formula).
    psnr_val = Inf;
else
    psnr_val = 10 * log10(1.0 / mse_val);
end

                                                                           % SSIM calculation.
xmu      = mean(img_double(:));                                            % average brightness of the original image.
ymu      = mean(img_filtered(:));                                          % average brightness of the filtered image.
xsig     = std(img_double(:));                                             % standard deviation of the original image.
ysig     = std(img_filtered(:));                                           % standard deviation of the filtered image.
xysig    = mean((img_double(:) - xmu) .* (img_filtered(:) - ymu));         % cross-covariance (measures how the two images vary together at the same spot. High value = they have similar patterns of light and dark).
C1       = (0.01)^2;                                                       % C1 & C2 are tiny constants that prevent having division by 0.
C2       = (0.03)^2;
ssim_val = (2*xmu*ymu + C1) * (2*xysig + C2) / ((xmu^2 + ymu^2 + C1) * (xsig^2 + ysig^2 + C2));

fprintf('\n--- Quality Metrics (filtered vs. noisy input) ---\n');
fprintf('PSNR : %.2f dB\n', psnr_val);
fprintf('SSIM : %.4f\n',    ssim_val);



%TASK2 yaha se.
% -------------------------------------------------------------------------
% 4. 2D FFT :  
% Median filtered image is converted from spatial to frequency domain.


fprintf('\nComputing 2D FFT');

F         =fft2(img_filtered);                                             % Converts from spatial to frequency domain. F is 256x256 array of complex numbers.
F_shifted =fftshift(F);                                                    % By default fft2 kept the zero freq in the left corner; fftshift will bring it to the centre.

magnitude_spectrum =log(1 + abs(F_shifted));                               % The 2D FFT magnitude spectrum. abs() gets the magnitude of the complex numbers; log helps to suppress a huge range to a small range of values.
                                                                           % Later displayed in panel 3 of the display.

% -------------------------------------------------------------------------
% 5. AUTOMATIC PEAK DETECTION : 
% DC component is moved to the centre; while the periodic grid noise
% concentrates in the big cross (+).
% We protect the DC circular region so the real image content stays.

% suppress the frequency cross
DC_PROTECT_R     = 16;                                                     % Radius about centre of DC component; the circular region needs to be protected.
CROSS_WIDTH      = 2;                                                      % Half-width of the cross.

mag = abs(F_shifted);                                                      % Magnitude of the complex numbers in F_shifted.
 
cy = floor(rows/2) + 1;                                                    % Centre coordinates. For 256x256, cx=cy=129.
cx = floor(cols/2) + 1;

[X, Y] = meshgrid(1:cols, 1:rows);                                         % meshgrid here returns two 256x256 arrays; X holds column of every position and Y holds row of every position.
dc_zone = ( (X - cx).^2 + (Y - cy).^2 ) <= ( DC_PROTECT_R^2 );             % The circle of DC component (to be protected).

freq_mask = ones(rows, cols);                                              % Start the mask with all 1s (Keep everything rn, we'll make it 0 when we detect noise).

freq_mask(:, cx-CROSS_WIDTH : cx+CROSS_WIDTH) = 0;                         % Mask out the vertical strip of the noise cross (+).

freq_mask(cy-CROSS_WIDTH : cy+CROSS_WIDTH, :) = 0;                         % Now mask out the remaining horizontal strip of the noise.

freq_mask(dc_zone) = 1;                                                    % In the process, we masked out the DC component, so we undo it by making the circular region as 1s.

 n_peaks = sum(freq_mask(:) == 0);                                          % Count of spikes detected.
 fprintf('Total frequencies masked : %d\n', n_peaks);


% -------------------------------------------------------------------------
% 6. APPLY MASK AND IFFT :
% We apply the mask and bring the image back to spatial domain using IFFT.

F_cleaned = F_shifted .* freq_mask;                                        % Multiply the freq spectrum by the mask; pixels with 1 stays, while picels with 0 get eliminated.
F_ishift =ifftshift(F_cleaned);                                            % The DC component is shifted back to it's original place.
img_restored =real(ifft2(F_ishift));                                       % Only real parts of the complex numbers is picked out.

img_restored =max(0, min(1, img_restored));                                % img_restored values capped 0 to 1 : min(1, ...) ensures maximum we get is 1. max(0, ...) ensures minimum we get is 0.
                                                                           % [This step is just for safety. For our project, the result remains the same event without this step].
fprintf('Inverse FFT done. Image restored. \n');


% -------------------------------------------------------------------------
% 7. BLUEPRINT :
% We build a custom 256 color RGB table that continously maps brightness to blue-shades. 
% Rough distribution can be :
% Dark pixels   ->   0 to 0.3 (1    to ~78)
% Mid picels    -> 0.3 to 0.7 (~79  to ~180)
% Bright pixels -> 0.7 to 1.0 (~181 to 256)

n_colors =256;
t=linspace(0, 1, n_colors);                                                % linspace creates 1x256 (evenly spaced numbers from 0 to 1). 
                                                                           % Changing the below given coefficients will only result in different shades of blue (may shrink the range of blue as well).
rbp = 0.05*t;                                                              % red part of the color. Stays near 0 for blue.
gbp = 0.4 * t.^1.5;                                                        % green part of the color. 
bbp = 0.3 + 0.7*t;                                                         % blue part of the color. 

blueprint_cmap= [rbp, gbp, bbp];                                           % this creates 1x768.
blueprint_cmap= max(0, min(1, blueprint_cmap));                            % min(1,...) ensures maximum is 1; 
                                                                           % capping at 1 was valid, but I didnt want unexpected errors of having <0 so I added max(0,...) so that minimum is 0.
blueprint_cmap = reshape(blueprint_cmap, [], 3);                           % makes it 256x3. Each row is a RGB color. It gets lighter as we go top to bottom. 
idx = max(1, min(256, round(img_restored * 255) + 1));                     % just an intermediate image, it's values will tell the RGB row number it needs. img_restored is 0to1; 
                                                                           % while we need 1to256. *255 and round() to get 0to255; add 1 to make it 1to256.
                                                                           % if the round() step exceeds 256, min(256,...) handles it. 
                                                                           % max(0,...) handles if negative number appears. (idk how it's possible, but looking at max(1,...) I thought it'll be safe).

blueprint_img = reshape(blueprint_cmap(idx(:), :), rows, cols, 3);         % % idx(:) flattens 256x256 to 65536x1. blueprint_cmap(...) fetches the RGB color as per row's number.
                                                                           % reshape() folds back → 256x256x3.

% -------------------------------------------------------------------------
% 8. OUTPUTS PANEL :
% display the outputs of the project.
% Original Corrupted image, After Spatial filtering, 2D FFT magnitude
% spectrum, Frequency mask, Final restored image (blueprint)

figure('Name', 'Full Restoration pipeline', ...
       'NumberTitle', 'off', ...
       'Units', 'normalized', ...
       'OuterPosition', [0 0 1 1], ...
       'Color', [0.08 0.08 0.12]);

ts={'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.9 0.9 0.9]};         % Pre-wrote the settings for title for every output instead of writing again n again.
xs={'FontSize', 10, 'Color', [0.7 0.7 0.7]};                               % Pre-wrote the settings for xlabel() for every output instead of writing again n again


subplot(2, 3, 1);                                                          % OG image
imshow(img_double, [0,1]);
title('1. Original corrupted image', ts{:});
xlabel('Salt & pepper + Grid noise', xs{:});


subplot(2, 3, 2)                                                           % After Median Filter
imshow(img_filtered, [0 1]);
title('2. After Spatial filtering', ts{:});
xlabel(sprintf('PSNR : %.2f dB | SSIM : %.4f', psnr_val, ssim_val), xs{:});


subplot(2, 3, 4);                                                          % FFT magnitude spectrum
imshow(magnitude_spectrum, []);
colormap(gca, jet);
title('3. 2D FFT Magnitude Spectrum',  ts{:});
xlabel('Centre = DC  |  Bright spikes = grid noise', xs{:});


subplot(2, 3, 5);                                                          % Frequency mask.
imshow(freq_mask, [0 1]);
title('4. Frequency Mask', ts{:});
xlabel(sprintf('White = kept  |  Black = zeroed  |  %d peaks masked', n_peaks), xs{:});
 

subplot(2, 3, 6);                                                          % The blueprint image.
imshow(blueprint_img);

title('5. Final Restored image (Blueprint)',   ts{:});
xlabel('Frequency noise eliminated',   xs{:});

sgtitle('Image Restoration Pipeline | Task1:Spatial + Task2:Frequency', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.0 0.9 1.0]);

% -------------------------------------------------------------------------
% 9. SAVE OUTPUT :

 imwrite(blueprint_img, 'task_blueprint.png');
 
 saveas(gcf, 'task_result.png');
 
 fprintf('\nOutputs saved:\n');
 fprintf('  task_blueprint.png  — blueprint colourised version\n');
 fprintf('  task_result.png     — full 5-panel figure\n');
 fprintf('\nPipeline complete.\n');

% -------------------------------------------------------------------------

