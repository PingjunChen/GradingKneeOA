function X = uppertri(M,N)
% UPPERTRI   Upper triagonal matrix (the elements are ones)
%            UPPER(M,N) is a M-by-N triagonal matrix

[J,I] = meshgrid(1:M,1:N);
X = (J>=I);