function [FD_x FD_y] = fourier_descriptor(sX, sY, M)

Z = sX + sqrt(-1)*sY;

N = length(Z);
if rem(N, 2) ~= 0
    Z(end) = [];
    N = N - 1;
end

if M > N/2
    M = N/2;
end

C = zeros(N, 1);
for k = (-N/2 + 1):N/2
    C(k + N/2) = 0;
    for n = 1:N
        tmp = Z(n)*exp(-2*pi*sqrt(-1)*k*n/N);
        C(k + N/2) = C(k + N/2) + tmp;
    end
    C(k + N/2) = C(k + N/2)/N;
end

Z_reconstruct = zeros(N, 1);
for n = 1:N
    Z_reconstruct(n) = 0;
    for k = (-M + 1):M
        tmp = C(k + N/2)*exp(2*pi*sqrt(-1)*k*n/N);
        Z_reconstruct(n) = Z_reconstruct(n) + tmp;
    end
end

FD_x = real(Z_reconstruct);
FD_y = imag(Z_reconstruct);

end

