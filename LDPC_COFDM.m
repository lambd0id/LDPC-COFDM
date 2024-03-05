clear all

disp("LDPC COFDM Rayleigh fading (full)")

M = 4;                 % Modulation alphabet (QPSK)
K = log2(M);           % Bits/symbol
numSC = 64;           % Number of OFDM subcarriers %48 data subcarriers + 4 pilot carriers w/ data + 12 guard band carriers
cpLen = 16;            % OFDM cyclic prefix length (default=16)
fs = 100000;               % Sample rate (Hz)
delta_fT = 0.1;  %frequency offset to subcarrier frequency spacing ratio
N = numSC+cpLen;
max_iteration = 1000;

qpskMod = comm.QPSKModulator('BitInput',true);
qpskDemod = comm.QPSKDemodulator('BitOutput',true);

ofdmMod = comm.OFDMModulator('FFTLength',numSC,'CyclicPrefixLength',cpLen, ...
    'InsertDCNull', true);
ofdmDemod = comm.OFDMDemodulator('FFTLength',numSC,'CyclicPrefixLength',cpLen, ...
    'RemoveDCCarrier', true);

ofdmDims = info(ofdmMod);
numDC = ofdmDims.DataInputSize(1); % 52 data subcarriers
frameSize = [K*numDC 1]; %number of bits per packet? (n=104 for LDPC?)

AWGN_noise = comm.AWGNChannel('NoiseMethod','Variance', ...
    'VarianceSource','Input port');

rayleighchan = comm.RayleighChannel('SampleRate',fs, ...
    'RandomStream','mt19937ar with seed', ...
    'Seed',17, ...
    'FadingTechnique','Sum of sinusoids', ...
    'MaximumDopplerShift', 0, 'PathGainsOutputPort', true);

n_partitions = 1;
n = K*numDC*n_partitions;
J = 4;
k = 8;


errorRate = comm.ErrorRate('ResetInputPort',true);
maxBitErrors = 100;    % Maximum number of bit errors
maxNumBits = 1e7;      % Maximum number of bits transmitted

EbNoVec = (0:10)';
snrVec = EbNoVec + 10*log10(K) + 10*log10(numDC/numSC);
berVec = zeros(length(EbNoVec),3);
errorStats = zeros(1,3);

check_matrix = parity_check_matrix(n,J,k);
[A,T,r_swap,swap] = maltform(check_matrix);
dataInSize = size(check_matrix,2) - size(check_matrix,1) + r_swap;

disp("start of loop")
for m = 1:length(EbNoVec)
    snr = snrVec(m);
    ldpcdecoder_in = zeros(1,n);
    while errorStats(2) <= maxBitErrors && errorStats(3) <= maxNumBits
        dataIn = randi([0,1],[1 dataInSize]);              % Generate binary data
        ldpcdataIn = ldpcEncode(A,T,r_swap,swap,check_matrix,dataIn); %LDPC encoding using parity check matrix
        for i = 1:n_partitions
            ldpcdataIn_part = ldpcdataIn(1+(K*numDC*(i-1)):K*numDC*i); % each partition is 104 bits
            
            qpskTx = qpskMod(ldpcdataIn_part');                     % Apply QPSK modulation
            txSig = ofdmMod(qpskTx);                      % Apply OFDM modulation
            
            [fadedSig,pathGains] = rayleighchan(txSig);     % Pass signal through Rayleigh fading channel
            fadedSig = fft(fadedSig);
            fadedSig_ICI = zeros(N,1);  % add ICI
            for kk = 1:N
                for l = 1:N
                    fadedSig_ICI(kk) = fadedSig_ICI(kk) + fadedSig(l)*U(l,kk,N,delta_fT);
                end
            end
            fadedSig_ICI = ifft(fadedSig_ICI);
            
            powerDB = 10*log10(var(fadedSig_ICI));               % Calculate Tx signal power
            noiseVar = 10.^(0.1*(powerDB-snr));           % Calculate the noise variance
            rxSig = AWGN_noise(fadedSig_ICI,noiseVar);              % Pass the signal through a noisy channel
            
            rxSig_eq = rxSig./pathGains;                    % channel equalization to compensate for fading
            qpskRx = ofdmDemod(rxSig_eq);                    % Apply OFDM demodulation
            ldpcdecoder_in_part = qpskDemod(qpskRx);                  % Apply QPSK demodulation
            
            ldpcdecoder_in(1+(K*numDC*(i-1)):(K*numDC*i)) = ldpcdecoder_in_part;  % add back each partition
        end
        Var = var(ldpcdecoder_in);
        codeword = SPA(ldpcdecoder_in',check_matrix,max_iteration,Var);  % Decode using Sum-Product Algorithm
        dataOut = cw_to_message(r_swap,swap,check_matrix,codeword); %get message back from codeword        
        errorStats = errorRate(dataIn',dataOut',0);     % Collect error statistics
    end
    snr
    berVec(m,:) = errorStats;                         % Save BER data
    errorStats = errorRate(dataIn',dataOut',1);         % Reset the error rate calculator    
end
%berTheory_2 = berfading(EbNoVec,'psk',M,1);
%berTheory = berawgn(EbNoVec,'psk',M,'nondiff');
figure
semilogy(EbNoVec,berVec(:,1))
hold on
%semilogy(EbNoVec,berTheory)
title(['n=',num2str(n),', j=',num2str(J),', k=',num2str(k),', deltafT=',num2str(delta_fT)])
legend('Simulation','Location','Best')
xlabel('Eb/No (dB)')
ylabel('Bit Error Rate')
grid on
hold off
save('ICI_1.mat','berVec','EbNoVec');

function U_lk = U(l, k, N, delta_fT) % maybe abandon idea of making this a separate function
    temp = 1i*2*pi*(1/N)*(l-k+delta_fT); % maybe make this a matrix that depends on l and k
    n = 0:N-1;
    U_temp = exp(n*temp);
    U_lk = (1/N)*sum(U_temp);
end

function pcm = parity_check_matrix(n,j,k) %generate parity check matrix
    base_submat = zeros(n/k,n);
    pcm = zeros((j*n)/k,n);
    
    for x = 1:n/k
        base_submat(x,(x-1)*k+1:x*k) = 1;
    end
    
    pcm(1:n/k,1:n) = base_submat;
    for x = 1:j-1
        temp = base_submat(:, randperm(size(base_submat, 2)));
        pcm((x*n/k)+1:(x+1)*(n/k),1:n) = temp;
    end
end

function [A,T,r_swap,swap] = maltform(H)
    % Realized by Djamel Slimani (djmslimani@gmail.com)
    % This program allow the encoding of an LDPC code using the parity-check matrix even if this matrix is not 
    % a full rank matrix. 
    
    % Ref: The paper entitled "Modified Approximate Lower Triangular Encoding of LDPC Codes," 2015 International Conference
    % on Advances in Computer Engineering and Applications (ICACEA) IMS Engineering College, Ghaziabad, India (IEEE).
    
    % Input variables:
    % H:      Parity-check matrix.
    % u:      Information bit vector. 
    
    % Output variable:
    % codeWord:      The produced codeword for the information bit vector u.
       
    %%*******Remove all zero rows********  
    H = H(any(H,2),:);
    %%***********************************
    
    Horg = H;%%Original matrix H   
    %%**************The initial step for the algorithm  H = [A T] ***********************
    
    %%******Check the sub matrix T ********
    m = size(H,1);
    n = size(H,2);
    
    A = [];
    T = [];
    swap = [];
    
    idx1 = 0;
    for j = n-m+1:n
      if (H(1,j)==1 && j==n-m+1)
        idx1 = 1;
        break; 
      elseif (H(1,j)==1 && j>n-m+1)
            idx1 = 1;
            H(:,[n-m+1,j]) = H(:,[j,n-m+1]); 
            swap = [swap; n-m+1 j];          
            break;
      end
    end  
    %%*****************************************
    %%******Check the sub matrix A ************
    for j = 1:n-m
          if (H(1,j)==1 && idx1 == 0) 
            H(:,[j,n-m+1]) = H(:,[n-m+1,j]); 
            idx1 = 1;
            swap = [swap; j n-m+1];          
            break; 
          end
    end 
    %%******************************************
    
    %%*************************************************************************
    
    %%***************************Beginning of the algorithm******************************
    for i = 1:m
      j = n-m+i;
        if i==1
          idxOnes = find(H(:,j)==1);
          for iOnes = 1:length(idxOnes)%%Addition of rows of the matrix that have 1 in a specific column  
            if (idxOnes(iOnes)>=i+1)
              H(idxOnes(iOnes),:) = mod(H(idxOnes(iOnes),:) + H(i,:),2);
            end
          end
          %%*****Move the rows with all zeros to the end of H*****%%
             idx_zeros = find(all(H==0,2)); %Index of all zero rows.
             if(length(idx_zeros)>=1)
               H(idx_zeros,:)=[];%Remove those rows.%%This instruction don't work if H is in galois field
               H = [H; zeros(length(idx_zeros),n)];%Add all zero rows to the end of H.
             end
          %%******************************************************%%       
         else         
      %%******Check the sub matrix T ********
        idx2 = 0;
        for j2 = n-m+i:n
          if (H(i,j2)==1 && j2==n-m+i)
            idx2 = 1;
            break; 
          elseif (H(i,j2)==1 && j2>n-m+i)
                idx2 = 1;
                H(:,[n-m+i,j2]) = H(:,[j2,n-m+i]); 
                swap = [swap; n-m+i j2];          
                break;
          end
        end  
      %%*****************************************
      %%******Check the sub matrix A ************
        for j2 = 1:n-m
              if (H(i,j2)==1 && idx2 == 0) 
                H(:,[j2,n-m+i]) = H(:,[n-m+i,j2]); 
                idx2 = 1;
                swap = [swap; j2 n-m+i];         
                break; 
              end
        end 
     %*****************************************
        idxOnes = find(H(:,j)==1);
        
          for iOnes = 1:length(idxOnes)%%Addition of rows of the matrix that have 1 in a specific column 
            if (idxOnes(iOnes)>=i+1)
              H(idxOnes(iOnes),:) = mod(H(idxOnes(iOnes),:) + H(i,:),2);
            end
          end
         %%*********Move the all zero rows to the end of H********%%
             idx_zeros = find(all(H==0,2)); %Index of all zero rows.
             if(length(idx_zeros)>=1)
               H(idx_zeros,:)=[];%Remove those rows.
               H = [H; zeros(length(idx_zeros),n)];%Add all zero rows to the end of H.
             end
          %%******************************************************%%       
         
         end    
     end
  %%***********************************************************************************
  
  %%***Remove rows with all zeros from H and specify the matricies A and T*************
  idx_zeros = find(all(H==0,2)); %Index of all zero rows.
  r_swap = length(idx_zeros); 
    if r_swap>=1
      H(idx_zeros,:)=[];%Remove those rows.
      A  = [H(:,1:n-m) H(:,n-r_swap+1:n)]; 
      T = H(:,n-m+1:n-r_swap);         
    else
      H(idx_zeros,:)=[];%Remove those rows.
      A = H(:,1:n-m);
      T = H(:,n-m+1:n);      
    end
end        

function codeWord = ldpcEncode(A,T,r_swap,swap,H,u)
    m = size(H,1);
    n = size(H,2);
  %%*******************Encoding process using backward substitution********************
    a = T;
    b = mod(A * u',2); 
    %%*********************BackwardSub algorithm: is not my algorithm**************************
    l = length(b);
    y(l,1) = mod(b(l)/a(l,l),2);
    for i = l-1:-1:1
      y(i,1)=mod((b(i)-a(i,i+1:l)*y(i+1:l,1))./a(i,i),2);%  Parity bit vector.
    end
    %%***********************************************************************************
    
    c = [u y']; % The codeword
    %r_swap
    %%*****Apply the inverse permutation to the codeword(using swap and r)******
    if(r_swap>=1)
      c1 = c([1:n-m n-m+r_swap+1:n]);
      c2 = c(n-m+1:n-m+r_swap);
      c  = [c1 c2];
    end
    if (~isempty(swap))
      for i=size(swap,1):-1:1
        c(:,[swap(i,1),swap(i,2)]) = c(:,[swap(i,2),swap(i,1)]);
      end      
    end   
    %%**************************************************************************
    
  %%***********************************************************************************
    codeWord = c;
end

function message = cw_to_message(r_swap,swap,H,cw)
    m = size(H,1);
    n = size(H,2);

    if (~isempty(swap))
        for i=1:size(swap,1)
            cw(:,[swap(i,1),swap(i,2)]) = cw(:,[swap(i,2),swap(i,1)]);
        end
    end

    if(r_swap>=1) 
        c1 = cw([1:n-m n-r_swap+1:n]);
        c2 = cw(n-m+1:n-r_swap);

        cw = [c1 c2];
        message = cw(1:n-m+r_swap);
    end
end

function x_hat = SPA(codeword,check_mat,max_iter,noise_var)
    [nrow,ncol]=size(check_mat);

    %compute for Lpl
    Lpl = zeros(1,ncol);
    x_hat = zeros(1,ncol);
    
    %00 = 0.7071 + 0.7071i
    %01 = -0.7071 + 0.7071i ([0;1])
    %10 = 0.7071 - 0.7071i ([1;0])
    %11 = -0.7071 - 0.7071i
    qpskMod = comm.QPSKModulator('BitInput',true);
    qpsk_cw = qpskMod(codeword);
    
    for l = 1:ncol
        if mod(l,2) %L(p_s,2) J(1,1) = 10,11? J(1,0) = 00,01?
            xi_Jt1 = [0.7071 - 0.7071j, -0.7071 - 0.7071j];
            xi_Jt0 = [0.7071 + 0.7071j, -0.7071 + 0.7071j];
        else %L(p_s,1)  J(2,1) = 01,11? J(2,0) = 00,10?
            xi_Jt1 = [-0.7071 + 0.7071j, -0.7071 - 0.7071j];
            xi_Jt0 = [0.7071 + 0.7071j, 0.7071 - 0.7071i];
        end
        y = qpsk_cw(ceil(l/2));
        Lpl(l) = sum(exp(-((((real(y)-real(xi_Jt1)).^2)+((imag(y)-imag(xi_Jt1)).^2))/(2*noise_var))))...
        /sum(exp(-((((real(y)-real(xi_Jt0)).^2)+((imag(y)-imag(xi_Jt0)).^2))/(2*noise_var))));
    end
    Lpl = log(Lpl);
    Lqlm = (Lpl.*ones(nrow,ncol)).*check_mat; %not sure
    Lrml = zeros(nrow,ncol);
    %Lqlm
    for iter = 1:max_iter

        %checks to bits
        for m = 1:nrow
            idx = find(check_mat(m,:));
            for id = 1:length(idx)
                temp = Lqlm(m,idx);
                temp(id) = 1/0;
                Lrml(m,idx(id)) = prod(tanh(temp/2));
            end
        end
        Lrml = 2*atanh(Lrml);
        
        %bits to checks
        for l = 1:ncol
            idx = find(check_mat(:,l));
            for id = 1:length(idx)
                temp = Lrml(idx,l);
                temp(id) = 0;
                Lqlm(idx(id),l) = Lpl(l) + sum(temp);
            end
        end

        %check stop criterion
        Lql = Lpl + sum(Lrml.*check_mat);
        x_hat(Lql>=0)=1;
        x_hat(Lql<0)=0;
        
        if mod(x_hat*check_mat',2) == 0
            break
        end
        
    end
end