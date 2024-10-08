%% set up
t = readtable('test_audioMNIST.csv'); 
N_samples = 8000; 
algs = ["PhaseMax"; "AmplitudeFlow"; "WirtFlow"]; 
measures = [32, 16, 8, 4];
A_spar = load('A_sparse_32K.mat'); 
A_spar = A_spar.A_spar; 
%% start loop 
for k = 1:1
    n_measurements = measures(k); % set up num of measurements
    for j=1:1
        % set up data and measurement
        algname = algs(j);
        fprintf(strcat('\nAlgorithm: ', algname, ', N_meas: ', num2str(n_measurements)))
        folderpath = strcat(pwd, '/reconstructions_', num2str(n_measurements), 'K/', algname, '/');  
        %% now loop thru data 
        % i = 1; 
        N = size(t,1); 
        % N = 1; 
        recon_error_aligned = zeros(N,1); 
        residual_error = zeros(N,1); 
        for i=1:1
                %% get audio names and filenames 
                c = strsplit(t(i,4).Var4{1}, ','); 
                filename = c{1}; 
                d = strsplit(filename, '.'); 
                audioname = d{1}; 
                
                %% get true subsampled 
                [x_true,Fs] = audioread(filename); 
                x_sub = resample(x_true,N_samples,Fs); 
                x_pad = [x_sub; zeros(N_samples -size(x_sub,1),1)]; 
                
                %% get solved data
                data = load(strcat(folderpath, audioname, '_', algname, '.mat')); 
                x = data.x; 
                
                %% post process for metrics 
                alpha_alignment = (x_pad'*x)/abs(x_pad'*x);
                recon_error_aligned(i) = 20*log10(norm(x_pad - alpha_alignment*x)/norm(x_pad));
                spec_true = abs(A_spar*x_pad); 
                spec_recon = abs(A_spar*x); 
                residual_error(i) = 20*log10(norm(spec_true(:) - spec_recon(:))/norm(spec_true(:))); 
         end
            
        %% get final metrics
        % avg_residual_error = (1/N)*residual_error; 
        % avg_recon_error_aligned = (1/N)*r_pad__econ_error_aligned; 
        % avg_recon_error_min = (1/N)*recon_error_min; 
        
        avg_QSD = mean(recon_error_aligned); 
        std_QSD = std(recon_error_aligned); 
        avg_SC = mean(residual_error); 
        std_SC = std(residual_error); 
        
        % write to file 
        fileID = fopen('results.txt', 'a+'); 

        fprintf(fileID, ...
            strcat('\n', algname, ', ', num2str(n_measurements), '_K, QSD = ', '%8.3f (%8.3f)'), ...
            avg_QSD, std_QSD);
        fprintf(fileID, ...
            strcat('\n', algname, ', ', num2str(n_measurements), '_K, SC = ', '%8.3f (%8.3f)'), ...
            avg_SC, std_SC);
        fclose(fileID); 

        fprintf(strcat('\n Done with ', algname, ', measurements: ', num2str(n_measurements))); 
    end 
end