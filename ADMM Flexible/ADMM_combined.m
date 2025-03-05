clear all; clc; close all;
% List of models
models = {'RelU', 'CSF', 'MinMax', 'Modulus'};

% List of norms
norms = {'Frobenius', 'L1', 'KL Divergence'};

% Prompt for model selection
disp('Select a Nonlinear Matrix Decomposition Model:');
for i = 1:length(models)
    fprintf('%d: %s\n', i, models{i});
end
model_choice = input('Enter the number corresponding to your choice: ');

% Validate model choice
if model_choice < 1 || model_choice > length(models)
    error('Invalid model choice. Please restart and select a valid option.');
end

% Prompt for norm selection
disp('Select an Error Norm:');
for i = 1:length(norms)
    fprintf('%d: %s\n', i, norms{i});
end
norm_choice = input('Enter the number corresponding to your choice: ');

% Validate norm choice
if norm_choice < 1 || norm_choice > length(norms)
    error('Invalid norm choice. Please restart and select a valid option.');
end


% Execute the selected model and norm
switch model_choice
    case 1
        disp(['Running ', models{1}, ' with ', norms{norm_choice}, ' norm.']);
        % Call your function or algorithm here
        run_ReLU(norm_choice);
        
    case 2
        disp(['Running ', models{2}, ' with ', norms{norm_choice}, ' norm.']);
        run_CSF(norm_choice);
        
    case 3
        disp(['Running ', models{3}, ' with ', norms{norm_choice}, ' norm.']);
        run_MinMax(norm_choice);
        
    case 4
        disp(['Running ', models{4}, ' with ', norms{norm_choice}, ' norm.']);
        run_Modulus(norm_choice);
end




function run_ReLU(norm_choice)
 
 
    switch norm_choice
        case 1
            disp('Applying Frobenius norm...');
            run relu_frobenius.m
           
        case 2
            disp('Applying L1 norm...');
            run relu_L1.m
        
            
        case 3
            disp('Applying KL Divergence...');
            run relu_KLdivergence.m
    end
end


function run_CSF(norm_choice)

    
    
    switch norm_choice
        case 1
            disp('Applying Frobenius norm...');
            run csf_frobenius.m
        case 2
            disp('Applying L1 norm...');
            run csf_L1.m
        case 3
            disp('Applying KL Divergence...');
            run csf_KLdivergence.m
    end
end


function run_MinMax(norm_choice)

 

    switch norm_choice
        case 1
            disp('Applying Frobenius norm...');
            run minmax_frobenius.m
        case 2
            disp('Applying L1 norm...');
            run minmax_L1.m
        case 3
            disp('Applying KL Divergence...');
            run minmax_KLdivergence.m
    end
end

function run_Modulus(norm_choice)

    switch norm_choice
        case 1
            disp('Applying Frobenius norm...');
            run modulus_frobenius.m %rho=5
        case 2
            disp('Applying L1 norm...');
            run modulus_L1.m  % rho=10
        case 3
            disp('Applying KL Divergence...');
            run modulus_KLdivergence.m
    end
end