function [reward_hat, p] = ACE(learn, decay, reward)
	% ASE : generate [action]
    global w x_vec e BETA;
    % BETA: magnitude of noise added to choice 

    noise = BETA * rand();
    z = w' * x_vec + noise;

    y = 2 % right
    if z < 0
        y = 1;
    end

    w = w + learn * reward * e;
    e = decay * e + (1-decay) * y * x_vec;


end