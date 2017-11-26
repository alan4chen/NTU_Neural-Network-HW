function [reward_hat, p] = ACE(learn, decay, reward, gamma, p_before)
	% ACE : generate [improved reinforcement signal (reward_hat)]
    global v x_vec x_bar;
    
    if reward == -1
        p = 0;
    else
        p =  v' * x_vec;
    end
    
    reward_hat = reward + gamma * p - p_before;
    v = v + learn * reward_hat * x_bar;
    x_bar = decay * x_bar + (1-decay) * x_vec;
end